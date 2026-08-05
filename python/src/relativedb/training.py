"""Task-head adaptation: fit a head on labelled anchors, or finetune.

Extracted from :mod:`relativedb.engine`, where these four functions were a
third of the class and shared nothing with query execution beyond the schema,
the wiring and the backend. They take the engine as their first argument and
are re-exported as ``Engine.fit_head`` / ``Engine.finetune``, so the public API
is unchanged.
"""
from __future__ import annotations

import json
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

import numpy as np

from .errors import ExecutionError
from .evaluate import eval_bool, eval_value
from .relql.ast import ParsedQuery, TaskType
from .relql.parser import parse, validate
from .retrieve import TemporalBound

if TYPE_CHECKING:  # engine imports this module, so these stay annotation-only
    from .engine import EntityContext

__all__ = ["finetune", "fit_head"]


def _execution_input(**kwargs):
    """Deferred import: relativedb.engine imports this module at load time, so
    ExecutionInput cannot be imported at module scope without a cycle."""
    from .engine import ExecutionInput
    return ExecutionInput(**kwargs)


def finetune(engine, query: Union[str, ParsedQuery], anchors: Sequence[datetime],
             *, output_dir: Union[str, os.PathLike] = "finetuned_model",
             entity_ids: Optional[Sequence[Any]] = None,
             params: Optional[dict[str, Any]] = None,
             labels: Optional[dict] = None,
             epochs: int = 1, batch_size: int = 1,
             learning_rate: float = 1e-5,
             weight_decay: float = 1e-2,
             grad_clip_norm: float = 1.0,
             model_uri: Optional[str] = None):
    """Fine-tune the complete RT-J checkpoint with torch.

    Unlike :meth:`fit_head`, this differentiates through all transformer
    blocks, encoders, learned masks, normalization scales, and the number
    decoder. It currently supports scalar binary/regression tasks, whose
    labels use the reference bool-as-number/Huber training contract.
    """
    from .model import NormalizationMode
    from .scoring import ColumnStats
    backend = engine._require_backend()
    if not hasattr(backend, "_collate_native"):
        raise ExecutionError(
            "full-backbone fine-tuning requires RtBackend")
    from .rt import FineTunedCheckpoint, resolve_model_path
    if not anchors:
        raise ExecutionError("full-model fine-tuning needs at least one anchor")
    if epochs <= 0 or batch_size <= 0:
        raise ExecutionError("epochs and batch_size must be positive")
    anchors = [engine._coerce_anchor(a) for a in anchors]
    pq = parse(query) if isinstance(query, str) else query
    pq = validate(pq, engine.schema).query.bind_params(params)
    task_type = pq.task_type(engine.schema)
    if task_type not in (TaskType.BINARY_CLASSIFICATION,
                         TaskType.REGRESSION):
        raise ExecutionError(
            "native full-model fine-tuning currently supports binary "
            "classification and regression; use fit_head for multiclass/ranking")
    model_uri = model_uri or engine.model_config.model_uri_for(task_type)
    normalization_mode = backend._mode(engine.model_config)
    task_spec = backend.task_spec(pq, task_type)
    if normalization_mode is NormalizationMode.REFERENCE:
        backend.column_stats = ColumnStats.fit(
            engine.schema, engine.wiring,
            TemporalBound.at_or_before(max(anchors)))

    examples: list[tuple[EntityContext, float]] = []
    entity_table = pq.entity_key.table
    dropped = 0
    for anchor in anchors:
        ids = (list(entity_ids) if entity_ids is not None else
               engine._resolve_entity_ids(
                   pq, _execution_input(query=pq, anchor_time=anchor)))
        for eid in ids:
            if labels is not None and (eid, anchor) not in labels:
                continue
            # features see only what was knowable at the anchor; the label
            # is a fact fetched from the database (engine._label_rows), not
            # read off an assembled context sample.
            ctx = engine.assemble_context(entity_table, eid, anchor, query=pq)
            y = _scalar_label(engine, pq, task_type, anchor, labels, eid, [])
            if y is not None:
                examples.append((ctx, float(y)))
            else:
                dropped += 1
    if dropped:
        warnings.warn(
            f"fine-tuning dropped {dropped} example(s) whose derived label "
            f"was NULL (empty target window or non-numeric outcome)",
            UserWarning, stacklevel=2)
    if not examples:
        raise ExecutionError(
            "full-model fine-tuning produced no training examples")
    if normalization_mode is NormalizationMode.REFERENCE:
        backend.column_stats = backend.column_stats.with_task_values(
            task_spec, [y for _, y in examples])

    seqs = []
    for ctx, y in examples:
        one, mus, sds = backend._build_sequences(
            pq, task_type, [ctx], normalization_mode=normalization_mode,
            task_spec=task_spec)
        seq = one[0]
        target = (y - mus[0]) / sds[0]
        for i, is_target in enumerate(seq.is_tgt):
            if is_target:
                seq.value[i] = target
        seqs.append(seq)

    import time

    from relational_transformers import (RelationalExample as _Example,
                                         RelationalTrainer,
                                         RelationalTrainingArguments,
                                         RelationalTransformer)

    from .rt.scorer import RelationalScorer

    source_path = resolve_model_path(model_uri)
    # An independent mutable instance: the backend's serving cache must not
    # receive gradient updates.
    transformer = RelationalTransformer(source_path)

    # One RelationalBatch per example. The normalized label was written into
    # the target cell's number channel; the model never reads it there
    # (target values are masked), so it doubles as the training label.
    train_examples = []
    for seq in seqs:
        batch = RelationalScorer._relational_batch(backend._collate_native([seq]))
        label = float((batch.number_values.squeeze(-1)
                       * batch.is_targets).sum())
        train_examples.append(_Example(batch, label))

    out = Path(output_dir).expanduser().resolve()
    # problem_type="regression" keeps the reference bool-as-number/Huber
    # training contract for binary tasks as well; labels are already
    # normalized scalars.
    trainer = RelationalTrainer(
        model=transformer,
        args=RelationalTrainingArguments(
            output_dir=str(out),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_grad_norm=grad_clip_norm,
            save_strategy="none",
        ),
        train_dataset=train_examples,
        problem_type="regression",
    )
    started = time.perf_counter()
    try:
        trainer.train()
    except RuntimeError as exc:
        raise ExecutionError(str(exc)) from exc
    total_seconds = time.perf_counter() - started
    losses = list(trainer.losses)

    out.mkdir(parents=True, exist_ok=True)
    transformer.save_pretrained(out)
    source_config = Path(source_path).with_name("config.json")
    config = (json.loads(source_config.read_text())
              if source_config.exists() else {})
    config["checkpoint_file"] = "model.safetensors"
    config["finetune"] = {
        "backend": "torch", "full_model": True,
        "source_model": model_uri, "steps": len(losses),
        "examples": len(examples), "final_loss": losses[-1],
        "normalization_mode": normalization_mode.value,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return FineTunedCheckpoint(
        out, tuple(losses), (), total_seconds,
        len(examples), len(losses), backend.column_stats,
        normalization_mode)

def fit_head(engine, query: Union[str, ParsedQuery], anchors: Sequence[datetime],
             *, entity_ids: Optional[Sequence[Any]] = None,
             params: Optional[dict[str, Any]] = None,
             labels: Optional[dict] = None,
             epochs: int = 100, learning_rate: float = 1e-3,
             weight_decay: float = 1e-4,
             model_uri: Optional[str] = None):
    """Fit a task head for ``query`` over the frozen backbone.

    The transformer is not updated; each training example is encoded once
    into its target-cell state and a small head is fitted on those. Returns
    a :class:`~relativedb.rt.FineTunedHead` — inspect its losses,
    ``save(path)`` it, and serve it by passing ``head=`` to
    :class:`~relativedb.rt.RtBackend`.

    ``anchors`` are past cut-off times. For each one the context is bounded
    at the anchor — exactly as at prediction time — while the **label** is
    read from what actually happened in the target's window after it. So
    the query defines its own supervision and no labels need supplying.

    Pass ``labels`` to override that: ``{(entity_id, anchor): value}``, or
    for ranking ``{(entity_id, anchor): {candidate_id: relevance}}``. When
    given, it also *selects* the examples — a pair it does not name is
    skipped rather than derived — so passing every row's own timestamp as
    an anchor and labelling only the diagonal trains each entity at its own
    cut-off instead of at a cut-off shared with rows it predates.
    """
    backend = engine._require_backend()
    if not hasattr(backend, "fit_head"):
        raise ExecutionError(
            "task-head fitting requires the native RT backend (RtNativeBackend)")
    if not anchors:
        raise ExecutionError("task-head fitting needs at least one anchor")
    # Row timestamps are UTC-aware; naive anchors would fail to compare.
    anchors = [engine._coerce_anchor(a) for a in anchors]

    pq = parse(query) if isinstance(query, str) else query
    pq = validate(pq, engine.schema).query.bind_params(params)
    task_type = pq.task_type(engine.schema)
    if task_type not in (TaskType.MULTICLASS_CLASSIFICATION,
                         TaskType.MULTILABEL_RANKING):
        raise ExecutionError(
            "frozen task-head fitting is limited to multiclass and "
            "multilabel-ranking adapters; scalar binary/regression tasks "
            "require full-backbone fine-tuning")
    from .rt import FineTunedHead, RtNativeError  # noqa: F401
    model_uri = model_uri or engine.model_config.model_uri_for(task_type)
    model = backend._model_for(model_uri)

    from .model import NormalizationMode
    from .scoring import ColumnStats
    normalization_mode = backend._mode(engine.model_config)
    if normalization_mode is NormalizationMode.REFERENCE:
        # Reference preprocessing is fitted only on rows knowable during
        # training. Target stats are added after labels are collected.
        backend.column_stats = ColumnStats.fit(
            engine.schema, engine.wiring,
            TemporalBound.at_or_before(max(anchors)))

    entity_table = pq.entity_key.table
    ys: list[float] = []
    groups: list[int] = [0]
    classes: list[Any] = []
    skipped = 0
    skipped_scalar = 0
    scalar_examples: list[tuple[EntityContext, float]] = []
    ranking_examples: list[tuple[EntityContext, str, list, list[float]]] = []

    for t in anchors:
        ids = (list(entity_ids) if entity_ids is not None
               else engine._resolve_entity_ids(
                   pq, _execution_input(query=pq, anchor_time=t)))
        for eid in ids:
            # A supplied ``labels`` dict IS the training set: it names the
            # (entity, anchor) pairs that are examples, and pairs it does
            # not name are not examples. That is what lets every row carry
            # its own anchor -- pass each row's timestamp and label the
            # diagonal -- which is the shape a RelBench train table has.
            # Without it a shared anchor either hides the entities after it
            # or shows the ones before it their own outcome.
            if labels is not None and (eid, t) not in labels:
                continue
            # features see only what was knowable at the anchor; the label
            # is a FACT and is derived from the database (engine._label_rows),
            # never from an assembled context — a context is a sample that
            # carries peer entities' rows and truncates the entity's own.
            ctx = engine.assemble_context(entity_table, eid, t, query=pq)
            if task_type is TaskType.MULTILABEL_RANKING:
                parent = backend.ranking_parent_table(pq)
                cands = backend._rank_candidates(
                    parent, TemporalBound.at_or_before(t) if t
                    else TemporalBound.unbounded())
                if not cands:
                    continue
                rel = _ranking_relevance(engine, pq, t, cands, labels, eid)
                if not any(r > 0 for r in rel):
                    # listwise cross-entropy needs a positive in the group;
                    # an entity that interacted with nothing in the window
                    # carries no ranking signal, so it is not an example.
                    skipped += 1
                    continue
                ranking_examples.append((ctx, parent, cands, rel))
                ys.extend(rel)
                groups.append(len(ys))
            else:
                y = _scalar_label(engine, pq, task_type, t, labels, eid,
                                  classes)
                if y is None:
                    skipped_scalar += 1
                    continue
                scalar_examples.append((ctx, float(y)))
                ys.append(float(y))

    if not ys:
        extra = (f" ({skipped} ranking groups had no positive relevance in "
                 f"the target window)" if skipped else "")
        raise ExecutionError(
            f"task-head fitting produced no training examples — check the anchors "
            f"and that the cohort resolves at them{extra}")
    if skipped:
        warnings.warn(
            f"task-head fitting skipped {skipped} ranking group(s) with no "
            f"positive relevance in the target window; listwise loss needs "
            f"at least one relevant candidate per group",
            UserWarning, stacklevel=2)
    if skipped_scalar:
        warnings.warn(
            f"task-head fitting dropped {skipped_scalar} example(s) whose "
            f"derived label was NULL (empty target window or non-numeric "
            f"outcome); the training set is biased toward entities with "
            f"history", UserWarning, stacklevel=2)
    task_spec = backend.task_spec(pq, task_type)
    if normalization_mode is NormalizationMode.REFERENCE:
        # A derived target is materialized as a real task column by the
        # reference pipeline, and its transform is persisted alongside
        # physical column transforms. Ranking relevance is binary and
        # deliberately keeps the identity scale.
        task_values = ([y for _, y in scalar_examples]
                       if scalar_examples else [0.0, 1.0])
        backend.column_stats = backend.column_stats.with_task_values(
            task_spec, task_values)

    feats: list = []
    for ctx, _ in scalar_examples:
        seqs, _, _ = backend._build_sequences(
            pq, task_type, [ctx], normalization_mode=normalization_mode,
            task_spec=task_spec)
        feats.append(backend._encode(seqs, model_uri))
    for ctx, parent, cands, _ in ranking_examples:
        seqs = backend.candidate_seqs(
            pq, ctx, parent, cands,
            normalization_mode=normalization_mode)
        feats.append(backend._encode(seqs, model_uri))
    features = np.concatenate(feats, axis=0).astype(np.float32)
    y = np.asarray(ys, np.float32)
    n_outputs = len(classes) if task_type is TaskType.MULTICLASS_CLASSIFICATION else 1
    if task_type is TaskType.MULTICLASS_CLASSIFICATION and n_outputs < 2:
        raise ExecutionError(
            f"multiclass task-head fitting needs at least two observed classes, "
            f"saw {n_outputs}")
    group_off = (np.asarray(groups, np.int32)
                 if task_type is TaskType.MULTILABEL_RANKING
                 else np.zeros(1, np.int32))
    n_groups = (len(groups) - 1
                if task_type is TaskType.MULTILABEL_RANKING else 0)
    return backend.fit_head(
        model, task_type, features, y, group_off, n_groups,
        epochs=epochs, learning_rate=learning_rate,
        weight_decay=weight_decay, classes=classes,
        normalization_mode=normalization_mode)

def _scalar_label(engine, pq, task_type, t, labels, eid, classes):
    """The outcome the query asks about, as it actually turned out.

    "Actually" is load-bearing: the rows come from the database via
    ``engine._label_rows``, entity-scoped and bounded at the window's far
    edge — not from an assembled context, whose peer rows used to count
    into this entity's label and whose fanout caps under-counted it."""
    entity_table = pq.entity_key.table
    if labels is not None and (eid, t) in labels:
        v = labels[(eid, t)]
    else:
        rows, cells = engine._label_rows(pq, entity_table, eid, t)
        if task_type is TaskType.BINARY_CLASSIFICATION:
            v = 1.0 if eval_bool(pq.target, rows, cells, t,
                                 entity_table=entity_table) else 0.0
        else:
            v = eval_value(pq.target, rows, cells, t,
                           entity_table=entity_table)
    if task_type is TaskType.MULTICLASS_CLASSIFICATION:
        if v is None:
            return None
        if v not in classes:
            classes.append(v)
        return float(classes.index(v))
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if not isinstance(v, (int, float)):
        return None
    return float(v)

def _ranking_relevance(engine, pq, t, candidates, labels, eid):
    """Per-candidate relevance: which candidate ids the ENTITY actually
    touched in the target's window — from the database, not the context
    sample (peer interactions used to mark candidates relevant that this
    entity never touched)."""
    entity_table = pq.entity_key.table
    if labels is not None and (eid, t) in labels:
        given = labels[(eid, t)] or {}
        return [float(given.get(c.id, 0.0)) for c in candidates]
    rows, cells = engine._label_rows(pq, entity_table, eid, t)
    observed = eval_value(pq.target, rows, cells, t,
                          entity_table=entity_table)
    seen = {str(x) for x in (observed or [])}
    return [1.0 if str(c.id) in seen else 0.0 for c in candidates]
