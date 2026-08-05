"""Error feedback is part of the contract: failures name the cause and the fix.

Every assertion here pins a message a rel-studio operator will actually see,
so a refactor that turns a precise error into a vague one fails the suite.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import pytest
from conftest import StubBackend, churn_rows, dt, in_memory_wiring

from relativedb import (
    Engine,
    ExecutionInput,
    LinkDef,
    MissingParameterError,
    RelqlValidationError,
    Schema,
    TableDef,
    ValueType,
    parse,
    validate,
)
from relativedb.errors import ExecutionError

ANCHOR = dt("2026-07-01")
CHURN = ("PREDICT NOT EXISTS(orders.*) OVER (90 DAYS FOLLOWING) "
         "FROM customers WHERE customers.customer_id IN :ids")


def _schema() -> Schema:
    return (Schema.new_schema()
            .table(TableDef.new_table("customers")
                   .column("age", ValueType.NUMBER)
                   .primary_key("customer_id").build())
            .table(TableDef.new_table("orders")
                   .column("qty", ValueType.NUMBER)
                   .column("order_date", ValueType.DATETIME)
                   .primary_key("order_id").time_column("order_date").build())
            .link(LinkDef("orders", "customer_id", "customers"))
            .build())


def _engine() -> Engine:
    rows = {k: v for k, v in churn_rows().items()
            if k in ("customers", "orders")}
    return Engine(_schema(), in_memory_wiring(rows),
                  model_backend=StubBackend())


def test_shared_context_names_the_traversal_and_the_fix(recwarn):
    engine = _engine()   # default BreadthFirstTraversal
    with pytest.raises(ExecutionError) as excinfo:
        engine.execute(ExecutionInput(
            query=CHURN, params={"ids": ["C7"]}, anchor_time=ANCHOR,
            shared_context=True))
    message = str(excinfo.value)
    assert "BreadthFirstTraversal" in message
    assert "ReferenceTraversal" in message
    assert "shared_context=True" in message
    # The old behavior warned about a failed chunk-sizing probe before
    # raising; the failure must arrive clean.
    assert not [w for w in recwarn.list
                if "chunk sizing" in str(w.message)]


def test_unconsumed_ids_binding_names_the_exact_where_clause():
    from relativedb import UnreferencedParameterError

    engine = _engine()
    with pytest.raises(UnreferencedParameterError) as excinfo:
        engine.execute(ExecutionInput(
            query=("PREDICT NOT EXISTS(orders.*) OVER (90 DAYS FOLLOWING) "
                   "FROM customers"),
            params={"ids": ["C7"]}, anchor_time=ANCHOR))
    message = str(excinfo.value)
    assert "never references" in message
    assert "WHERE <table>.<pk> IN :ids" in message


def test_missing_parameter_is_named():
    engine = _engine()
    with pytest.raises(MissingParameterError, match="ids"):
        engine.execute(ExecutionInput(query=CHURN, anchor_time=ANCHOR))


def test_return_kind_incompatibility_names_kind_and_task():
    query = ("PREDICT SUM(orders.qty) OVER (30 DAYS FOLLOWING) "
             "FROM customers RETURN PROBABILITY")
    with pytest.raises(RelqlValidationError) as excinfo:
        validate(parse(query), _schema())
    message = str(excinfo.value)
    assert "PROBABILITY" in message
    assert "regression" in message.lower()


def test_unknown_columns_are_named_where_columns_must_exist():
    with pytest.raises(RelqlValidationError,
                       match="unknown column 'wrong_col' on table 'orders'"):
        validate(parse("PREDICT SUM(orders.wrong_col) OVER "
                       "(30 DAYS FOLLOWING) FROM customers"), _schema())
    with pytest.raises(RelqlValidationError, match="unknown table 'ordrs'"):
        validate(parse("PREDICT COUNT(ordrs.*) OVER (30 DAYS FOLLOWING) "
                       "FROM customers"), _schema())


def test_a_bare_direct_target_may_be_virtual():
    """A direct target need not be a stored column: it names the task the
    zero-shot model is asked about. A typo here therefore validates —
    it becomes a question the model answers from context alone."""
    vq = validate(parse("PREDICT customers.plan FROM customers"), _schema())
    assert vq.task_type is not None


def test_empty_context_warns_by_default_and_raises_under_strict(monkeypatch):
    engine = _engine()
    request = ExecutionInput(query=CHURN, params={"ids": ["C404"]},
                             anchor_time=ANCHOR)
    monkeypatch.delenv("RELATIVEDB_STRICT", raising=False)
    with pytest.warns(UserWarning, match="C404.*EMPTY context"):
        engine.execute(request)

    monkeypatch.setenv("RELATIVEDB_STRICT", "1")
    with pytest.raises(ExecutionError, match="C404"):
        engine.execute(request)


def test_mismatched_head_warns_once_and_scores_zero_shot():
    from relativedb.relql.ast import TaskType
    from relativedb.scoring import SequenceBackend

    backend = object.__new__(SequenceBackend)
    backend.head = SimpleNamespace(task="ranking")
    with pytest.warns(UserWarning, match="NOT used"):
        assert backend._head_for(TaskType.BINARY_CLASSIFICATION) is None
    with warnings.catch_warnings():
        warnings.simplefilter("error")           # a second warning would raise
        assert backend._head_for(TaskType.BINARY_CLASSIFICATION) is None
    # the matching task still gets the head
    assert backend._head_for(TaskType.MULTILABEL_RANKING) is backend.head


def test_onnx_backend_requires_an_exported_model():
    from relativedb.rt.scorer import RelationalScorer
    from relativedb.scoring import ScoringError

    scorer = object.__new__(RelationalScorer)
    scorer.inference_backend = "onnx"
    scorer.onnx_model_path = None
    scorer._relational_models = {}
    with pytest.raises(ScoringError, match="onnx_model_path"):
        scorer._relational_model_for("/checkpoint", "onnx", 0)
