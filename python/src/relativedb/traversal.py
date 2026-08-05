"""Context construction, bound to RelQL: traversal strategies and the
columnar store, re-exported from relational-transformers-utils.

The strategies, the :class:`ContextPolicy`, and the deterministic sampling
they draw from live in :mod:`relational_transformers_utils.traversal`; this
module binds them to RelQL by supplying the task adapter that derives task
specs and self-label values from a parsed query.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from relational_transformers_utils.columnar import \
    ColumnarStore as _ColumnarStore
from relational_transformers_utils.columnar import \
    ColumnarTraversal as _ColumnarTraversal
from relational_transformers_utils.columnar import _scalar
from relational_transformers_utils.traversal import (BreadthFirstTraversal,
                                                     ContextPolicy,
                                                     GraphAccess,
                                                     GraphTraversal,
                                                     TraversalResult)
from relational_transformers_utils.traversal import \
    ReferenceTraversal as _ReferenceTraversal
from relational_transformers_utils.walks import U32 as _U32  # noqa: F401
from relational_transformers_utils.walks import U64 as _U64  # noqa: F401
from relational_transformers_utils.walks import StdRng as _StdRng  # noqa: F401
from relational_transformers_utils.walks import \
    rand_sample as _rand_sample  # noqa: F401
from relational_transformers_utils.walks import \
    reference_walk_counts as _reference_walk_counts  # noqa: F401
from relational_transformers_utils.walks import \
    stdrng_first_u64_batch as _stdrng_first_u64_batch  # noqa: F401

from .retrieve import RetrieverWiring, TemporalBound

from .evaluate import eval_bool, eval_value
from .relql.ast import TaskType
from .task import TaskSpec

__all__ = ["ContextPolicy", "GraphAccess", "GraphTraversal", "TraversalResult",
           "BreadthFirstTraversal", "ReferenceTraversal", "RelqlTaskAdapter",
           "ColumnarStore", "ColumnarTraversal"]


class RelqlTaskAdapter:
    """Interprets a parsed RelQL query for derived-target traversal."""

    def __init__(self, task_spec_factory) -> None:
        self.task_spec_factory = task_spec_factory

    def spec(self, query: Any, schema) -> TaskSpec:
        spec = self.task_spec_factory(query, query.task_type(schema))
        if not isinstance(spec, TaskSpec):
            raise TypeError("task_spec_factory must return a TaskSpec")
        return spec

    def window_span(self, query: Any):
        return next((a.window.span() for a in query.target_aggregations
                     if a.window is not None), None)

    def aggregated_tables(self, query: Any, entity_table: str) -> set:
        return {a.column.table for a in query.target_aggregations
                if a.column.table != entity_table}

    def label(self, query: Any, schema, visible: dict, entity_cells: dict,
              ts) -> Optional[float]:
        if query.task_type(schema) is TaskType.BINARY_CLASSIFICATION:
            return 1.0 if eval_bool(query.target, visible, entity_cells,
                                    ts) else 0.0
        value = eval_value(query.target, visible, entity_cells, ts)
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if not isinstance(value, (int, float)):
            return None
        return value


class ReferenceTraversal(_ReferenceTraversal):
    """Reference tiering bound to RelQL task derivation."""

    def __init__(self, task_spec_factory=None, task_graph_factory=None):
        factory = task_spec_factory or TaskSpec.from_query
        super().__init__(task_spec_factory=factory,
                         task_graph_factory=task_graph_factory,
                         task_adapter=RelqlTaskAdapter(factory))


def _engine_fallback(message: str) -> None:
    from .engine import _fallback
    _fallback(message)


class ColumnarStore(_ColumnarStore):
    """Array-backed store plus retrievers speaking the Engine wiring contract."""

    def wiring(self) -> RetrieverWiring:
        store = self

        def entity(table, ids, bound: TemporalBound):
            out = []
            t = store._table(table)
            for eid in ids:
                pos = t.id_pos.get(_scalar(eid))
                if pos is None:
                    continue
                row = store.row(store.base[table] + int(pos))
                if bound.admits_row(row):
                    out.append(row)
            return out

        def links(link, parent_id, bound: TemporalBound, limit):
            pnode = store.node_of(link.to_table, parent_id)
            if pnode is None:
                return []
            kids = [store.row(int(c)) for c in store.children(pnode)
                    if store.table_of(int(c)) == link.from_table]
            kids = [r for r in kids if bound.admits_row(r)]
            kids.sort(key=lambda r: (r.timestamp is None,
                                     -(r.timestamp.timestamp()
                                       if r.timestamp else 0.0)))
            return kids[:limit]

        def make_scanner(table):
            def scan(t, bound: TemporalBound):
                base = store.base[table]
                for pos in range(store._table(table).n):
                    row = store.row(base + pos)
                    if bound.admits_row(row):
                        yield row
            return scan

        builder = RetrieverWiring.new_wiring().default_links(links)
        for name in self.tables:
            builder.entities(name, entity)
            builder.scanner(name, make_scanner(name))
        return builder.build()


class ColumnarTraversal(_ColumnarTraversal):
    """Shared-context traversal over a :class:`ColumnarStore`, RelQL-bound."""

    def __init__(self, store: ColumnarStore,
                 task_spec_factory: Optional[Callable] = None):
        factory = task_spec_factory or TaskSpec.from_query
        self.task_spec_factory = factory
        super().__init__(store, task_adapter=RelqlTaskAdapter(factory),
                         fallback=_engine_fallback)
