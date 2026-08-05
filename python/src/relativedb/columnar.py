"""Columnar context population, bound to RelQL and the retriever wiring.

The store and traversal live in
:mod:`relational_transformers_utils.columnar`; this module adds the
:class:`RetrieverWiring` factory and the RelQL task adapter.

    store = ColumnarStore(schema, frames, task_frames=..., task_links=...)
    engine = Engine(schema, store.wiring(), model_backend=...,
                    traversal=ColumnarTraversal(store))
"""

from __future__ import annotations

from typing import Callable, Optional

from relational_transformers_utils.columnar import \
    ColumnarStore as _ColumnarStore
from relational_transformers_utils.columnar import \
    ColumnarTraversal as _ColumnarTraversal
from relational_transformers_utils.columnar import _scalar

from .retrieve import RetrieverWiring, TemporalBound
from .task import TaskSpec
from .traversal import RelqlTaskAdapter

__all__ = ["ColumnarStore", "ColumnarTraversal"]


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
