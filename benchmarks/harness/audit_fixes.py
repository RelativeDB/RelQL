"""Regression guard for context-truncation instrumentation: a warning +
``PredictionResult.stats`` when a windowed COUNT/SUM/AVG is computed over a
fanout-capped context. Instrumentation only — predictions are unchanged.
"""
from __future__ import annotations

import warnings

from relativedb import (ContextTruncationWarning, Engine, ExecutionInput,
                        SamplerMode)

from .datasets import Dataset


def run(ret: Dataset) -> dict:
    findings, checks = [], {}
    anchor = ret.anchors[1]
    pk = ret.schema.table(ret.entity_table).primary_key
    child = "purchases"
    count_q = (f"PREDICT COUNT({child}.*) OVER (90 DAYS FOLLOWING) FOR EACH "
               f"{ret.entity_table}.{pk} WHERE COUNT({child}.*) OVER (90 DAYS PRECEDING) > 0")

    # Under the default cap the truncation is surfaced (warns + stats > 0);
    # under a wide policy it is silent (stats == 0). Predictions are unchanged
    # either way — this is pure observability.
    default_eng = Engine(ret.schema, ret.engine.wiring, sampler_mode=SamplerMode.CSC)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = default_eng.execute(ExecutionInput(query=count_q, anchor_time=anchor))
    warned = any(issubclass(x.category, ContextTruncationWarning) for x in w)
    checks["truncation_warned_default"] = warned
    checks["truncation_stats"] = res.stats
    wide = ret.engine.execute(ExecutionInput(query=count_q, anchor_time=anchor))
    checks["truncation_silent_wide"] = wide.stats.get("contexts_truncated", -1)

    if not (warned and res.stats.get("contexts_truncated", 0) > 0):
        findings.append("truncation instrumentation regressed: default-policy "
                        "truncation no longer surfaced")
    if wide.stats.get("contexts_truncated", 0) != 0:
        findings.append("truncation instrumentation: wide policy wrongly "
                        "reported truncation")
    return {"checks": checks, "findings": findings}
