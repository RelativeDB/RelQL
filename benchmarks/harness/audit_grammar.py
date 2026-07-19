"""Parser / grammar conformance + fuzz audit.

Three things:
  1. Conformance — every query in the shared ``examples.relql`` corpus must
     parse (it is the cross-language conformance suite).
  2. Coverage — which AggFuncs / Operators / TimeUnits / TaskTypes does the
     corpus actually exercise? Unused ones are blind spots in the suite.
  3. Robustness — a battery of hand-written *should-parse* and *should-reject*
     probes. A should-parse that raises, or a should-reject that is silently
     accepted, is a finding.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from relativedb import RelqlSyntaxError, RelqlValidationError, parse
from relativedb.relql.ast import AggFunc, Aggregation, Operator, TimeUnit

CORPUS_RelQL = Path(__file__).resolve().parent.parent.parent / "python/tests/data/examples.relql"

# (query, why) — grammatically valid per README/grammar; must parse.
SHOULD_PARSE = [
    ("PREDICT SUM(t.x) OVER (30 DAYS FOLLOWING) FROM e", "minimal regression"),
    ("PREDICT COUNT(t.*) OVER (UNBOUNDED PRECEDING) > 0 FROM e", "unbounded lookback"),
    ("PREDICT COUNT(t.*) OVER (30 DAYS FOLLOWING) = 0 FROM e WHERE COUNT(t.*) OVER (90 DAYS PRECEDING) > 0", "where"),
    ("PREDICT LAST(s.status) OVER (90 DAYS FOLLOWING) = 'X' FROM e", "string eq"),
    ("PREDICT t.col IN ('A','B','C') FROM e", "IN list"),
    ("PREDICT t.col NOT IN ('A') FROM e", "NOT IN"),
    ("PREDICT t.d IS NULL FROM e", "IS NULL"),
    ("PREDICT t.d IS NOT NULL FROM e", "IS NOT NULL"),
    ("PREDICT t.title STARTS WITH 'The' FROM e", "STARTS WITH"),
    ("PREDICT t.title ENDS WITH 'ing' FROM e", "ENDS WITH"),
    ("PREDICT t.d CONTAINS 'x' FROM e", "CONTAINS"),
    ("PREDICT t.d NOT CONTAINS 'x' FROM e", "NOT CONTAINS"),
    ("PREDICT t.d LIKE '%x%' FROM e", "LIKE"),
    ("PREDICT t.d NOT LIKE '%x' FROM e", "NOT LIKE"),
    ("PREDICT NOT LAST(t.a) OVER (30 DAYS FOLLOWING) > 30 FROM e", "NOT prefix"),
    ("PREDICT SUM(t.p) OVER (30 DAYS FOLLOWING) > 10 OR COUNT(v.*) OVER (30 DAYS FOLLOWING) > 20 FROM e", "OR"),
    ("PREDICT a.x = 'IT' AND a.y <= 1990-01-01 FROM a", "AND + date literal"),
    ("PREDICT SUM(t.x) OVER (30 DAYS FOLLOWING) FROM e ASSUMING COUNT(n.*) OVER (7 DAYS FOLLOWING) > 2", "ASSUMING"),
    ("PREDICT LIST_DISTINCT(t.a) OVER (30 DAYS FOLLOWING RANK TOP 12) FROM e", "RANK TOP"),
    ("PREDICT ARRAY_AGG(t.a) OVER (RANK TOP 12) FROM e", "RANK TOP, implied frame"),
    ("PREDICT label FROM issues WHERE label IS NULL", "unqualified columns"),
    ("PREDICT issues.label WHERE issues.label IS NULL", "population inferred"),
    ("PREDICT NOT EXISTS(o.*) FROM customers c WHERE c.id = 1", "FROM alias"),
    ("PREDICT LIST_DISTINCT(t.a) OVER (30 DAYS FOLLOWING) CLASSIFY FROM e", "CLASSIFY"),
    ("PREDICT SUM(u.count) OVER (1 DAY FOLLOWING HORIZONS 28) FROM a", "multi-horizon window"),
    ("PREDICT COUNT(t.* WHERE t.amount > 100) FROM e", "inline agg filter"),
    ("PREDICT COUNT(t.*) OVER (30 MONTHS FOLLOWING) FROM e", "months unit"),
    ("PREDICT COUNT(t.*) OVER (6 HOURS FOLLOWING) > 0 FROM e", "hours unit"),
    ("PREDICT SUM(m.value) OVER (90 SECONDS FOLLOWING) FROM d", "seconds unit"),
    ("PREDICT SUM(t.x) OVER (30 DAYS FOLLOWING) FROM e WHERE (a.c='US' OR b.n<10000) AND a.d='V'", "nested parens"),
    ("predict sum(t.x) over (30 days following) from e", "lowercase keywords"),
    ("PREDICT  SUM( t.x )   OVER  ( 30 DAYS FOLLOWING )   FROM e", "whitespace"),
    ("PREDICT SUM(t.x) OVER (30 DAYS FOLLOWING) FROM e -- trailing comment", "line comment"),
    # --- new v2 frame / clause forms ---
    ("PREDICT SUM(t.value) OVER (RANGE BETWEEN 15 DAYS FOLLOWING AND 45 DAYS FOLLOWING) FROM e", "RANGE BETWEEN frame"),
    ("PREDICT MAX(t.value) OVER (7 DAYS FOLLOWING HORIZONS 4) FROM e", "frame with HORIZONS"),
    ("PREDICT SUM(o.revenue) OVER w - SUM(o.cost) OVER w FROM e WINDOW w AS (30 DAYS FOLLOWING)", "named WINDOW + OVER w"),
    ("PREDICT EXISTS(orders.*) OVER (90 DAYS FOLLOWING) FROM e", "EXISTS aggregation"),
    ("PREDICT NOT EXISTS(orders.*) OVER (90 DAYS FOLLOWING) FROM e WHERE EXISTS(orders.*) OVER (90 DAYS PRECEDING)", "NOT EXISTS + EXISTS where"),
    ("PREDICT SUM(t.x) OVER (30 DAYS FOLLOWING) FROM e AS OF :prediction_time", "AS OF param"),
    ("PREDICT EXISTS(orders.*) OVER (30 DAYS FOLLOWING) FROM e RETURN PROBABILITY", "RETURN PROBABILITY"),
    ("EXPLAIN PLAN FORMAT TEXT PREDICT EXISTS(orders.*) OVER (30 DAYS FOLLOWING) FROM e RETURN PROBABILITY", "EXPLAIN PLAN prefix"),
]

# (query, why) — must be rejected (syntax or validation). Silent accept = bug.
SHOULD_REJECT = [
    ("PREDICT SUM(t.x) OVER (30 DAYS FOLLOWING) FROM e RETURN QUANTILES (0.1, 0.9)", "RETURN QUANTILES removed"),
    ("PREDICT SUM(t.x) OVER (30 DAYS FOLLOWING) FROM e RETURN INTERVAL 90%", "RETURN INTERVAL removed"),
    ("PREDICT FROM e", "no target"),
    ("SUM(t.x) OVER (30 DAYS FOLLOWING) FROM e", "missing PREDICT"),
    ("PREDICT SUM(t.x) OVER (30 DAYS FOLLOWING)", "aggregate target needs FROM"),
    ("PREDICT SUM(t.x) OVER (30 DAYS FOLLOWING) FOR EACH e.id", "FOR EACH removed"),
    ("PREDICT SUM(t.x) OVER 30 DAYS FOLLOWING FROM e", "frame missing parens"),
    ("PREDICT SUM(t.x) OVER (30 DAYS FOLLOWING FROM e", "unbalanced paren"),
    ("PREDICT BOGUS(t.x) OVER (30 DAYS FOLLOWING) FROM e", "unknown agg func"),
    ("PREDICT SUM(t.x) OVER (30 DAYS FOLLOWING) FROM e WHERE", "dangling WHERE"),
    ("PREDICT SUM(t.x) OVER (30 DAYS FOLLOWING) RANK TOP FROM e", "RANK TOP without k"),
    ("PREDICT LIST_DISTINCT(t.a) OVER (30 DAYS FOLLOWING) RANK TOP -1 FROM e", "negative k"),
    ("PREDICT SUM(t.x) OVER (30 DAYS FOLLOWING) FROM e EXTRA JUNK", "trailing junk"),
    ("PREDICT t.d LIKE FROM e", "LIKE without pattern"),
    # --- old positional-window / FORECAST forms are now syntax errors ---
    ("PREDICT SUM(t.x, 0, 30) FROM e", "positional window removed"),
    ("PREDICT COUNT(orders.*, 0, 90, days) FROM e", "positional window w/ unit removed"),
    ("PREDICT COUNT(t.*, -90, 0) > 0 FROM e", "positional lookback removed"),
    ("PREDICT SUM(u.count, 0, 1, days) FORECAST 28 TIMEFRAMES FROM a", "FORECAST clause removed"),
    # --- pinned entity selector removed; only FROM remains ---
    ("PREDICT COUNT(o.*) OVER (90 DAYS FOLLOWING) = 0 FOR users.user_id IN (42, 123)", "pinned IN removed"),
    ("PREDICT COUNT(o.*) OVER (90 DAYS FOLLOWING) = 0 FOR users.user_id = 42", "pinned single removed"),
    ("PREDICT SUM(t.x) OVER (30 DAYS FOLLOWING) FOR e.id", "bare FOR (needs EACH)"),
    # --- new v2 frame validation errors ---
    ("PREDICT SUM(t.x) OVER (30 DAYS) FROM e", "frame missing PRECEDING/FOLLOWING"),
    ("PREDICT SUM(t.x) OVER (30 DAYS FOLLOWING HORIZONS 0) FROM e", "horizons must be positive"),
    ("PREDICT SUM(t.x) OVER undeclared_window FROM e", "undeclared window name"),
]


def _load_corpus() -> list[str]:
    return [ln.strip() for ln in CORPUS_RelQL.read_text().splitlines()
            if ln.strip() and not ln.startswith("#")]


def run() -> dict:
    findings: list[str] = []

    # 1. conformance
    corpus = _load_corpus()
    parsed, parse_fail = [], []
    for q in corpus:
        try:
            parsed.append(parse(q))
        except Exception as e:                       # noqa: BLE001
            parse_fail.append((q, f"{type(e).__name__}: {e}"))
    if parse_fail:
        findings.append(f"{len(parse_fail)} conformance queries FAILED to parse")

    # 2. coverage
    used_agg, used_op, used_unit, used_task = set(), set(), set(), Counter()
    for pq in parsed:
        for a in pq.target_aggregations:
            used_agg.add(a.func.name)
            if a.window is not None and a.window.unit is not None:
                used_unit.add(a.window.unit.name)
        used_task[pq.task_type().name] += 1
        used_op |= _ops_in(pq.target) | (_ops_in(pq.where) if pq.where else set())
    missing_agg = {f.name for f in AggFunc} - used_agg
    missing_op = {o.name for o in Operator} - used_op
    missing_unit = {u.name for u in TimeUnit} - used_unit
    if missing_op:
        findings.append(f"operators never exercised by corpus: {sorted(missing_op)}")
    if missing_agg:
        findings.append(f"agg funcs never exercised by corpus: {sorted(missing_agg)}")

    # 3. robustness probes
    should_parse_fail, should_reject_pass = [], []
    for q, why in SHOULD_PARSE:
        try:
            parse(q)
        except Exception as e:                       # noqa: BLE001
            should_parse_fail.append((q, why, f"{type(e).__name__}: {e}"))
    for q, why in SHOULD_REJECT:
        try:
            parse(q)
            should_reject_pass.append((q, why))
        except (RelqlSyntaxError, RelqlValidationError, ValueError):
            pass
        except Exception as e:                       # noqa: BLE001
            should_reject_pass.append((q, f"{why} — crashed with {type(e).__name__} instead of RelqlSyntaxError"))
    if should_parse_fail:
        findings.append(f"{len(should_parse_fail)} should-parse probes rejected")
    if should_reject_pass:
        findings.append(f"{len(should_reject_pass)} should-reject probes silently accepted")

    return {
        "corpus_total": len(corpus),
        "corpus_parsed": len(parsed),
        "parse_failures": parse_fail,
        "coverage": {"agg_used": sorted(used_agg), "agg_missing": sorted(missing_agg),
                     "op_used": sorted(used_op), "op_missing": sorted(missing_op),
                     "unit_used": sorted(used_unit), "unit_missing": sorted(missing_unit),
                     "task_dist": dict(used_task)},
        "should_parse_fail": should_parse_fail,
        "should_reject_pass": should_reject_pass,
        "findings": findings,
    }


def _ops_in(expr) -> set:
    from relativedb.relql.ast import Condition, LogicalOp, Not
    if expr is None:
        return set()
    if isinstance(expr, Condition):
        return {expr.op.name} | _ops_in(expr.left)
    if isinstance(expr, LogicalOp):
        return _ops_in(expr.left) | _ops_in(expr.right)
    if isinstance(expr, Not):
        return _ops_in(expr.expr)
    return set()
