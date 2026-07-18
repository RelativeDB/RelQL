"""Parser / grammar conformance + fuzz audit.

Three things:
  1. Conformance — every query in the shared ``examples.pql`` corpus must
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

from relativedb import PqlSyntaxError, PqlValidationError, parse
from relativedb.pql.ast import AggFunc, Aggregation, Operator, TimeUnit

CORPUS_RelQL = Path(__file__).resolve().parent.parent.parent / "python/tests/data/examples.pql"

# (query, why) — grammatically valid per README/grammar; must parse.
SHOULD_PARSE = [
    ("PREDICT SUM(t.x) OVER (30 DAYS FOLLOWING) FOR EACH e.id", "minimal regression"),
    ("PREDICT COUNT(t.*) OVER (UNBOUNDED PRECEDING) > 0 FOR EACH e.id", "unbounded lookback"),
    ("PREDICT COUNT(t.*) OVER (30 DAYS FOLLOWING) = 0 FOR EACH e.id WHERE COUNT(t.*) OVER (90 DAYS PRECEDING) > 0", "where"),
    ("PREDICT LAST(s.status) OVER (90 DAYS FOLLOWING) = 'X' FOR EACH e.id", "string eq"),
    ("PREDICT t.col IN ('A','B','C') FOR EACH e.id", "IN list"),
    ("PREDICT t.col NOT IN ('A') FOR EACH e.id", "NOT IN"),
    ("PREDICT t.d IS NULL FOR EACH e.id", "IS NULL"),
    ("PREDICT t.d IS NOT NULL FOR EACH e.id", "IS NOT NULL"),
    ("PREDICT t.title STARTS WITH 'The' FOR EACH e.id", "STARTS WITH"),
    ("PREDICT t.title ENDS WITH 'ing' FOR EACH e.id", "ENDS WITH"),
    ("PREDICT t.d CONTAINS 'x' FOR EACH e.id", "CONTAINS"),
    ("PREDICT t.d NOT CONTAINS 'x' FOR EACH e.id", "NOT CONTAINS"),
    ("PREDICT t.d LIKE '%x%' FOR EACH e.id", "LIKE"),
    ("PREDICT t.d NOT LIKE '%x' FOR EACH e.id", "NOT LIKE"),
    ("PREDICT NOT LAST(t.a) OVER (30 DAYS FOLLOWING) > 30 FOR EACH e.id", "NOT prefix"),
    ("PREDICT SUM(t.p) OVER (30 DAYS FOLLOWING) > 10 OR COUNT(v.*) OVER (30 DAYS FOLLOWING) > 20 FOR EACH e.id", "OR"),
    ("PREDICT a.x = 'IT' AND a.y <= 1990-01-01 FOR EACH a.id", "AND + date literal"),
    ("PREDICT SUM(t.x) OVER (30 DAYS FOLLOWING) FOR EACH e.id ASSUMING COUNT(n.*) OVER (7 DAYS FOLLOWING) > 2", "ASSUMING"),
    ("PREDICT LIST_DISTINCT(t.a) OVER (30 DAYS FOLLOWING) RANK TOP 12 FOR EACH e.id", "RANK TOP"),
    ("PREDICT LIST_DISTINCT(t.a) OVER (30 DAYS FOLLOWING) CLASSIFY FOR EACH e.id", "CLASSIFY"),
    ("PREDICT SUM(u.count) OVER (1 DAY FOLLOWING HORIZONS 28) FOR EACH a.id", "multi-horizon window"),
    ("PREDICT COUNT(t.* WHERE t.amount > 100) FOR EACH e.id", "inline agg filter"),
    ("PREDICT COUNT(t.*) OVER (30 MONTHS FOLLOWING) FOR EACH e.id", "months unit"),
    ("PREDICT COUNT(t.*) OVER (6 HOURS FOLLOWING) > 0 FOR EACH e.id", "hours unit"),
    ("PREDICT SUM(m.value) OVER (90 SECONDS FOLLOWING) FOR EACH d.id", "seconds unit"),
    ("PREDICT SUM(t.x) OVER (30 DAYS FOLLOWING) FOR EACH e.id WHERE (a.c='US' OR b.n<10000) AND a.d='V'", "nested parens"),
    ("predict sum(t.x) over (30 days following) for each e.id", "lowercase keywords"),
    ("PREDICT  SUM( t.x )   OVER  ( 30 DAYS FOLLOWING )   FOR   EACH   e.id", "whitespace"),
    ("PREDICT SUM(t.x) OVER (30 DAYS FOLLOWING) FOR EACH e.id -- trailing comment", "line comment"),
    # --- new v2 frame / clause forms ---
    ("PREDICT SUM(t.value) OVER (RANGE BETWEEN 15 DAYS FOLLOWING AND 45 DAYS FOLLOWING) FOR EACH e.id", "RANGE BETWEEN frame"),
    ("PREDICT MAX(t.value) OVER (7 DAYS FOLLOWING HORIZONS 4) FOR EACH e.id", "frame with HORIZONS"),
    ("PREDICT SUM(o.revenue) OVER w - SUM(o.cost) OVER w FOR EACH e.id WINDOW w AS (30 DAYS FOLLOWING)", "named WINDOW + OVER w"),
    ("PREDICT EXISTS(orders.*) OVER (90 DAYS FOLLOWING) FOR EACH e.id", "EXISTS aggregation"),
    ("PREDICT NOT EXISTS(orders.*) OVER (90 DAYS FOLLOWING) FOR EACH e.id WHERE EXISTS(orders.*) OVER (90 DAYS PRECEDING)", "NOT EXISTS + EXISTS where"),
    ("PREDICT SUM(t.x) OVER (30 DAYS FOLLOWING) FOR EACH e.id AS OF :prediction_time", "AS OF param"),
    ("PREDICT EXISTS(orders.*) OVER (30 DAYS FOLLOWING) FOR EACH e.id RETURN PROBABILITY", "RETURN PROBABILITY"),
    ("PREDICT SUM(t.x) OVER (30 DAYS FOLLOWING) FOR EACH e.id RETURN QUANTILES (0.10, 0.50, 0.90)", "RETURN QUANTILES"),
    ("EXPLAIN PLAN FORMAT TEXT PREDICT EXISTS(orders.*) OVER (30 DAYS FOLLOWING) FOR EACH e.id RETURN PROBABILITY", "EXPLAIN PLAN prefix"),
]

# (query, why) — must be rejected (syntax or validation). Silent accept = bug.
SHOULD_REJECT = [
    ("PREDICT FOR EACH e.id", "no target"),
    ("SUM(t.x) OVER (30 DAYS FOLLOWING) FOR EACH e.id", "missing PREDICT"),
    ("PREDICT SUM(t.x) OVER (30 DAYS FOLLOWING)", "missing FOR EACH"),
    ("PREDICT SUM(t.x) OVER (30 DAYS FOLLOWING) FOR EACH e", "entity key not table.col"),
    ("PREDICT SUM(t.x) OVER 30 DAYS FOLLOWING FOR EACH e.id", "frame missing parens"),
    ("PREDICT SUM(t.x) OVER (30 DAYS FOLLOWING FOR EACH e.id", "unbalanced paren"),
    ("PREDICT BOGUS(t.x) OVER (30 DAYS FOLLOWING) FOR EACH e.id", "unknown agg func"),
    ("PREDICT SUM(t.x) OVER (30 DAYS FOLLOWING) FOR EACH e.id WHERE", "dangling WHERE"),
    ("PREDICT SUM(t.x) OVER (30 DAYS FOLLOWING) RANK TOP FOR EACH e.id", "RANK TOP without k"),
    ("PREDICT LIST_DISTINCT(t.a) OVER (30 DAYS FOLLOWING) RANK TOP -1 FOR EACH e.id", "negative k"),
    ("PREDICT SUM(t.x) OVER (30 DAYS FOLLOWING) FOR EACH e.id EXTRA JUNK", "trailing junk"),
    ("PREDICT t.d LIKE FOR EACH e.id", "LIKE without pattern"),
    # --- old positional-window / FORECAST forms are now syntax errors ---
    ("PREDICT SUM(t.x, 0, 30) FOR EACH e.id", "positional window removed"),
    ("PREDICT COUNT(orders.*, 0, 90, days) FOR EACH e.id", "positional window w/ unit removed"),
    ("PREDICT COUNT(t.*, -90, 0) > 0 FOR EACH e.id", "positional lookback removed"),
    ("PREDICT SUM(u.count, 0, 1, days) FORECAST 28 TIMEFRAMES FOR EACH a.id", "FORECAST clause removed"),
    # --- pinned entity selector removed; only FOR EACH remains ---
    ("PREDICT COUNT(o.*) OVER (90 DAYS FOLLOWING) = 0 FOR users.user_id IN (42, 123)", "pinned IN removed"),
    ("PREDICT COUNT(o.*) OVER (90 DAYS FOLLOWING) = 0 FOR users.user_id = 42", "pinned single removed"),
    ("PREDICT SUM(t.x) OVER (30 DAYS FOLLOWING) FOR e.id", "bare FOR (needs EACH)"),
    # --- new v2 frame validation errors ---
    ("PREDICT SUM(t.x) OVER (30 DAYS) FOR EACH e.id", "frame missing PRECEDING/FOLLOWING"),
    ("PREDICT SUM(t.x) OVER (30 DAYS FOLLOWING HORIZONS 0) FOR EACH e.id", "horizons must be positive"),
    ("PREDICT SUM(t.x) OVER undeclared_window FOR EACH e.id", "undeclared window name"),
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
        except (PqlSyntaxError, PqlValidationError, ValueError):
            pass
        except Exception as e:                       # noqa: BLE001
            should_reject_pass.append((q, f"{why} — crashed with {type(e).__name__} instead of PqlSyntaxError"))
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
    from relativedb.pql.ast import Condition, LogicalOp, Not
    if expr is None:
        return set()
    if isinstance(expr, Condition):
        return {expr.op.name} | _ops_in(expr.left)
    if isinstance(expr, LogicalOp):
        return _ops_in(expr.left) | _ops_in(expr.right)
    if isinstance(expr, Not):
        return _ops_in(expr.expr)
    return set()
