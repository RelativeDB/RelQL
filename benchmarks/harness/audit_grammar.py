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

CORPUS_PQL = Path(__file__).resolve().parent.parent.parent / "python/tests/data/examples.pql"

# (query, why) — grammatically valid per README/grammar; must parse.
SHOULD_PARSE = [
    ("PREDICT SUM(t.x, 0, 30) FOR EACH e.id", "minimal regression"),
    ("PREDICT COUNT(t.*, -INF, 0) > 0 FOR EACH e.id", "unbounded lookback"),
    ("PREDICT COUNT(t.*, 0, 30, days) = 0 FOR EACH e.id WHERE COUNT(t.*, -90, 0) > 0", "where"),
    ("PREDICT LAST(s.status, 0, 90) = 'X' FOR EACH e.id", "string eq"),
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
    ("PREDICT NOT LAST(t.a, 0, 30) > 30 FOR EACH e.id", "NOT prefix"),
    ("PREDICT SUM(t.p,0,30)>10 OR COUNT(v.*,0,30)>20 FOR EACH e.id", "OR"),
    ("PREDICT a.x = 'IT' AND a.y <= 1990-01-01 FOR EACH a.id", "AND + date literal"),
    ("PREDICT SUM(t.x,0,30) FOR EACH e.id ASSUMING COUNT(n.*,0,7) > 2", "ASSUMING"),
    ("PREDICT LIST_DISTINCT(t.a,0,30) RANK TOP 12 FOR EACH e.id", "RANK TOP"),
    ("PREDICT LIST_DISTINCT(t.a,0,30) CLASSIFY FOR EACH e.id", "CLASSIFY"),
    ("PREDICT SUM(u.count,0,1,days) FORECAST 28 TIMEFRAMES FOR EACH a.id", "FORECAST"),
    ("PREDICT COUNT(t.* WHERE t.amount > 100) FOR EACH e.id", "inline agg filter"),
    ("PREDICT COUNT(t.*, 0, 30, months) FOR EACH e.id", "months unit"),
    ("PREDICT COUNT(t.*, 0, 6, hours) > 0 FOR EACH e.id", "hours unit"),
    ("PREDICT SUM(m.value,0,90,seconds) FOR EACH d.id", "seconds unit"),
    ("PREDICT COUNT(o.*,0,90,days)=0 FOR users.user_id IN (42, 123)", "pinned IN"),
    ("PREDICT COUNT(o.*,0,90,days)=0 FOR users.user_id = 42", "pinned single"),
    ("PREDICT SUM(t.x,0,30) FOR EACH e.id WHERE (a.c='US' OR b.n<10000) AND a.d='V'", "nested parens"),
    ("predict sum(t.x, 0, 30) for each e.id", "lowercase keywords"),
    ("PREDICT  SUM( t.x , 0 , 30 )   FOR   EACH   e.id", "whitespace"),
    ("PREDICT SUM(t.x, 0, 30) FOR EACH e.id -- trailing comment", "line comment"),
]

# (query, why) — must be rejected (syntax or validation). Silent accept = bug.
SHOULD_REJECT = [
    ("PREDICT FOR EACH e.id", "no target"),
    ("SUM(t.x,0,30) FOR EACH e.id", "missing PREDICT"),
    ("PREDICT SUM(t.x,0,30)", "missing FOR EACH"),
    ("PREDICT SUM(t.x,0,30) FOR EACH e", "entity key not table.col"),
    ("PREDICT SUM(t.x 0 30) FOR EACH e.id", "missing commas"),
    ("PREDICT SUM(t.x,0,30 FOR EACH e.id", "unbalanced paren"),
    ("PREDICT BOGUS(t.x,0,30) FOR EACH e.id", "unknown agg func"),
    ("PREDICT SUM(t.x,0,30) FOR EACH e.id WHERE", "dangling WHERE"),
    ("PREDICT SUM(t.x,0,30) RANK TOP FOR EACH e.id", "RANK TOP without k"),
    ("PREDICT LIST_DISTINCT(t.a,0,30) RANK TOP -1 FOR EACH e.id", "negative k"),
    ("PREDICT SUM(t.x,0,30) FORECAST TIMEFRAMES FOR EACH e.id", "FORECAST without n"),
    ("PREDICT SUM(t.x,0,30) FOR EACH e.id EXTRA JUNK", "trailing junk"),
    ("PREDICT t.d LIKE FOR EACH e.id", "LIKE without pattern"),
]


def _load_corpus() -> list[str]:
    return [ln.strip() for ln in CORPUS_PQL.read_text().splitlines()
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
