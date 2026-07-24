"""AS OF resolution (contract Part A), tested without standing up an Engine.

The precedence rules used to be buried in Engine methods, so exercising them
meant building a schema, a wiring and a backend first. They are pure: the
query's AS OF clause, the caller's anchor_time and the bound params are the
whole input.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from relativedb import parse, validate
from relativedb.anchors import coerce_anchor, effective_anchor, parse_anchor_date
from relativedb.errors import ExecutionError

BASE = "PREDICT NOT EXISTS(orders.*) OVER (90 DAYS FOLLOWING) FROM customers"
T0 = datetime(2026, 7, 1, tzinfo=timezone.utc)


def as_of_of(churn_schema, suffix: str):
    return validate(parse(BASE + suffix), churn_schema).query.as_of


# -- parse_anchor_date -----------------------------------------------------

def test_parses_date_and_datetime_forms():
    assert parse_anchor_date("2026-07-01") == T0
    assert (parse_anchor_date("2026-07-01 12:30:05")
            == datetime(2026, 7, 1, 12, 30, 5, tzinfo=timezone.utc))


def test_parsed_dates_are_utc_aware():
    """A naive anchor compared against aware row timestamps raises at compare
    time, so the boundary coerces once, here."""
    assert parse_anchor_date("2026-07-01").tzinfo is timezone.utc


@pytest.mark.parametrize("bad", ["", "  ", "07/01/2026", "2026-13-01",
                                 "yesterday", None])
def test_unparseable_dates_name_the_expected_format(bad):
    with pytest.raises(ExecutionError, match="cannot parse date"):
        parse_anchor_date(bad)


# -- coerce_anchor ---------------------------------------------------------

def test_coerce_accepts_aware_naive_and_string():
    assert coerce_anchor(T0) == T0
    assert coerce_anchor(datetime(2026, 7, 1)) == T0     # naive -> UTC
    assert coerce_anchor("2026-07-01") == T0


@pytest.mark.parametrize("bad", [1751328000, 3.5, [], {}, object()])
def test_coerce_rejects_non_datetime_bindings(bad):
    with pytest.raises(ExecutionError, match="must bind to a datetime"):
        coerce_anchor(bad)


# -- effective_anchor: the precedence rules --------------------------------

def test_absent_as_of_defers_to_the_execution_anchor(churn_schema):
    assert effective_anchor(as_of_of(churn_schema, ""), T0, None) == T0


def test_absent_as_of_stays_unbounded_without_an_anchor(churn_schema):
    assert effective_anchor(as_of_of(churn_schema, ""), None, None) is None


def test_query_date_overrides_the_execution_anchor(churn_schema):
    got = effective_anchor(as_of_of(churn_schema, " AS OF 2026-08-01"), T0, None)
    assert got == datetime(2026, 8, 1, tzinfo=timezone.utc)


def test_param_binds_from_params(churn_schema):
    ao = as_of_of(churn_schema, " AS OF :t")
    want = datetime(2026, 5, 1, tzinfo=timezone.utc)
    assert effective_anchor(ao, T0, {"t": want}) == want
    assert effective_anchor(ao, T0, {"t": "2026-05-01"}) == want


def test_param_falls_back_to_the_execution_anchor(churn_schema):
    """An unbound AS OF :t is not an error while anchor_time can stand in --
    that is what makes the same query text reusable across anchors."""
    assert effective_anchor(as_of_of(churn_schema, " AS OF :t"), T0, {}) == T0


def test_unbound_param_with_no_fallback_says_how_to_fix_it(churn_schema):
    with pytest.raises(ExecutionError, match=r"no value bound for parameter 't'"):
        effective_anchor(as_of_of(churn_schema, " AS OF :t"), None, {})


def test_engine_methods_still_delegate_here(churn_schema, stub_backend):
    """The Engine wrappers are part of the public surface; keep them working."""
    from relativedb import Engine, ExecutionInput
    from conftest import churn_rows, in_memory_wiring
    eng = Engine(churn_schema, in_memory_wiring(churn_rows()),
                 model_backend=stub_backend)
    pq = validate(parse(BASE + " AS OF 2026-08-01"), churn_schema).query
    got = eng._effective_anchor(pq, ExecutionInput(query=pq, anchor_time=T0))
    assert got == datetime(2026, 8, 1, tzinfo=timezone.utc)
    assert eng._coerce_anchor("2026-07-01") == T0
    assert eng._parse_anchor_date("2026-07-01") == T0
