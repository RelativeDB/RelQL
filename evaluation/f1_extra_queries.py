"""Leak-aware SQL features for extra rel-f1 scalar task tables."""

from __future__ import annotations


QUALIFYING_POSITION_SQL = r"""
WITH target AS (
    SELECT t.date, t."qualifyId", q."driverId", q."constructorId", q.number,
           r.year, r.round
    FROM task_table t
    LEFT JOIN qualifying q ON q."qualifyId" = t."qualifyId"
    LEFT JOIN races r ON r."raceId" = q."raceId"
), qual_history AS (
    SELECT t.date, t."qualifyId",
           AVG(q."position") FILTER (
               WHERE q.date < t.date AND q.date >= t.date - INTERVAL 365 DAY
           ) AS driver_qualifying_1y
    FROM target t
    LEFT JOIN qualifying q ON q."driverId" = t."driverId"
    GROUP BY t.date, t."qualifyId"
), result_history AS (
    SELECT t.date, t."qualifyId",
           AVG(re."positionOrder") FILTER (
               WHERE re.date < t.date AND re.date >= t.date - INTERVAL 365 DAY
           ) AS driver_finish_1y
    FROM target t
    LEFT JOIN results re ON re."driverId" = t."driverId"
    GROUP BY t.date, t."qualifyId"
)
SELECT t.date, t."qualifyId",
       COALESCE(t.number, 0) / 100.0 AS car_number,
       COALESCE(t.year, 2000) / 2020.0 AS race_year,
       COALESCE(t.round, 0) / 25.0 AS season_round,
       COALESCE(qh.driver_qualifying_1y, 13.0) / 30.0 AS prior_qualifying,
       COALESCE(rh.driver_finish_1y, 13.0) / 30.0 AS prior_finish
FROM target t
LEFT JOIN qual_history qh USING (date, "qualifyId")
LEFT JOIN result_history rh USING (date, "qualifyId")
"""


RESULTS_POSITION_SQL = r"""
WITH target AS (
    SELECT t.date, t."resultId", re."driverId", re."constructorId",
           re.number, re.grid,
           ra.year, ra.round
    FROM task_table t
    LEFT JOIN results re ON re."resultId" = t."resultId"
    LEFT JOIN races ra ON ra."raceId" = re."raceId"
), history AS (
    SELECT t.date, t."resultId",
           AVG(re."positionOrder") FILTER (
               WHERE re.date < t.date AND re.date >= t.date - INTERVAL 365 DAY
           ) AS driver_finish_1y
    FROM target t
    LEFT JOIN results re ON re."driverId" = t."driverId"
    GROUP BY t.date, t."resultId"
)
SELECT t.date, t."resultId",
       COALESCE(t.number, 0) / 100.0 AS car_number,
       COALESCE(t.grid, 0) / 30.0 AS grid_position,
       COALESCE(t.year, 2000) / 2020.0 AS race_year,
       COALESCE(t.round, 0) / 25.0 AS season_round,
       COALESCE(h.driver_finish_1y, 13.0) / 30.0 AS prior_finish
FROM target t
LEFT JOIN history h USING (date, "resultId")
"""


def install_sql_queries(registry: dict) -> None:
    registry.update({
        ("rel-f1", "qualifying-position"): {
            "sql": QUALIFYING_POSITION_SQL,
            "entity_col": "qualifyId",
            "time_col": "date",
        },
        ("rel-f1", "results-position"): {
            "sql": RESULTS_POSITION_SQL,
            "entity_col": "resultId",
            "time_col": "date",
        },
    })
