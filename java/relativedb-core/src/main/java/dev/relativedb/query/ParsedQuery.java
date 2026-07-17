package dev.relativedb.query;

import java.util.List;
import java.util.Optional;
import java.util.OptionalInt;

/** The typed result of parsing a PQL string (no schema binding yet). */
public record ParsedQuery(
    TargetExpr target,
    ColumnRef entityKey,               // FOR [EACH] table.pk
    List<Literal> entityIds,           // FOR t.pk = v | IN (...)  (empty = all)
    Optional<TargetExpr> where,
    Optional<TargetExpr> assuming,
    OptionalInt topK,                  // RANK TOP K
    Optional<ProblemType> problemType, // RANK | CLASSIFY
    OptionalInt numForecasts           // FORECAST N TIMEFRAMES
) { }
