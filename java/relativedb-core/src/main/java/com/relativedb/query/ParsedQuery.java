package com.relativedb.query;

import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalInt;

/** The typed result of parsing a RelQL string (no schema binding yet). */
public record ParsedQuery(
    TargetExpr target,
    ColumnRef entityKey,               // FOR EACH table.pk
    Optional<TargetExpr> where,
    Optional<TargetExpr> assuming,
    OptionalInt topK,                  // RANK TOP K
    Optional<ProblemType> problemType, // RANK | CLASSIFY
    OptionalInt numForecasts,          // derived: target window HORIZONS n (>1)
    Optional<Explain> explain,         // EXPLAIN [mode] [FORMAT fmt] prefix
    Optional<AsOf> asOf,               // AS OF <anchor>
    List<Ablation> ablations,          // ABLATE TABLE ... (repeatable)
    Optional<ReturnSpec> ret,          // RETURN <output>
    Map<String, Window> windows        // declared WINDOW name AS (...) templates
) { }
