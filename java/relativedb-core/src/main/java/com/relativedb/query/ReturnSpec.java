package com.relativedb.query;

import java.util.Arrays;
import java.util.OptionalInt;

/**
 * A {@code RETURN <output>} clause: the requested output shape. {@code quantiles}
 * is populated for {@code QUANTILES(...)}; {@code interval} for {@code INTERVAL n[%]}.
 */
public record ReturnSpec(Kind kind, double[] quantiles, OptionalInt interval) {

    public enum Kind {
        EXPECTED_VALUE, PROBABILITY, CLASS, DISTRIBUTION,
        QUANTILES, INTERVAL, MULTILABEL, MULTICLASS
    }

    @Override public String toString() {
        return "ReturnSpec[kind=" + kind + ", quantiles=" + Arrays.toString(quantiles)
                + ", interval=" + interval + "]";
    }
}
