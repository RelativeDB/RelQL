package com.relativedb.engine;

import com.relativedb.query.Explain;

import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * The result of {@link RelativeDbEngine#explain(ExecutionInput)}: the compiled
 * plan (always), an optionally-assembled context report (CONTEXT / ANALYZE), and
 * optional scored predictions (ANALYZE only). {@link #render()} produces a human
 * TEXT dump or a stable machine JSON object per the query's requested format.
 *
 * <p>The {@code plan} / {@code context} maps use snake_case string keys and hold
 * only JSON-native values: {@link String}, {@link Number}, {@link Boolean},
 * {@link Map}, {@link List}, or {@code null}. This keeps the JSON encoding total
 * and lossless.
 */
public final class ExplainResult {

    private final Explain.Mode mode;
    private final Explain.Format format;
    private final Map<String, Object> plan;
    private final Map<String, Object> context;      // nullable (PLAN / ABLATION)
    private final PredictionResult predictions;     // nullable (non-ANALYZE)

    ExplainResult(Explain.Mode mode, Explain.Format format, Map<String, Object> plan,
                  Map<String, Object> context, PredictionResult predictions) {
        this.mode = mode;
        this.format = format;
        this.plan = plan;
        this.context = context;
        this.predictions = predictions;
    }

    public Explain.Mode mode() { return mode; }
    public Explain.Format format() { return format; }
    public Map<String, Object> plan() { return plan; }
    public Optional<Map<String, Object>> context() { return Optional.ofNullable(context); }
    public Optional<PredictionResult> predictions() { return Optional.ofNullable(predictions); }

    /** Render per the query's requested format. */
    public String render() {
        return format == Explain.Format.JSON ? renderJson() : renderText();
    }

    // ------------------------------------------------------------------
    //  JSON
    // ------------------------------------------------------------------

    private String renderJson() {
        java.util.LinkedHashMap<String, Object> root = new java.util.LinkedHashMap<>();
        root.put("mode", mode.name());
        root.put("format", format.name());
        root.put("plan", plan);
        if (context != null) root.put("context", context);
        if (predictions != null) root.put("predictions", predictionsJson());
        StringBuilder sb = new StringBuilder();
        writeJson(sb, root);
        return sb.toString();
    }

    private Object predictionsJson() {
        java.util.LinkedHashMap<String, Object> out = new java.util.LinkedHashMap<>();
        out.put("task_type", predictions.taskType().name());
        List<Object> rows = new java.util.ArrayList<>();
        for (PredictionResult.EntityPrediction p : predictions.predictions()) {
            java.util.LinkedHashMap<String, Object> m = new java.util.LinkedHashMap<>();
            m.put("id", String.valueOf(p.id().raw()));
            p.value().ifPresent(v -> m.put("value", v));
            p.probability().ifPresent(v -> m.put("probability", v));
            if (!p.classProbs().isEmpty()) m.put("class_probs", p.classProbs());
            if (!p.ranked().isEmpty()) {
                List<Object> rk = new java.util.ArrayList<>();
                for (PredictionResult.RankedItem r : p.ranked()) {
                    java.util.LinkedHashMap<String, Object> ri = new java.util.LinkedHashMap<>();
                    ri.put("item", r.item());
                    ri.put("score", r.score());
                    rk.add(ri);
                }
                m.put("ranked", rk);
            }
            if (!p.forecast().isEmpty()) {
                List<Object> fc = new java.util.ArrayList<>();
                for (PredictionResult.TimeframeValue tv : p.forecast()) {
                    java.util.LinkedHashMap<String, Object> fv = new java.util.LinkedHashMap<>();
                    fv.put("timeframe", tv.timeframe());
                    fv.put("value", tv.value());
                    fc.add(fv);
                }
                m.put("forecast", fc);
            }
            p.predictedClass().ifPresent(c -> m.put("class", c));
            rows.add(m);
        }
        out.put("entities", rows);
        return out;
    }

    @SuppressWarnings("unchecked")
    private static void writeJson(StringBuilder sb, Object v) {
        if (v == null) {
            sb.append("null");
        } else if (v instanceof Map<?, ?> m) {
            sb.append('{');
            boolean first = true;
            for (Map.Entry<?, ?> e : m.entrySet()) {
                if (!first) sb.append(',');
                first = false;
                writeString(sb, String.valueOf(e.getKey()));
                sb.append(':');
                writeJson(sb, e.getValue());
            }
            sb.append('}');
        } else if (v instanceof List<?> list) {
            sb.append('[');
            for (int i = 0; i < list.size(); i++) {
                if (i > 0) sb.append(',');
                writeJson(sb, list.get(i));
            }
            sb.append(']');
        } else if (v instanceof String s) {
            writeString(sb, s);
        } else if (v instanceof Boolean b) {
            sb.append(b.booleanValue() ? "true" : "false");
        } else if (v instanceof Double d) {
            if (d.isNaN() || d.isInfinite()) writeString(sb, String.valueOf(d));
            else sb.append(d.doubleValue() == Math.rint(d) && !d.isInfinite()
                    ? String.valueOf(d.longValue()) : d.toString());
        } else if (v instanceof Number n) {
            sb.append(n.toString());
        } else {
            writeString(sb, String.valueOf(v));
        }
    }

    private static void writeString(StringBuilder sb, String s) {
        sb.append('"');
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"':  sb.append("\\\""); break;
                case '\\': sb.append("\\\\"); break;
                case '\n': sb.append("\\n"); break;
                case '\r': sb.append("\\r"); break;
                case '\t': sb.append("\\t"); break;
                case '\b': sb.append("\\b"); break;
                case '\f': sb.append("\\f"); break;
                default:
                    if (c < 0x20) sb.append(String.format("\\u%04x", (int) c));
                    else sb.append(c);
            }
        }
        sb.append('"');
    }

    // ------------------------------------------------------------------
    //  TEXT
    // ------------------------------------------------------------------

    private String renderText() {
        StringBuilder sb = new StringBuilder();
        sb.append("EXPLAIN ").append(mode.name()).append('\n');
        sb.append("PLAN\n");
        appendMap(sb, plan, "  ");
        if (context != null) {
            sb.append("CONTEXT\n");
            appendMap(sb, context, "  ");
        }
        if (predictions != null) {
            sb.append("PREDICTIONS\n");
            sb.append("  task_type: ").append(predictions.taskType().name()).append('\n');
            for (PredictionResult.EntityPrediction p : predictions.predictions()) {
                sb.append("  - id=").append(p.id().raw());
                p.value().ifPresent(v -> sb.append(" value=").append(v));
                p.probability().ifPresent(v -> sb.append(" probability=").append(v));
                if (!p.classProbs().isEmpty()) sb.append(" class_probs=").append(p.classProbs());
                if (!p.ranked().isEmpty()) sb.append(" ranked=").append(p.ranked());
                if (!p.forecast().isEmpty()) sb.append(" forecast=").append(p.forecast());
                p.predictedClass().ifPresent(c -> sb.append(" class=").append(c));
                sb.append('\n');
            }
        }
        return sb.toString();
    }

    @SuppressWarnings("unchecked")
    private static void appendMap(StringBuilder sb, Map<String, Object> map, String indent) {
        for (Map.Entry<String, Object> e : map.entrySet()) {
            Object v = e.getValue();
            if (v instanceof Map<?, ?> m) {
                sb.append(indent).append(e.getKey()).append(":\n");
                appendMap(sb, (Map<String, Object>) m, indent + "  ");
            } else if (v instanceof List<?> list) {
                sb.append(indent).append(e.getKey()).append(":\n");
                for (Object item : list) {
                    if (item instanceof Map<?, ?> im) {
                        sb.append(indent).append("  -\n");
                        appendMap(sb, (Map<String, Object>) im, indent + "    ");
                    } else {
                        sb.append(indent).append("  - ").append(item).append('\n');
                    }
                }
            } else {
                sb.append(indent).append(e.getKey()).append(": ").append(v).append('\n');
            }
        }
    }
}
