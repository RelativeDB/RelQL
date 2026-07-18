package com.relativedb.query;

import com.relativedb.nat.RtCLib;
import com.relativedb.nat.RtCNative;

import java.nio.charset.StandardCharsets;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalInt;
import java.util.OptionalLong;

/**
 * The Java RelQL parser: delegates to the shared C++ parser ({@code pql_parse} in
 * {@code librt_c}) and deserializes its JSON AST into {@link ParsedQuery}.
 * Grammar and lexing live once in the C++ layer, shared with the Python and
 * Rust bindings; this is the Java analogue of
 * {@code python/src/relativedb/pql/native.py}. {@code librt_c} is a hard
 * dependency — {@link Pql#parse} throws a clear error when it is
 * {@link #available() unavailable}.
 */
public final class NativePqlParser {

    private static final int OUT = 1 << 16;   // 64 KiB — beyond any real AST
    private static final int ERR = 1024;

    private NativePqlParser() { }

    /** True if the native parser can be used. */
    public static boolean available() {
        return RtCNative.isAvailable();
    }

    /**
     * Parse {@code query} with the shared C++ parser. This is the only RelQL
     * parser: {@code librt_c} is a hard dependency, so a missing library is a
     * fatal error rather than a fallback.
     *
     * @throws PqlSyntaxException if the native parser rejects the query
     * @throws IllegalStateException if {@code librt_c} cannot be loaded
     */
    public static ParsedQuery parse(String query) {
        RtCLib lib = RtCNative.get();
        if (lib == null) {
            throw new IllegalStateException(
                    "RelQL parsing requires the native runtime 'librt_c', which could not be "
                    + "loaded. Build it (cd cpp && cmake --build build) and point the "
                    + "'relativedb.rt.lib' system property or RELATIVEDB_RT_LIB env var at the "
                    + "library file. Underlying cause: " + RtCNative.failure());
        }
        byte[] out = new byte[OUT];
        byte[] err = new byte[ERR];
        int rc = lib.pql_parse(query, out, out.length, err, err.length);
        if (rc != 0) {
            throw new PqlSyntaxException(cString(err), 0, 0);
        }
        Object json = Json.parse(cString(out));
        return query(asObject(json));
    }

    // ---- JSON -> AST -------------------------------------------------------

    private static ParsedQuery query(Map<String, Object> o) {
        Map<String, Object> ek = asObject(o.get("entity_key"));
        ColumnRef entityKey = new ColumnRef(str(ek.get("table")), str(ek.get("column")));

        List<Literal> entityIds = new ArrayList<>();
        Object ids = o.get("entity_ids");
        if (ids != null) {
            for (Object v : asArray(ids)) {
                entityIds.add(literal(v));
            }
        }

        Optional<TargetExpr> where = optExpr(o.get("where"));
        Optional<TargetExpr> assuming = optExpr(o.get("assuming"));

        Object rank = o.get("rank");
        Optional<ProblemType> problemType = rank == null
                ? Optional.empty()
                : Optional.of(ProblemType.valueOf(str(rank)));

        OptionalInt topK = o.get("top_k") == null
                ? OptionalInt.empty()
                : OptionalInt.of((int) num(o.get("top_k")));
        OptionalInt numForecasts = o.get("num_forecasts") == null
                ? OptionalInt.empty()
                : OptionalInt.of((int) num(o.get("num_forecasts")));

        Optional<Explain> explain = explain(o.get("explain"));
        Optional<AsOf> asOf = asOf(o.get("as_of"));
        List<Ablation> ablations = ablations(o.get("ablations"));
        Optional<ReturnSpec> ret = returnSpec(o.get("ret"));
        Map<String, Window> windows = windows(o.get("windows"));

        return new ParsedQuery(expr(o.get("target")), entityKey, List.copyOf(entityIds),
                where, assuming, topK, problemType, numForecasts,
                explain, asOf, ablations, ret, windows);
    }

    // ---- query-level clauses ----------------------------------------------

    private static Optional<Explain> explain(Object node) {
        if (node == null) return Optional.empty();
        Map<String, Object> e = asObject(node);
        return Optional.of(new Explain(
                Explain.Mode.valueOf(str(e.get("mode"))),
                Explain.Format.valueOf(str(e.get("format")))));
    }

    private static Optional<AsOf> asOf(Object node) {
        if (node == null) return Optional.empty();
        Map<String, Object> a = asObject(node);
        AsOf.Kind kind = AsOf.Kind.valueOf(str(a.get("kind")).toUpperCase());
        Object value = a.get("value");
        return Optional.of(new AsOf(kind, value == null ? null : str(value)));
    }

    private static List<Ablation> ablations(Object node) {
        if (node == null) return List.of();
        List<Ablation> out = new ArrayList<>();
        for (Object x : asArray(node)) {
            Map<String, Object> m = asObject(x);
            out.add(new Ablation(str(m.get("kind")), str(m.get("name"))));
        }
        return List.copyOf(out);
    }

    private static Optional<ReturnSpec> returnSpec(Object node) {
        if (node == null) return Optional.empty();
        Map<String, Object> r = asObject(node);
        ReturnSpec.Kind kind = ReturnSpec.Kind.valueOf(str(r.get("kind")));
        Object qs = r.get("quantiles");
        double[] quantiles;
        if (qs == null) {
            quantiles = new double[0];
        } else {
            List<?> list = asArray(qs);
            quantiles = new double[list.size()];
            for (int i = 0; i < list.size(); i++) quantiles[i] = num(list.get(i));
        }
        OptionalInt interval = r.get("interval") == null
                ? OptionalInt.empty()
                : OptionalInt.of((int) num(r.get("interval")));
        return Optional.of(new ReturnSpec(kind, quantiles, interval));
    }

    private static Map<String, Window> windows(Object node) {
        if (node == null) return Map.of();
        Map<String, Window> out = new LinkedHashMap<>();
        for (Map.Entry<String, Object> e : asObject(node).entrySet()) {
            out.put(e.getKey(), window(e.getValue()));
        }
        return out;
    }

    private static Window window(Object node) {
        Map<String, Object> w = asObject(node);
        OptionalLong start = OptionalLong.of(bound(w.get("start")));
        long end = bound(w.get("end"));
        Object unit = w.get("unit");
        TimeUnit timeUnit = unit == null
                ? TimeUnit.DAYS
                : TimeUnit.valueOf(str(unit).toUpperCase());
        long horizons = w.get("horizons") == null ? 1L : (long) num(w.get("horizons"));
        OptionalLong step = w.get("step") == null
                ? OptionalLong.empty()
                : OptionalLong.of((long) num(w.get("step")));
        return new Window(start, end, timeUnit, horizons, step);
    }

    private static Optional<TargetExpr> optExpr(Object o) {
        return o == null ? Optional.empty() : Optional.of(expr(o));
    }

    private static TargetExpr expr(Object node) {
        Map<String, Object> o = asObject(node);
        String kind = str(o.get("kind"));
        switch (kind) {
            case "col":
                return new ColumnRef(str(o.get("table")), str(o.get("column")));
            case "agg":
                return aggregation(o);
            case "cond": {
                Object rightExprNode = o.get("right_expr");
                Optional<TargetExpr> rightExpr = rightExprNode == null
                        ? Optional.empty() : Optional.of(expr(rightExprNode));
                return new Condition(expr(o.get("left")),
                        Operator.valueOf(str(o.get("op"))),
                        literal(o.get("right")), rightExpr);
            }
            case "logic":
                return new LogicalOp(expr(o.get("left")),
                        BoolOp.valueOf(str(o.get("op"))), expr(o.get("right")));
            case "not":
                return new Not(expr(o.get("expr")));
            case "arith":
                return new Arith(str(o.get("op")).charAt(0),
                        expr(o.get("left")), expr(o.get("right")));
            case "func": {
                List<TargetExpr> args = new ArrayList<>();
                for (Object a : asArray(o.get("args"))) args.add(expr(a));
                return new Func(str(o.get("name")), List.copyOf(args));
            }
            case "case": {
                List<Case.When> whens = new ArrayList<>();
                for (Object wobj : asArray(o.get("whens"))) {
                    Map<String, Object> wm = asObject(wobj);
                    whens.add(new Case.When(expr(wm.get("cond")), expr(wm.get("then"))));
                }
                Object elseNode = o.get("else");
                return new Case(List.copyOf(whens),
                        elseNode == null ? null : expr(elseNode));
            }
            case "lit":
                return new LitExpr(literal(o.get("value")));
            default:
                throw new IllegalArgumentException("unknown expr kind: " + kind);
        }
    }

    private static Aggregation aggregation(Map<String, Object> o) {
        AggFunc func = AggFunc.valueOf(str(o.get("func")));
        Map<String, Object> col = asObject(o.get("column"));
        ColumnRef column = new ColumnRef(str(col.get("table")), str(col.get("column")));

        Object filterNode = o.get("filter");
        Optional<Filter> filter = filterNode == null
                ? Optional.empty()
                : Optional.of(new Filter(expr(filterNode)));

        Object windowNode = o.get("window");
        if (windowNode == null) {
            return new Aggregation(func, column, filter, OptionalLong.empty(), 0L, null,
                    1L, OptionalLong.empty());
        }
        Window w = window(windowNode);
        return new Aggregation(func, column, filter, w.start(), w.end(), w.unit(),
                w.horizons(), w.step());
    }

    private static long bound(Object v) {
        if (v instanceof String) {
            String s = (String) v;
            if (s.equals("inf")) return Aggregation.POS_INF;
            if (s.equals("-inf")) return Aggregation.NEG_INF;
            throw new IllegalArgumentException("bad bound: " + s);
        }
        return (long) num(v);
    }

    private static Literal literal(Object v) {
        if (v == null) return Literal.NULL;
        if (v instanceof String) return Literal.string((String) v);
        if (v instanceof Double) return Literal.number((Double) v);
        if (v instanceof Boolean) return Literal.bool((Boolean) v);
        if (v instanceof List) {
            List<Literal> items = new ArrayList<>();
            for (Object e : (List<?>) v) items.add(literal(e));
            return Literal.list(items);
        }
        if (v instanceof Map) {
            Map<?, ?> m = (Map<?, ?>) v;
            Object date = m.get("date");
            if (date != null) return Literal.date(parseDate(str(date)));
        }
        throw new IllegalArgumentException("unsupported literal: " + v);
    }

    private static LocalDateTime parseDate(String text) {
        if (text.length() > 10) {
            return LocalDateTime.of(LocalDate.parse(text.substring(0, 10)),
                    LocalTime.parse(text.substring(11)));
        }
        return LocalDate.parse(text).atStartOfDay();
    }

    // ---- helpers -----------------------------------------------------------

    @SuppressWarnings("unchecked")
    private static Map<String, Object> asObject(Object o) {
        if (!(o instanceof Map)) throw new IllegalArgumentException("expected object, got " + o);
        return (Map<String, Object>) o;
    }

    private static List<?> asArray(Object o) {
        if (!(o instanceof List)) throw new IllegalArgumentException("expected array, got " + o);
        return (List<?>) o;
    }

    private static String str(Object o) {
        if (!(o instanceof String)) throw new IllegalArgumentException("expected string, got " + o);
        return (String) o;
    }

    private static double num(Object o) {
        if (!(o instanceof Double)) throw new IllegalArgumentException("expected number, got " + o);
        return (Double) o;
    }

    private static String cString(byte[] buf) {
        int n = 0;
        while (n < buf.length && buf[n] != 0) n++;
        return new String(buf, 0, n, StandardCharsets.UTF_8);
    }
}
