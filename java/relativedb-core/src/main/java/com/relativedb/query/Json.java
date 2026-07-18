package com.relativedb.query;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * A tiny, dependency-free JSON reader — just enough to deserialize the fixed,
 * small AST schema {@code pql_parse} emits (see the schema doc). Deliberately
 * no external JSON library: the grammar of what we consume is known and closed.
 *
 * <p>Maps values to: {@link Map}{@code <String,Object>} (object), {@link List}
 * {@code <Object>} (array), {@link String}, {@link Double} (any number),
 * {@link Boolean}, or {@code null}.
 */
final class Json {

    private final String s;
    private int i;

    private Json(String s) { this.s = s; }

    static Object parse(String text) {
        Json j = new Json(text);
        j.ws();
        Object v = j.value();
        j.ws();
        if (j.i != j.s.length()) {
            throw new IllegalArgumentException("trailing JSON at index " + j.i);
        }
        return v;
    }

    private Object value() {
        char c = peek();
        switch (c) {
            case '{': return object();
            case '[': return array();
            case '"': return string();
            case 't': case 'f': return bool();
            case 'n': return nul();
            default:  return number();
        }
    }

    private Map<String, Object> object() {
        expect('{');
        Map<String, Object> m = new LinkedHashMap<>();
        ws();
        if (peek() == '}') { i++; return m; }
        while (true) {
            ws();
            String key = string();
            ws();
            expect(':');
            ws();
            m.put(key, value());
            ws();
            char c = next();
            if (c == '}') return m;
            if (c != ',') throw err("',' or '}'");
        }
    }

    private List<Object> array() {
        expect('[');
        List<Object> list = new ArrayList<>();
        ws();
        if (peek() == ']') { i++; return list; }
        while (true) {
            ws();
            list.add(value());
            ws();
            char c = next();
            if (c == ']') return list;
            if (c != ',') throw err("',' or ']'");
        }
    }

    private String string() {
        expect('"');
        StringBuilder b = new StringBuilder();
        while (true) {
            char c = next();
            if (c == '"') return b.toString();
            if (c == '\\') {
                char e = next();
                switch (e) {
                    case '"':  b.append('"'); break;
                    case '\\': b.append('\\'); break;
                    case '/':  b.append('/'); break;
                    case 'b':  b.append('\b'); break;
                    case 'f':  b.append('\f'); break;
                    case 'n':  b.append('\n'); break;
                    case 'r':  b.append('\r'); break;
                    case 't':  b.append('\t'); break;
                    case 'u':
                        b.append((char) Integer.parseInt(s.substring(i, i + 4), 16));
                        i += 4;
                        break;
                    default: throw err("valid escape");
                }
            } else {
                b.append(c);
            }
        }
    }

    private Boolean bool() {
        if (s.startsWith("true", i)) { i += 4; return Boolean.TRUE; }
        if (s.startsWith("false", i)) { i += 5; return Boolean.FALSE; }
        throw err("boolean");
    }

    private Object nul() {
        if (s.startsWith("null", i)) { i += 4; return null; }
        throw err("null");
    }

    private Double number() {
        int start = i;
        if (peek() == '-' || peek() == '+') i++;
        while (i < s.length()) {
            char c = s.charAt(i);
            if ((c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E'
                    || c == '+' || c == '-') {
                i++;
            } else {
                break;
            }
        }
        if (i == start) throw err("value");
        return Double.parseDouble(s.substring(start, i));
    }

    private void ws() {
        while (i < s.length()) {
            char c = s.charAt(i);
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') i++;
            else break;
        }
    }

    private char peek() {
        if (i >= s.length()) throw err("more input");
        return s.charAt(i);
    }

    private char next() {
        if (i >= s.length()) throw err("more input");
        return s.charAt(i++);
    }

    private void expect(char c) {
        if (next() != c) throw err("'" + c + "'");
    }

    private IllegalArgumentException err(String want) {
        return new IllegalArgumentException(
                "invalid JSON: expected " + want + " at index " + i);
    }
}
