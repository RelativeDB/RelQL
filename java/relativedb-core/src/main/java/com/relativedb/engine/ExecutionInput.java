package com.relativedb.engine;

import com.relativedb.query.ValidatedQuery;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/** One execution request: the query plus anchor-time and entity overrides. */
public final class ExecutionInput {
    private final String pql;                 // one of pql / validated is set
    private final ValidatedQuery validated;
    private final Instant anchorTime;         // nullable = unbounded max
    private final boolean perEntityAnchor;
    private final Instant contextAnchorTime;  // nullable = same as anchorTime
    private final List<Object> entityIds;     // empty = from query / all
    private final Map<String, Instant> params; // AS OF :name bindings (default empty)

    private ExecutionInput(BuilderImpl b) {
        this.pql = b.pql;
        this.validated = b.validated;
        this.anchorTime = b.anchorTime;
        this.perEntityAnchor = b.perEntityAnchor;
        this.contextAnchorTime = b.contextAnchorTime;
        this.entityIds = List.copyOf(b.entityIds);
        this.params = Map.copyOf(b.params);
    }

    public static Builder newInput() { return new BuilderImpl(); }

    public interface Builder {
        Builder query(String pql);
        Builder query(ValidatedQuery query);
        /** "now"; default: unbounded max. */
        Builder anchorTime(Instant t);
        /** anchor_time="entity" semantics: each entity's own timestamp is its "now". */
        Builder perEntityAnchor(boolean b);
        /** Decouple the context "now" from the prediction "now". */
        Builder contextAnchorTime(Instant t);
        /** Overrides {@code FOR ... IN (...)}. */
        Builder entityIds(List<Object> ids);
        /** Timestamp bindings for {@code AS OF :name} params (name -> instant). */
        Builder params(Map<String, Instant> params);
        ExecutionInput build();
    }

    public Optional<String> pql() { return Optional.ofNullable(pql); }
    public Optional<ValidatedQuery> validatedQuery() { return Optional.ofNullable(validated); }
    public Optional<Instant> anchorTime() { return Optional.ofNullable(anchorTime); }
    public boolean perEntityAnchor() { return perEntityAnchor; }
    public Optional<Instant> contextAnchorTime() { return Optional.ofNullable(contextAnchorTime); }
    public List<Object> entityIds() { return entityIds; }
    public Map<String, Instant> params() { return params; }

    private static final class BuilderImpl implements Builder {
        String pql;
        ValidatedQuery validated;
        Instant anchorTime;
        boolean perEntityAnchor;
        Instant contextAnchorTime;
        List<Object> entityIds = new ArrayList<>();
        Map<String, Instant> params = new java.util.LinkedHashMap<>();

        @Override public Builder query(String pql) {
            this.pql = Objects.requireNonNull(pql); this.validated = null; return this;
        }
        @Override public Builder query(ValidatedQuery query) {
            this.validated = Objects.requireNonNull(query); this.pql = null; return this;
        }
        @Override public Builder anchorTime(Instant t) { this.anchorTime = t; return this; }
        @Override public Builder perEntityAnchor(boolean b) { this.perEntityAnchor = b; return this; }
        @Override public Builder contextAnchorTime(Instant t) { this.contextAnchorTime = t; return this; }
        @Override public Builder entityIds(List<Object> ids) {
            this.entityIds = new ArrayList<>(Objects.requireNonNull(ids)); return this;
        }
        @Override public Builder params(Map<String, Instant> params) {
            this.params = new java.util.LinkedHashMap<>(Objects.requireNonNull(params)); return this;
        }
        @Override public ExecutionInput build() {
            if (pql == null && validated == null) {
                throw new IllegalStateException("ExecutionInput requires a query");
            }
            return new ExecutionInput(this);
        }
    }
}
