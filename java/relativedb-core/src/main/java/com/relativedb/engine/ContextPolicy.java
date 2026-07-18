package com.relativedb.engine;

/**
 * Context assembly knobs, storage-agnostic. Two budget geometries are
 * supported: RT's global cell budget (maxContextCells + uniform bfsWidth) and
 * KumoRFM's per-hop fanout caps ({@link Builder#fanouts(int...)}, which
 * override bfsWidth when set).
 */
public final class ContextPolicy {
    private final int maxContextCells;
    private final int bfsWidth;
    private final int[] fanouts;      // nullable
    private final int maxHops;
    private final int cohortSize;
    private final boolean preferLatest;

    private ContextPolicy(BuilderImpl b) {
        this.maxContextCells = b.maxContextCells;
        this.bfsWidth = b.bfsWidth;
        this.fanouts = b.fanouts;
        this.maxHops = b.maxHops;
        this.cohortSize = b.cohortSize;
        this.preferLatest = b.preferLatest;
    }

    public static ContextPolicy defaults() { return newPolicy().build(); }

    public static Builder newPolicy() { return new BuilderImpl(); }

    public interface Builder {
        /** RT-style: global cell budget, e.g. 1024–8192. */
        Builder maxContextCells(int n);
        /** Children per row (F23). */
        Builder bfsWidth(int w);
        /**
         * KumoRFM-style per-hop fanout caps (overrides bfsWidth when set),
         * e.g. {@code fanouts(64, 64)} ≈ KumoRFM NORMAL with num_hops=2.
         * Also sets maxHops to the number of fanouts.
         */
        Builder fanouts(int... perHop);
        Builder maxHops(int hops);
        /** Similar entities (Tier 1). */
        Builder cohortSize(int n);
        /** Newest-first child selection. */
        Builder preferLatest(boolean b);
        ContextPolicy build();
    }

    public int maxContextCells() { return maxContextCells; }
    public int bfsWidth() { return bfsWidth; }
    public int maxHops() { return maxHops; }
    public int cohortSize() { return cohortSize; }
    public boolean preferLatest() { return preferLatest; }
    public boolean hasFanouts() { return fanouts != null; }

    /** The child cap at {@code hop} (0-based): fanouts[hop] if set, else bfsWidth. */
    public int fanoutAt(int hop) {
        if (fanouts == null || fanouts.length == 0) return bfsWidth;
        return fanouts[Math.min(hop, fanouts.length - 1)];
    }

    private static final class BuilderImpl implements Builder {
        int maxContextCells = 4096;
        int bfsWidth = 32;
        int[] fanouts;
        int maxHops = 2;
        int cohortSize = 0;
        boolean preferLatest = true;

        @Override public Builder maxContextCells(int n) {
            if (n <= 0) throw new IllegalArgumentException("maxContextCells must be > 0");
            maxContextCells = n; return this;
        }
        @Override public Builder bfsWidth(int w) {
            if (w <= 0) throw new IllegalArgumentException("bfsWidth must be > 0");
            bfsWidth = w; return this;
        }
        @Override public Builder fanouts(int... perHop) {
            if (perHop == null || perHop.length == 0) {
                throw new IllegalArgumentException("fanouts requires at least one hop cap");
            }
            for (int f : perHop) if (f <= 0) throw new IllegalArgumentException("fanout must be > 0");
            fanouts = perHop.clone();
            maxHops = perHop.length;
            return this;
        }
        @Override public Builder maxHops(int hops) {
            if (hops <= 0) throw new IllegalArgumentException("maxHops must be > 0");
            maxHops = hops; return this;
        }
        @Override public Builder cohortSize(int n) {
            if (n < 0) throw new IllegalArgumentException("cohortSize must be >= 0");
            cohortSize = n; return this;
        }
        @Override public Builder preferLatest(boolean b) { preferLatest = b; return this; }
        @Override public ContextPolicy build() { return new ContextPolicy(this); }
    }
}
