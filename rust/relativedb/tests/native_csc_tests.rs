//! Cross-language conformance: the shared C++ CSC adjacency (`csc_build` /
//! `csc_children` in `librt_c`) must agree with a brute-force reference of the
//! `crate::csc::CscIndex::children` semantics, node for node, across randomized
//! graphs. This is the guardrail that lets the CSC hot path live in one place
//! (C++) without the language bindings diverging.
//!
//! Mirrors the Python `test_native_csc.py`. Skips cleanly (passes without
//! asserting) if `librt_c` has not been built. RNG is a fixed-seed
//! deterministic LCG — no time-based randomness.

use relativedb::csc_native::{native_available, NativeCsc};

const NEG_INF: f64 = f64::NEG_INFINITY;
const POS_INF: f64 = f64::INFINITY;

/// Deterministic linear-congruential generator (glibc constants). Fixed seed
/// in, fully reproducible sequence out.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Lcg {
        Lcg(seed.wrapping_mul(2862933555777941757).wrapping_add(3037000493))
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
    /// Uniform integer in [lo, hi] inclusive.
    fn range(&mut self, lo: i64, hi: i64) -> i64 {
        let span = (hi - lo + 1) as u64;
        lo + (self.next_u64() % span) as i64
    }
}

/// Brute-force reference matching `crate::csc::CscIndex::children`: within the
/// parent's bucket sorted by ts asc (stable), keep ts <= anchor, take the last
/// `limit` reversed to newest-first. Static rows use ts = -inf.
fn ref_children(
    parent: i64,
    anchor: f64,
    limit: i32,
    ep: &[i64],
    ec: &[i64],
    et: &[f64],
) -> Vec<i64> {
    if limit <= 0 {
        return Vec::new();
    }
    let mut bucket: Vec<usize> = (0..ep.len()).filter(|&i| ep[i] == parent).collect();
    // stable sort by ts asc (ties preserve edge order)
    bucket.sort_by(|&a, &b| et[a].partial_cmp(&et[b]).unwrap());
    let admitted: Vec<usize> = bucket.into_iter().filter(|&i| et[i] <= anchor).collect();
    let start = admitted.len().saturating_sub(limit as usize);
    admitted[start..].iter().rev().map(|&i| ec[i]).collect()
}

fn random_graph(rng: &mut Lcg, n_parents: i64, n_edges: usize) -> (Vec<i64>, Vec<i64>, Vec<f64>) {
    let mut ep = Vec::with_capacity(n_edges);
    let mut ec = Vec::with_capacity(n_edges);
    let mut et = Vec::with_capacity(n_edges);
    for _ in 0..n_edges {
        ep.push(rng.range(0, n_parents - 1));
        ec.push(rng.range(0, 100_000));
        let roll = rng.range(0, 9);
        if roll == 0 {
            et.push(NEG_INF); // static row
        } else if roll <= 3 {
            et.push(rng.range(0, 5) as f64); // heavy ties
        } else {
            et.push(rng.range(0, 1000) as f64);
        }
    }
    (ep, ec, et)
}

#[test]
fn matches_reference() {
    if !native_available() {
        eprintln!("SKIP native_csc::matches_reference — librt_c not loadable");
        return;
    }
    let cases: &[(u64, i64, usize)] = &[
        (1, 1, 20),
        (2, 8, 200),
        (3, 50, 2000),
        (4, 200, 50),   // sparse: many parents with no edges
        (5, 4, 3000),   // dense: heavy ties per parent
    ];
    let mut total_triples = 0usize;
    for &(seed, n_parents, n_edges) in cases {
        let mut rng = Lcg::new(seed);
        let (ep, ec, et) = random_graph(&mut rng, n_parents, n_edges);
        let idx = NativeCsc::new(n_parents, &ep, &ec, &et).expect("csc_build");

        let mut anchors: Vec<f64> = vec![NEG_INF, POS_INF];
        for v in -2..8 {
            anchors.push(v as f64);
        }
        for _ in 0..2000 {
            let parent = rng.range(-2, n_parents); // includes out-of-range
            let limit = rng.range(-1, 8) as i32; // includes 0 and > bucket
            let anchor = if rng.range(0, 1) == 0 {
                anchors[(rng.range(0, anchors.len() as i64 - 1)) as usize]
            } else {
                rng.range(-5, 1005) as f64
            };
            let got = idx.children(parent, anchor, limit);
            let want = ref_children(parent, anchor, limit, &ep, &ec, &et);
            assert_eq!(got, want, "seed={} parent={} anchor={} limit={}", seed, parent, anchor, limit);
            total_triples += 1;
        }
    }
    println!("native_csc::matches_reference checked {} (parent,anchor,limit) triples", total_triples);
    assert!(total_triples > 0);
}

#[test]
fn edge_cases() {
    if !native_available() {
        eprintln!("SKIP native_csc::edge_cases — librt_c not loadable");
        return;
    }
    // Empty graph: every query is empty.
    let empty = NativeCsc::new(5, &[], &[], &[]).expect("build empty");
    assert_eq!(empty.children(2, 1e9, 4), Vec::<i64>::new());
    // limit 0 and negative -> empty.
    let idx = NativeCsc::new(1, &[0, 0], &[7, 8], &[1.0, 2.0]).expect("build");
    assert_eq!(idx.children(0, 100.0, 0), Vec::<i64>::new());
    assert_eq!(idx.children(0, 100.0, -3), Vec::<i64>::new());
    // anchor before all -> empty; static row (-inf) admitted under every bound.
    let idx2 = NativeCsc::new(1, &[0, 0], &[9, 10], &[NEG_INF, 5.0]).expect("build");
    assert_eq!(idx2.children(0, -100.0, 4), vec![9]); // only the static row
    assert_eq!(idx2.children(0, 100.0, 4), vec![10, 9]); // newest-first
}
