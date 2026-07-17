//! Golden regression gate: feed the PRE-sort golden batch from `cpp/testdata`
//! straight into the native binding and match the PyTorch-verified target
//! scores for BOTH checkpoints (within 2e-3).
//!
//! Gated on the dylib + checkpoints being present (via `RELATIVEDB_RT_LIB` or the
//! sibling `cpp/build`), but RUN when they are. Mirrors the Java
//! `RtGoldenForwardTest` and Python `test_golden_scores_through_ctypes`.

use std::path::{Path, PathBuf};

use relativedb::native::{load_lib, resolve_model_path};

const B: usize = 5;
const S: usize = 16;
const TOL: f32 = 2e-3;

const EXPECTED_CLASSIFICATION: [f32; 5] = [-0.18470, -0.33108, 0.43363, -0.14449, 0.46848];
const EXPECTED_REGRESSION: [f32; 5] = [-0.27052, -0.41538, 0.39998, -0.30649, 0.26804];

fn testdata_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..").join("cpp").join("testdata")
}

fn read_i64(dir: &Path, name: &str, count: usize) -> Vec<i64> {
    let bytes = std::fs::read(dir.join(name)).expect("read bin");
    assert_eq!(bytes.len(), count * 8, "{}", name);
    bytes.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect()
}
fn read_u8(dir: &Path, name: &str, count: usize) -> Vec<u8> {
    let bytes = std::fs::read(dir.join(name)).expect("read bin");
    assert_eq!(bytes.len(), count, "{}", name);
    bytes
}
fn read_f32(dir: &Path, name: &str, count: usize) -> Vec<f32> {
    let bytes = std::fs::read(dir.join(name)).expect("read bin");
    assert_eq!(bytes.len(), count * 4, "{}", name);
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn run_golden(variant: &str, expected: &[f32; 5]) {
    let dir = testdata_dir();
    if !dir.join("manifest.json").exists() {
        eprintln!("SKIP golden ({variant}): cpp/testdata not found");
        return;
    }
    let lib = match load_lib(None) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("SKIP golden ({variant}): librt_c unavailable: {e}");
            return;
        }
    };
    let path = match resolve_model_path(&format!("hf://stanford-star/rt-j/{variant}")) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("SKIP golden ({variant}): checkpoint unavailable: {e}");
            return;
        }
    };

    let n = B * S;
    let node_idxs = read_i64(&dir, "node_idxs.bin", n);
    let f2p = read_i64(&dir, "f2p_nbr_idxs.bin", n * 5);
    let col_idxs = read_i64(&dir, "col_name_idxs.bin", n);
    let table_idxs = read_i64(&dir, "table_name_idxs.bin", n);
    let is_padding = read_u8(&dir, "is_padding.bin", n);
    let sem_types = read_i64(&dir, "sem_types.bin", n);
    let is_target = read_u8(&dir, "is_targets.bin", n);
    let number_v = read_f32(&dir, "number_values.bin", n);
    let datetime_v = read_f32(&dir, "datetime_values.bin", n);
    let boolean_v = read_f32(&dir, "boolean_values.bin", n);
    let text_v = read_f32(&dir, "text_values.bin", n * 384);
    let col_name_v = read_f32(&dir, "col_name_values.bin", n * 384);

    let model = lib.load_model(&path).expect("load_model");
    assert!(model.num_params() > 80_000_000, "unexpected param count");
    let scores = model
        .forward(
            B as i32, S as i32, &node_idxs, &f2p, &col_idxs, &table_idxs, &is_padding,
            &sem_types, &is_target, &number_v, &datetime_v, &boolean_v, &text_v, &col_name_v, 0,
        )
        .expect("forward");
    assert_eq!(scores.len(), B);
    for i in 0..B {
        assert!(
            (scores[i] - expected[i]).abs() < TOL,
            "{variant} score[{i}] = {} expected {} (all: {:?})",
            scores[i],
            expected[i],
            scores
        );
    }
    eprintln!("golden {variant}: OK {:?}", scores);
}

#[test]
fn classification_checkpoint_matches_golden() {
    run_golden("classification", &EXPECTED_CLASSIFICATION);
}

#[test]
fn regression_checkpoint_matches_golden() {
    run_golden("regression", &EXPECTED_REGRESSION);
}
