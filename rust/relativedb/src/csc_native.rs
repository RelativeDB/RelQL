//! Native CSC adjacency binding — delegates the time-bounded "latest <= anchor"
//! children query to the shared C++ implementation (`csc_build` / `csc_children`
//! / `csc_free` in `librt_c`).
//!
//! Mirrors `relativedb.csc_native` (Python) / the Java peer: one array
//! algorithm for all three bindings. Callers keep their own id<->dense mapping
//! and row storage (see [`crate::csc::CscIndex`]); only the CSC/CSR arrays and
//! the binary-searched children query live here. Falls back transparently when
//! `librt_c` is not built ([`native_available`] is `false`).

use std::ffi::{c_char, c_void};
use std::path::Path;
use std::sync::OnceLock;

use libloading::Library;

const ERR: usize = 1024;

type CscBuildFn = unsafe extern "C" fn(
    i64,
    i64,
    *const i64,
    *const i64,
    *const f64,
    *mut c_char,
    usize,
) -> *mut c_void;
type CscFreeFn = unsafe extern "C" fn(*mut c_void);
type CscChildrenFn = unsafe extern "C" fn(
    *const c_void,
    i64,
    f64,
    i32,
    *mut i64,
    *mut i32,
    *mut c_char,
    usize,
) -> i32;

struct CscLib {
    _lib: Library,
    build: CscBuildFn,
    free: CscFreeFn,
    children: CscChildrenFn,
}

// The bound symbols are plain reentrant C functions (see csc_c.h).
unsafe impl Send for CscLib {}
unsafe impl Sync for CscLib {}

fn lib() -> Option<&'static CscLib> {
    static LIB: OnceLock<Option<CscLib>> = OnceLock::new();
    LIB.get_or_init(load).as_ref()
}

fn load() -> Option<CscLib> {
    for cand in crate::native::candidate_lib_paths() {
        if cand.is_empty() || !Path::new(&cand).exists() {
            continue;
        }
        unsafe {
            let l = match Library::new(&cand) {
                Ok(l) => l,
                Err(_) => continue,
            };
            let build: CscBuildFn = match l.get::<CscBuildFn>(b"csc_build\0") {
                Ok(s) => *s,
                Err(_) => continue,
            };
            let free: CscFreeFn = match l.get::<CscFreeFn>(b"csc_free\0") {
                Ok(s) => *s,
                Err(_) => continue,
            };
            let children: CscChildrenFn = match l.get::<CscChildrenFn>(b"csc_children\0") {
                Ok(s) => *s,
                Err(_) => continue,
            };
            return Some(CscLib { _lib: l, build, free, children });
        }
    }
    None
}

/// Whether the shared C++ CSC index (`librt_c`) is loadable.
pub fn native_available() -> bool {
    lib().is_some()
}

fn cstr(buf: &[u8]) -> String {
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..end]).into_owned()
}

/// Wraps a native `csc_index`. Build once from edge arrays, then answer many
/// [`children`](Self::children) queries. Frees the native handle on [`Drop`].
pub struct NativeCsc {
    lib: &'static CscLib,
    handle: *mut c_void,
}

// The native handle is only touched behind &self (children) or in Drop; the C
// index is immutable after build.
unsafe impl Send for NativeCsc {}
unsafe impl Sync for NativeCsc {}

impl NativeCsc {
    /// Build from edge arrays (`edge_parent`/`edge_child`/`edge_ts` must have
    /// equal length). Parent/child are dense ids; ts is epoch seconds (use
    /// `f64::NEG_INFINITY` for static rows so they sort first).
    pub fn new(
        n_parents: i64,
        edge_parent: &[i64],
        edge_child: &[i64],
        edge_ts: &[f64],
    ) -> Result<NativeCsc, String> {
        let lib = lib().ok_or_else(|| "librt_c unavailable (build cpp/ with cmake)".to_string())?;
        let n_edges = edge_parent.len();
        if edge_child.len() != n_edges || edge_ts.len() != n_edges {
            return Err("edge arrays must have equal length".to_string());
        }
        let mut err = vec![0u8; ERR];
        let (ep, ec, et) = if n_edges > 0 {
            (edge_parent.as_ptr(), edge_child.as_ptr(), edge_ts.as_ptr())
        } else {
            (std::ptr::null(), std::ptr::null(), std::ptr::null())
        };
        let handle = unsafe {
            (lib.build)(
                n_parents,
                n_edges as i64,
                ep,
                ec,
                et,
                err.as_mut_ptr() as *mut c_char,
                ERR,
            )
        };
        if handle.is_null() {
            let m = cstr(&err);
            return Err(if m.is_empty() { "csc_build failed".to_string() } else { m });
        }
        Ok(NativeCsc { lib, handle })
    }

    /// Up to `limit` dense child ids with `ts <= anchor_ts`, newest-first.
    /// `limit <= 0` or an out-of-range parent yields an empty vec.
    pub fn children(&self, parent_dense: i64, anchor_ts: f64, limit: i32) -> Vec<i64> {
        if limit <= 0 {
            return Vec::new();
        }
        let mut out = vec![0i64; limit as usize];
        let mut n: i32 = 0;
        let mut err = vec![0u8; ERR];
        let rc = unsafe {
            (self.lib.children)(
                self.handle,
                parent_dense,
                anchor_ts,
                limit,
                out.as_mut_ptr(),
                &mut n as *mut i32,
                err.as_mut_ptr() as *mut c_char,
                ERR,
            )
        };
        if rc != 0 {
            return Vec::new();
        }
        out.truncate(n.max(0) as usize);
        out
    }
}

impl Drop for NativeCsc {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { (self.lib.free)(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}
