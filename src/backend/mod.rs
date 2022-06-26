//! Backend indirection layer.

use crate::hir;
use crate::typeck::Tck;

mod qbe;
mod rust;

/// Specifies which backend to use.
#[derive(Copy, Clone, Debug)]
pub enum Backend {
    /// Rust backend
    Rust,
    /// QBE backend
    Qbe,
    /// Backend that actually doesn't output anything.
    /// Sometimes useful for debuggin.
    Null,
}

/// Produce a binary module output of some kind for the given backend.
pub fn output(backend: Backend, program: &hir::Ir, tck: &Tck) -> Vec<u8> {
    match backend {
        Backend::Rust => rust::output(program, tck),
        Backend::Qbe => qbe::output(program, tck),
        Backend::Null => vec![],
    }
}
