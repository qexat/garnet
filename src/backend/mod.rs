//! Backend indirection layer.

use crate::hir;
use crate::typeck::Tck;

mod rust;

/// Specifies which backend to use.
#[derive(Copy, Clone, Debug)]
pub enum Backend {
    /// Rust backend
    Rust,
    /// Backend that actually doesn't output anything.
    /// Sometimes useful for debuggin.
    Null,
}

impl std::str::FromStr for Backend {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "rust" => Ok(Backend::Rust),
            "null" => Ok(Backend::Null),
            _ => Err(String::from(
                "Unknown backend!  See the source for which ones are valid.",
            )),
        }
    }
}

/// Produce a binary module output of some kind for the given backend.
pub fn output(backend: Backend, program: &hir::Ir, tck: &Tck) -> Vec<u8> {
    match backend {
        Backend::Rust => rust::output(program, tck),
        Backend::Null => vec![],
    }
}
