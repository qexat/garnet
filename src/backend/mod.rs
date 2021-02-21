//! Backend indirection layer.

use crate::hir;
use crate::{Cx, TypeSym};

mod lwasm32;
mod rust;
mod wasm32;

/// Specifies which backend to use.
#[derive(Copy, Clone, Debug)]
pub enum Backend {
    /// Original wasm backend
    Wasm32,
    /// wasm backend with a LIR layer
    LirWasm32,
    /// Rust backend
    Rust,
}

/// Produce a binary module output of some kind for the given backend.
pub fn output(backend: Backend, cx: &Cx, program: &hir::Ir<TypeSym>) -> Vec<u8> {
    match backend {
        Backend::Wasm32 => wasm32::output(cx, program),
        Backend::LirWasm32 => {
            //unimplemented!()
            let lir = crate::lir::lower_hir(cx, &program);
            lwasm32::output(cx, &lir)
        }
        Backend::Rust => {
            //let lir = crate::lir::lower_hir(cx, &program);
            rust::output(cx, &program)
        }
    }
}
