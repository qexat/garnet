//! Backend indirection layer.

use crate::hir;
use crate::TypeSym;

//mod lrust;
mod rust;

/// Specifies which backend to use.
#[derive(Copy, Clone, Debug)]
pub enum Backend {
    /*
    /// Original wasm backend
    Wasm32,
    /// wasm backend with a LIR layer
    LirWasm32,
    */
    /// Rust backend
    Rust,
    /// Backend that actually doesn't output anything.
    /// Sometimes useful for debuggin.
    Null,
    // /// LIR Rust backend
    // LRust,
}

/// Produce a binary module output of some kind for the given backend.
pub fn output(backend: Backend, program: &hir::Ir<TypeSym>) -> Vec<u8> {
    match backend {
        /*
        Backend::Wasm32 => wasm32::output(cx, program),
        Backend::LirWasm32 => {
            //unimplemented!()
            let lir = crate::lir::lower_hir(cx, &program);
            lwasm32::output(cx, &lir)
        }
        */
        Backend::Rust => rust::output(&program),
        Backend::Null => vec![],
        /*
        Backend::LRust => {
            let lir = crate::lir::lower_hir(cx, &program);
            lrust::output(cx, &lir)
        }
        */
    }
}
