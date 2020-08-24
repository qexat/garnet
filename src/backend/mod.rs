use crate::hir;
use crate::{Cx, TypeSym};

mod lwasm32;
mod wasm32;

#[derive(Copy, Clone, Debug)]
pub enum Backend {
    Wasm32,
    LirWasm32,
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
    }
}
