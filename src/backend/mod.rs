use crate::ir;
use crate::{Cx, TypeSym};

mod wasm32;

#[derive(Copy, Clone, Debug)]
pub enum Backend {
    Wasm32,
}

/// Produce a binary module output of some kind for the given backend.
pub fn output(backend: Backend, cx: &Cx, program: &ir::Ir<TypeSym>) -> Vec<u8> {
    match backend {
        Backend::Wasm32 => wasm32::output(cx, program),
    }
}