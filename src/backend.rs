//! Output code.
//!
//! For now, we're going to output WASM.  That should let us get interesting output
//! and maybe bootstrap stuff if we feel like it.

use std::collections::HashSet;

use parity_wasm::{self as w, elements as elem};

use crate::ir;
use crate::*;

/// A useful structure that lets us put a wasm module
/// together piecemeal.  parity_wasm does things like
/// preserve order of segments that are irrelevant to us.
#[derive(Debug, Clone)]
struct Module {
    typesec: HashSet<elem::Type>,
    codesec: HashSet<elem::FuncBody>,
}

impl Module {
    fn new() -> Self {
        Module {
            typesec: HashSet::new(),
            codesec: HashSet::new(),
        }
    }

    /// Finally output the thing to actual wasm
    fn output(self) -> Result<Vec<u8>, elem::Error> {
        let typesec = elem::Section::Type(elem::TypeSection::with_types(
            self.typesec.into_iter().collect(),
        ));
        let codesec = elem::Section::Code(elem::CodeSection::with_bodies(
            self.codesec.into_iter().collect(),
        ));
        let sections = vec![typesec, codesec];
        let m = elem::Module::new(sections);
        m.to_bytes()
    }
}

/// Entry point to turn the IR into a compiled wasm module
pub fn output(cx: &mut Cx, program: &ir::Ir) -> Vec<u8> {
    let mut m = Module::new();
    for decl in program.decls.iter() {
        compile_decl(cx, &mut m, decl);
    }
    m.output().unwrap()
}

fn compile_decl(cx: &mut Cx, m: &mut Module, decl: &ir::Decl) {
    use ir::*;
    match decl {
        Decl::Function {
            name,
            signature,
            body,
        } => {
            let sig = function_signature(cx, m, signature);
            m.typesec.insert(elem::Type::Function(sig));
        }
        Decl::Const {
            name,
            typename,
            init,
        } => {
            // Okay this is actually harder than it looks because
            // wasm only lets globals be value types.
            // ...and if it's really const then why do we need it anyway
            let tdef = cx.unintern_type(*typename).clone();
            let wasm_type = compile_type(cx, &tdef);
        }
    }
}

fn function_signature(cx: &mut Cx, m: &mut Module, sig: &ir::Signature) -> elem::FunctionType {
    let params: Vec<elem::ValueType> = sig
        .params
        .iter()
        .map(|(_varsym, typesym)| {
            let typedef = cx.unintern_type(*typesym);
            compile_type(cx, typedef)
        })
        .collect();
    let rettypedef = cx.unintern_type(sig.rettype);
    let rettype = compile_type(cx, rettypedef);
    elem::FunctionType::new(params, Some(rettype))
}

fn compile_expr(cx: &mut Cx, expr: &ir::Expr) {
    use ir::*;
    match expr {
        Expr::Lit { val } => match val {
            Literal::Integer(i) => {}
            Literal::Bool(b) => (),
            Literal::Unit => (),
        },
        _ => panic!("whlarg!"),
    }
}

fn compile_type(cx: &Cx, t: &TypeDef) -> elem::ValueType {
    match t {
        TypeDef::Unknown => panic!("Can't happen!"),
        TypeDef::Ref(_) => panic!("Can't happen"),
        TypeDef::SInt(4) => w::elements::ValueType::I32,
        TypeDef::SInt(_) => panic!("TODO"),
        TypeDef::Bool => w::elements::ValueType::I32,
        TypeDef::Tuple(_) => panic!("Unimplemented"),
        TypeDef::Lambda(_, _) => panic!("Unimplemented"),
    }
}
