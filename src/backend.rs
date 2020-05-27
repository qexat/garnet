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
    /// Eventually we'll need to dedupe this but we need to preserve order too,
    /// 'cause everything is resolved by indices, so for now we just fake this
    typesec: Vec<elem::Type>,
    /// For now assume we don't need to dedupe whole functions
    codesec: Vec<elem::FuncBody>,
    /// Function signatures
    funcsec: Vec<elem::Func>,
    /// Exports.  Name and typed reference.
    exportsec: Vec<elem::ExportEntry>,
}

impl Module {
    fn new() -> Self {
        Module {
            typesec: Vec::new(),
            codesec: Vec::new(),
            funcsec: Vec::new(),
            exportsec: Vec::new(),
        }
    }

    /// Finally output the thing to actual wasm
    fn output(self) -> Result<Vec<u8>, elem::Error> {
        let typesec = elem::Section::Type(elem::TypeSection::with_types(self.typesec));
        let codesec = elem::Section::Code(elem::CodeSection::with_bodies(self.codesec));
        let funcsec = elem::Section::Function(elem::FunctionSection::with_entries(self.funcsec));
        let exportsec = elem::Section::Export(elem::ExportSection::with_entries(self.exportsec));
        // TODO: The ordering of these sections matters!  Live and learn.
        let sections = vec![typesec, funcsec, exportsec, codesec];
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
            // TODO For now we don't try to dedupe types, just add 'em as redundantly
            // as necessary.
            let typesec_idx = m.typesec.len();
            m.typesec.push(elem::Type::Function(sig));
            let body = compile_function(cx, signature, body);
            m.codesec.push(body);
            let funcsec_idx = m.funcsec.len();
            m.funcsec.push(elem::Func::new(typesec_idx as u32));
            m.exportsec.push(elem::ExportEntry::new(
                cx.unintern(*name).to_owned(),
                elem::Internal::Function(funcsec_idx as u32),
            ));
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

/// Tracks local variables in a function for each type
#[derive(Clone, Default, Debug)]
struct Locals {
    // TODO: Eventually fill these out with whatever var info we need
    i32: Vec<()>,
    i64: Vec<()>,
    f32: Vec<()>,
    f64: Vec<()>,
}

/// Compile a function, specifically -- wasm needs to know about its params and such.
fn compile_function(cx: &mut Cx, sig: &ir::Signature, body: &[ir::Expr]) -> elem::FuncBody {
    let mut locals = vec![];
    let mut isns = vec![];

    use elem::Instruction as I;
    for expr in body {
        use ir::*;
        match expr {
            Expr::Lit { val } => match val {
                Literal::Integer(i) => isns.push(I::I32Const(*i as i32)),
                Literal::Bool(b) => isns.push(I::I32Const(if *b { 1 } else { 0 })),
                Literal::Unit => (),
            },
            _ => panic!("whlarg!"),
        }
    }
    // Functions must end with END
    isns.push(I::End);

    elem::FuncBody::new(locals, elem::Instructions::new(isns))
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
