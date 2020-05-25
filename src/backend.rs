//! Output code.
//!
//! For now, we're going to output WASM.  That should let us get interesting output
//! and maybe bootstrap stuff if we feel like it.

use parity_wasm::{self as w, elements as elem};

use crate::ir;
use crate::*;

/// Entry point to turn the IR into a compiled wasm module
pub fn output(cx: &mut Cx, program: &ir::Ir) {
    let mut builder = w::builder::ModuleBuilder::new();
    for decl in program.decls.iter() {
        builder = compile_decl(cx, builder, decl);
    }
}

fn compile_decl(
    cx: &mut Cx,
    builder: w::builder::ModuleBuilder,
    decl: &ir::Decl,
) -> w::builder::ModuleBuilder {
    use ir::*;
    match decl {
        Decl::Function {
            name,
            signature,
            body,
        } => {
            let params: Vec<elem::ValueType> = signature
                .params
                .iter()
                .map(|(_varsym, typesym)| {
                    let typedef = cx.unintern_type(*typesym);
                    compile_type(cx, typedef)
                })
                .collect();
            let ret = compile_type(cx, cx.unintern_type(signature.rettype));
            let x1: parity_wasm::builder::code::SignatureBuilder = builder.function().signature();
            let x2 = x1.with_params(params).with_return_type(Some(ret)).build();
            let x3 = x2.build();
            x3
            //let funbuilder = builder.function();
            //let sigbuilder: () = builder.function().signature();
            //let xig = build_function_signature(cx, x1, signature);
            //function.build()
            //xig.build()
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
            //builder = builder.global().with_type(wasm_type);
            builder
        }
    }
}

/*
fn build_function_signature(
    cx: &mut Cx,
    builder: w::builder::SignatureBuilder,
    sig: &ir::Signature,
) -> w::builder::FunctionBuilder {
    //) -> elem::FunctionType {
    let params: Vec<elem::ValueType> = sig
        .params
        .iter()
        .map(|(_varsym, typesym)| {
            let typedef = cx.unintern_type(*typesym);
            compile_type(cx, typedef)
        })
        .collect();
    let ret = compile_type(cx, cx.unintern_type(sig.rettype));
    builder
        .with_params(params)
        .with_return_type(Some(ret))
        .build()
}
*/

fn compile_expr(cx: &mut Cx, builder: &mut w::builder::ModuleBuilder, expr: &ir::Expr) {
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
