//! Output code.
//!
//! For now, we're going to output WASM.  That should let us get interesting output
//! and maybe bootstrap stuff if we feel like it.

use parity_wasm as w;

use crate::ir;
use crate::*;

fn output(cx: &mut Cx, program: &ir::Ir) {
    let mut builder = w::builder::ModuleBuilder::new();
    for decl in program.decls.iter() {
        compile_decl(cx, &mut builder, decl)
    }
}

fn compile_decl(cx: &mut Cx, builder: w::builder::ModuleBuilder, decl: &ir::Decl) {
    use ir::*;
    match decl {
        Decl::Function {
            name,
            signature,
            body,
        } => (),
        Decl::Const {
            name,
            typename,
            init,
        } => {
            let tdef = cx.unintern_type(*typename).clone();
            let wasm_type = compile_type(cx, &tdef);
            builder.global().with_type(wasm_type);
        }
    }
}

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

fn compile_type(cx: &mut Cx, t: &TypeDef) -> w::elements::ValueType {
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
