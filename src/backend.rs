//! Output code.
//!
//! For now, we're going to output WASM.  That should let us get interesting output
//! and maybe bootstrap stuff if we feel like it.

use std::cmp;
use std::collections::HashMap;

use wasm_builder::{instr as i, module as m, sections as s, types as t};

use crate::ir;
use crate::*;

/// Entry point to turn the IR into a compiled wasm module
pub fn output(cx: &mut Cx, program: &ir::Ir) -> Vec<u8> {
    let mut m = m::Module::new();
    for decl in program.decls.iter() {
        compile_decl(cx, &mut m, decl);
    }
    let mut output = std::io::Cursor::new(vec![]);
    m.encode(&mut output).unwrap();
    output.into_inner()
}

fn compile_decl(cx: &mut Cx, m: &mut m::Module, decl: &ir::Decl) {
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
            let typesec_idx = m.types.len();
            let funcsec_idx = m.functions.len();
            let body = compile_function(cx, signature, body);

            m.code.push(body);
            m.types.push(sig);
            m.functions.push(typesec_idx as u32);
            m.exports.push(s::Export {
                name: cx.unintern(*name).to_owned(),
                desc: s::Desc::Function(funcsec_idx as u32),
            });
        }
        Decl::Const {
            name,
            typename,
            init,
        } => {
            // Okay this is actually harder than it looks because
            // wasm only lets globals be value types.
            // ...and if it's really const then why do we need it anyway
            let wasm_type = compile_typesym(cx, *typename);
        }
    }
}

fn function_signature(cx: &mut Cx, m: &mut m::Module, sig: &ir::Signature) -> t::FunctionType {
    let params: Vec<t::ValType> = sig
        .params
        .iter()
        .map(|(_varsym, typesym)| compile_typesym(cx, *typesym))
        .collect();
    let rettype = compile_typesym(cx, sig.rettype);
    t::FunctionType {
        parameter_types: params,
        return_types: vec![rettype],
    }
}
#[derive(Copy, Clone, Debug, PartialEq)]
struct LocalVar {
    local_idx: u32,
    wasm_type: t::ValType,
}

/// Order by local index
impl cmp::PartialOrd for LocalVar {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.local_idx.cmp(&other.local_idx))
    }
}

/// Order by local index
impl cmp::Ord for LocalVar {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.local_idx.cmp(&other.local_idx)
    }
}

impl cmp::Eq for LocalVar {}

/// Compile a function, specifically -- wasm needs to know about its params and such.
fn compile_function(cx: &mut Cx, sig: &ir::Signature, body: &[ir::Expr]) -> s::Function {
    // A simple, scope-less symbol table.
    let mut locals: HashMap<VarSym, LocalVar> = HashMap::new();
    let mut isns = vec![];

    // Function params are passed in the "locals" section.
    // So, we add them all to the symbol table.
    for (pname, ptype) in sig.params.iter() {
        let p = LocalVar {
            local_idx: locals.len() as u32,
            wasm_type: compile_typesym(cx, *ptype),
        };
        locals.insert(*pname, p);
    }

    // Compile the actual thing.
    let _ = compile_exprs(cx, &mut locals, &mut isns, body);

    // Turn locals into an array that just contains the types.
    let mut local_arr: Vec<_> = locals.values().map(|l| l).collect();
    local_arr.sort();
    let wasm_locals = local_arr
        .into_iter()
        .map(|l| s::Local {
            n: 1,
            ty: l.wasm_type,
        })
        .collect();

    s::Function {
        locals: wasm_locals,
        body: i::Expr(isns),
    }
}

/// Compile multiple exprs, making sure they don't leave unused
/// values on the stack by adding drop's as necessary.
/// Returns the number of values left on the stack at the end.
fn compile_exprs(
    cx: &mut Cx,
    locals: &mut HashMap<VarSym, LocalVar>,
    isns: &mut Vec<i::Instruction>,
    exprs: &[ir::Expr],
) -> usize {
    // TODO Implement
    // I considered making this a step in the IR that basically turns
    // every `foo(); bar()` into `ignore(foo()); bar()` explicitly,
    // but how hard to ignore it is dependent on the return type and
    // so doing it here seems easier.  IR doesn't explicitly store
    // the return type of each expression... though maybe it should.
    exprs.iter().for_each(|expr| {
        compile_expr(cx, locals, isns, expr);
    });
    0
}

/// Generates code to evaluate the given expr, inserting the instructions into
/// the given instruction list, leaving values on the stack.
/// Returns how many values it leaves on the stack, so we know how many
/// values to get rid of if the return val is ignored.
fn compile_expr(
    cx: &mut Cx,
    locals: &mut HashMap<VarSym, LocalVar>,
    isns: &mut Vec<i::Instruction>,
    expr: &ir::Expr,
) -> usize {
    use i::Instruction as I;
    use ir::Expr as E;
    use ir::*;
    match expr {
        E::Lit { val } => match val {
            Literal::Integer(i) => {
                isns.push(I::Const(i::Literal::I32(*i as i32)));
                1
            }
            Literal::Bool(b) => {
                isns.push(I::Const(i::Literal::I32(if *b { 1 } else { 0 })));
                1
            }
            Literal::Unit => 0,
        },
        E::Var { name } => {
            let ldef = locals
                .get(name)
                .expect(&format!("Unknown local {:?}; should never happen", name));
            isns.push(I::LocalGet(ldef.local_idx));
            0
        }
        E::BinOp { op, lhs, rhs } => {
            // Currently we only have signed integers
            // so this is pretty simple.
            compile_expr(cx, locals, isns, lhs);
            compile_expr(cx, locals, isns, rhs);
            match op {
                ir::BOp::Add => isns.push(I::Add(t::ValType::I32)),
                ir::BOp::Sub => isns.push(I::Subtract(t::ValType::I32)),
                ir::BOp::Mul => isns.push(I::Multiply(t::ValType::I32)),
                // TODO: Check for div0?
                ir::BOp::Div => isns.push(I::I32Division {
                    ty: i::IntegerType::I32,
                    signed: true,
                }),
                // TODO: Check for div0?
                ir::BOp::Mod => isns.push(I::Remainder {
                    ty: i::IntegerType::I32,
                    signed: true,
                }),
            }
            0
        }
        E::UniOp { op, rhs } => match op {
            // We just implement this as 0 - thing.
            // By definition this only works on signed integers anyway.
            ir::UOp::Neg => {
                isns.push(I::Const(i::Literal::I32(0)));
                compile_expr(cx, locals, isns, rhs);
                isns.push(I::Subtract(t::ValType::I32));
                0
            }
        },
        // This is pretty much just a list of expr's by now.
        // However, functions/etc must not leave extra values
        // on the stack, so this needs to
        E::Block { body } => compile_exprs(cx, locals, isns, body),
        E::Let {
            varname,
            typename,
            init,
        } => {
            // Declare local var storage
            let wasm_type = compile_typesym(cx, *typename);
            let local_idx = locals.len() as u32;
            let l = LocalVar {
                local_idx,
                wasm_type,
            };
            locals.insert(*varname, l);
            // Compile init expression
            compile_expr(cx, locals, isns, init);
            // Store result of expression
            isns.push(I::LocalSet(l.local_idx));
            0
        }
        E::If {
            condition,
            trueblock,
            falseblock,
        } => 0,
        E::Loop { body } => 0,
        E::Lambda { signature, body } => 0,
        E::Funcall { func, params } => 0,
        E::Break => 0,
        E::Return { retval } => 0,
    }
}

fn compile_typesym(cx: &Cx, t: TypeSym) -> t::ValType {
    let tdef = cx.unintern_type(t);
    compile_type(cx, tdef)
}

fn compile_type(_cx: &Cx, t: &TypeDef) -> t::ValType {
    match t {
        TypeDef::SInt(4) => t::ValType::I32,
        TypeDef::SInt(_) => panic!("TODO"),
        TypeDef::Bool => t::ValType::I32,
        TypeDef::Tuple(_) => panic!("Unimplemented"),
        TypeDef::Lambda(_, _) => panic!("Unimplemented"),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use wasm_builder::{instr as i, instr::Instruction as I};

    use crate::backend::*;
    use crate::ir::Expr as E;
    use crate::*;

    /// Test compiling a let expr and var lookup
    #[test]
    fn test_compile_var() {
        let cx = &mut Cx::new();
        let locals = &mut HashMap::new();
        let isns = &mut vec![];

        let varname = cx.intern("foo");

        let expr = E::Let {
            varname: cx.intern("foo"),
            typename: cx.intern_type(&TypeDef::SInt(4)),
            init: Box::new(ir::Expr::int(9)),
        };
        compile_expr(cx, locals, isns, &expr);

        assert_eq!(locals.len(), 1);
        assert_eq!(locals[&varname].local_idx, 0);
        assert_eq!(isns[0], I::Const(i::Literal::I32(9)));
        assert_eq!(isns[1], I::LocalSet(0));

        let expr = E::Var { name: varname };
        compile_expr(cx, locals, isns, &expr);
        assert_eq!(isns[2], I::LocalGet(0));
    }

    #[test]
    fn test_compile_binop() {
        let cx = &mut Cx::new();
        let locals = &mut HashMap::new();
        let isns = &mut vec![];

        let expr = ir::Expr::BinOp {
            op: ir::BOp::Sub,
            lhs: Box::new(ir::Expr::int(9)),
            rhs: Box::new(ir::Expr::int(-3)),
        };

        compile_expr(cx, locals, isns, &expr);
        assert_eq!(isns[0], I::Const(i::Literal::I32(9)));
        assert_eq!(isns[1], I::Const(i::Literal::I32(-3)));
        assert_eq!(isns[2], I::Subtract(t::ValType::I32));
    }
}
