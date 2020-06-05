//! Output code.
//!
//! For now, we're going to output WASM.  That should let us get interesting output
//! and maybe bootstrap stuff if we feel like it.

use std::cmp;
use std::collections::HashMap;

use walrus as w;
use wasm_builder::{instr as i, module as m, sections as s, types as t};

use crate::ir;
use crate::*;

/// Backend context
struct BCx<'a> {
    cx: &'a mut Cx,
    m: w::Module,
}

impl<'a> BCx<'a> {
    pub fn new(cx: &'a mut Cx) -> BCx<'a> {
        let config = w::ModuleConfig::new();
        let m = w::Module::with_config(config);
        Self { cx, m }
    }
}

/// Entry point to turn the IR into a compiled wasm module
pub fn output(cx: &mut Cx, program: &ir::Ir) -> Vec<u8> {
    let bcx = &mut BCx::new(cx);
    for decl in program.decls.iter() {
        compile_decl(bcx, decl);
    }
    bcx.m.emit_wasm()
}

fn compile_decl(bcx: &mut BCx, decl: &ir::Decl) {
    use ir::*;
    match decl {
        Decl::Function {
            name,
            signature,
            body,
        } => {
            let function_id = compile_function(bcx, signature, body);
            let name = bcx.cx.fetch(*name);
            bcx.m.exports.add(&name, function_id);
            /*
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
                name: cx.fetch(*name).to_owned(),
                desc: s::Desc::Function(funcsec_idx as u32),
            });
            */
        }
        Decl::Const {
            name,
            typename,
            init,
        } => {
            // Okay this is actually harder than it looks because
            // wasm only lets globals be value types.
            // ...and if it's really const then why do we need it anyway
            //let wasm_type = compile_typesym(cx, *typename);
        }
    }
}

fn function_signature(bcx: &mut BCx, sig: &ir::Signature) -> (Vec<w::ValType>, Vec<w::ValType>) {
    let params: Vec<w::ValType> = sig
        .params
        .iter()
        .map(|(_varsym, typesym)| compile_typesym(bcx, *typesym))
        .collect();
    let rettype = compile_typesym(bcx, sig.rettype);
    (params, vec![rettype])
}
#[derive(Copy, Clone, Debug, PartialEq)]
struct LocalVar {
    local_idx: w::ir::LocalId,
    wasm_type: w::ValType,
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
fn compile_function(bcx: &mut BCx, sig: &ir::Signature, body: &[ir::Expr]) -> w::FunctionId {
    let (paramtype, rettype) = function_signature(bcx, sig);
    let mut fb = w::FunctionBuilder::new(&mut bcx.m.types, &paramtype, &rettype);
    // A simple, scope-less symbol table.
    let mut locals: HashMap<VarSym, LocalVar> = HashMap::new();
    // add params
    for (pname, ptype) in sig.params.iter() {
        let idx = bcx.m.locals.add(compile_typesym(bcx, *ptype));
        let p = LocalVar {
            local_idx: idx,
            wasm_type: compile_typesym(bcx, *ptype),
        };
        locals.insert(*pname, p);
    }
    // add locals
    //bcx.m.locals.add();
    let instrs = &mut fb.func_body();
    let _ = compile_exprs(bcx, &mut locals, instrs, &body);
    fb.finish(vec![], &mut bcx.m.funcs)

    /*
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
                */
}

/// Compile multiple exprs, making sure they don't leave unused
/// values on the stack by adding drop's as necessary.
/// Returns the number of values left on the stack at the end.
fn compile_exprs(
    bcx: &mut BCx,
    locals: &mut HashMap<VarSym, LocalVar>,
    instrs: &mut w::InstrSeqBuilder,
    exprs: &[ir::Expr],
) -> usize {
    // TODO Implement
    // I considered making this a step in the IR that basically turns
    // every `foo(); bar()` into `ignore(foo()); bar()` explicitly,
    // but how hard to ignore it is dependent on the return type and
    // so doing it here seems easier.  IR doesn't explicitly store
    // the return type of each expression... though maybe it should.
    //
    // Easier to do is just drop all but the results of the last expr.
    let l = exprs.len();
    match l {
        0 => 0,
        1 => compile_expr(bcx, locals, instrs, &exprs[0]),
        l => {
            let exprs_prefix = &exprs[..l - 1];
            exprs_prefix.iter().for_each(|expr| {
                let x = compile_expr(bcx, locals, instrs, expr);
                for _ in 0..x {
                    instrs.drop();
                }
            });
            compile_expr(bcx, locals, instrs, &exprs[l - 1])
        }
    }
}

/// Generates code to evaluate the given expr, inserting the instructions into
/// the given instruction list, leaving values on the stack.
/// Returns how many values it leaves on the stack, so we know how many
/// values to get rid of if the return val is ignored.
fn compile_expr(
    bcx: &mut BCx,
    locals: &mut HashMap<VarSym, LocalVar>,
    instrs: &mut w::InstrSeqBuilder,
    expr: &ir::Expr,
) -> usize {
    use i::Instruction as I;
    use ir::Expr as E;
    use ir::*;
    match expr {
        E::Lit { val } => match val {
            Literal::Integer(i) => {
                assert!(*i < (i32::MAX as i64));
                instrs.i32_const(*i as i32);
                1
            }
            Literal::Bool(b) => {
                instrs.i32_const(if *b { 1 } else { 0 });
                1
            }
            Literal::Unit => 0,
        },
        E::Var { name } => {
            let ldef = locals
                .get(name)
                .expect(&format!("Unknown local {:?}; should never happen", name));
            instrs.local_get(ldef.local_idx);
            0
        }
        E::BinOp { op, lhs, rhs } => {
            // Currently we only have signed integers
            // so this is pretty simple.
            fn compile_binop(op: &ir::BOp) -> w::ir::BinaryOp {
                match op {
                    ir::BOp::Add => w::ir::BinaryOp::I32Add,
                    ir::BOp::Sub => w::ir::BinaryOp::I32Sub,
                    ir::BOp::Mul => w::ir::BinaryOp::I32Mul,

                    // TODO: Check for div0?
                    ir::BOp::Div => w::ir::BinaryOp::I32DivS,
                    // TODO: Check for div0?
                    ir::BOp::Mod => w::ir::BinaryOp::I32RemS,
                }
            }
            assert_eq!(compile_expr(bcx, locals, instrs, lhs), 1);
            assert_eq!(compile_expr(bcx, locals, instrs, rhs), 1);
            let op = compile_binop(op);
            instrs.binop(op);
            1
        }
        E::UniOp { op, rhs } => match op {
            // We just implement this as 0 - thing.
            // By definition this only works on signed integers anyway.
            ir::UOp::Neg => {
                instrs.i32_const(0);
                assert_eq!(compile_expr(bcx, locals, instrs, rhs), 1);
                instrs.binop(w::ir::BinaryOp::I32Sub);
                1
            }
        },
        // This is pretty much just a list of expr's by now.
        // However, functions/etc must not leave extra values
        // on the stack, so this needs to insert drop's as appropriate
        E::Block { body } => compile_exprs(bcx, locals, instrs, body),
        E::Let {
            varname,
            typename,
            init,
        } => {
            // Declare local var storage
            let wasm_type = compile_typesym(bcx, *typename);
            let local_idx = bcx.m.locals.add(wasm_type);
            let l = LocalVar {
                local_idx,
                wasm_type,
            };
            locals.insert(*varname, l);
            // Compile init expression
            compile_expr(bcx, locals, instrs, init);
            // Store result of expression
            instrs.local_set(local_idx);

            0
        }
        E::If {
            condition,
            trueblock,
            falseblock,
        } => {
            // TODO: We need to know what type the blocks return.  Hmm.
            assert_eq!(compile_expr(bcx, locals, instrs, condition), 1);
            let mut true_count = 0;
            let mut false_count = 0;
            /* TODO
             * sort out this double-borrow
             * Basically just needs immutable bcx and properly stacked locals.
            instrs.if_else(
                None,
                |then| {
                    true_count = compile_exprs(bcx, locals, then, trueblock);
                },
                |else_| {
                    false_count = compile_exprs(bcx, locals, else_, falseblock);
                },
            );
            */
            assert_eq!(true_count, false_count);
            true_count
        }

        E::Loop { body } => 0,
        E::Lambda { signature, body } => 0,
        E::Funcall { func, params } => {
            //assert_eq!(compile_expr(bcx, locals, instrs, func), 1);
            0
        }
        E::Break => 0,
        E::Return { retval } => 0,
    }
}

fn compile_typesym(bcx: &BCx, t: TypeSym) -> w::ValType {
    let tdef = bcx.cx.fetch_type(t);
    compile_type(bcx.cx, &tdef)
}

fn compile_type(_cx: &Cx, t: &TypeDef) -> w::ValType {
    match t {
        TypeDef::SInt(4) => w::ValType::I32,
        TypeDef::SInt(_) => panic!("TODO"),
        TypeDef::Bool => w::ValType::I32,
        TypeDef::Tuple(_) => panic!("Unimplemented"),
        TypeDef::Lambda(_, _) => panic!("Unimplemented"),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use walrus as w;

    use crate::backend::*;
    use crate::ir::Expr as E;
    use crate::*;

    /// Test compiling a let expr and var lookup
    #[test]
    fn test_compile_var() {
        let cx = &mut Cx::new();

        let varname = cx.intern("foo");
        let unit_t = cx.intern_type(&TypeDef::Tuple(vec![]));
        let i32_t = cx.intern_type(&TypeDef::SInt(4));

        let bcx = &mut BCx::new(cx);
        let (paramtype, rettype) = function_signature(
            bcx,
            &ir::Signature {
                params: vec![],
                rettype: i32_t,
            },
        );
        let mut fb = w::FunctionBuilder::new(&mut bcx.m.types, &paramtype, &rettype);
        let instrs = &mut fb.func_body();
        let locals = &mut HashMap::new();

        let expr = E::Let {
            varname: varname,
            typename: i32_t,
            init: Box::new(ir::Expr::int(9)),
        };
        compile_expr(bcx, locals, instrs, &expr);

        assert_eq!(locals.len(), 1);
        assert!(locals.get(&varname).is_some());
        /*
        assert_eq!(
            instrs.instrs()[0].0,
            w::ir::Instr::Const(w::ir::Const {
                value: w::ir::Value::I32(9)
            })
        );
        */
        /*
        assert_eq!(isns[1], I::LocalSet(0));

        let expr = E::Var { name: varname };
        compile_expr(cx, locals, isns, &expr);
        assert_eq!(isns[2], I::LocalGet(0));
        */
    }

    #[test]
    fn test_compile_binop() {
        /*
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
        */
    }
}
