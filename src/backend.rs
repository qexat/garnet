//! Output code.
//!
//! For now, we're going to output WASM.  That should let us get interesting output
//! and maybe bootstrap stuff if we feel like it.

use std::cell::RefCell;
use std::cmp;
use std::collections::HashMap;

use walrus as w;

use crate::ir;
use crate::*;

/// Backend context
struct BCx<'a> {
    cx: &'a Cx,
    m: w::Module,
}

impl<'a> BCx<'a> {
    pub fn new(cx: &'a Cx) -> BCx<'a> {
        let config = w::ModuleConfig::new();
        let m = w::Module::with_config(config);
        Self { cx, m }
    }
}

/// Entry point to turn the IR into a compiled wasm module
pub fn output(cx: &Cx, program: &ir::Ir<TypeSym>) -> Vec<u8> {
    let bcx = &mut BCx::new(cx);
    for decl in program.decls.iter() {
        compile_decl(bcx, decl);
    }
    //m.emit_wasm_file("/home/icefox/test.wasm");
    bcx.m.emit_wasm()
}

fn compile_decl(bcx: &mut BCx, decl: &ir::Decl<TypeSym>) {
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
        }
        Decl::Const {
            /*
            name,
            typename,
            init,
            */
            ..
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
        .map(|(_varsym, typesym)| compile_typesym(&bcx.cx, *typesym))
        .collect();
    let rettype = compile_typesym(&bcx.cx, sig.rettype);
    (params, vec![rettype])
}
#[derive(Copy, Clone, Debug, PartialEq)]
struct LocalVar {
    local_idx: w::ir::LocalId,
    wasm_type: w::ValType,
}

/*
enum SymbolNode<'a> {
    Nil,
    Symbol(VarSym, LocalVar, &'a SymbolNode<'a>),
}

impl<'a> SymbolNode<'a> {
    fn add(s: VarSym, local: LocalVar, prev: &'a SymbolNode) -> SymbolNode<'a> {
        SymbolNode::Symbol(s, local, prev)
    }
}
*/

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
fn compile_function(
    bcx: &mut BCx,
    sig: &ir::Signature,
    body: &[ir::TypedExpr<TypeSym>],
) -> w::FunctionId {
    let (paramtype, rettype) = function_signature(bcx, sig);
    let mod_locals = &mut bcx.m.locals;

    // A simple, scope-less symbol table.
    // TODO Wait what if two vars have the same name
    let locals: &mut HashMap<VarSym, LocalVar> = &mut HashMap::new();
    // add params
    let mut fn_params = vec![];
    for (pname, ptype) in sig.params.iter() {
        let idx = mod_locals.add(compile_typesym(&bcx.cx, *ptype));
        fn_params.push(idx);
        let p = LocalVar {
            local_idx: idx,
            wasm_type: compile_typesym(&bcx.cx, *ptype),
        };
        locals.insert(*pname, p);
    }
    // So we have to create the function now, to get the FunctionId
    // and know what to call in case there's a recursion or such.
    let fb = w::FunctionBuilder::new(&mut bcx.m.types, &paramtype, &rettype);
    let f_idx = fb.finish(fn_params, &mut bcx.m.funcs);
    let defined_func = bcx.m.funcs.get_mut(f_idx);
    match &mut defined_func.kind {
        w::FunctionKind::Local(lfunc) => {
            let instrs = &mut lfunc.builder_mut().func_body();
            let _ = compile_exprs(&bcx.cx, locals, mod_locals, instrs, &body);
        }
        _ => unreachable!("Can't happen!"),
    }
    f_idx
}

/// Compile multiple exprs, making sure they don't leave unused
/// values on the stack by adding drop's as necessary.
/// Returns the number of values left on the stack at the end.
fn compile_exprs(
    cx: &Cx,
    locals: &mut HashMap<VarSym, LocalVar>,
    mod_locals: &mut w::ModuleLocals,
    instrs: &mut w::InstrSeqBuilder,
    exprs: &[ir::TypedExpr<TypeSym>],
) {
    // The trick here is that wasm doesn't let you leave stuff on
    // the stack and just ignore it forevermore.  Like, functions must
    // have only their return value on the stack when they return,
    // stuff like that.  So we need to explicitly get rid of the
    // things we ignore.
    //
    // I considered making this a step in the IR that basically turns
    // every `foo(); bar()` into `ignore(foo()); bar()` explicitly,
    // but it seems easier for now to just drop any values left by
    // any but the last expr.
    //
    // TODO: Revisit this sometime.
    let l = exprs.len();
    match l {
        0 => (),
        1 => compile_expr(cx, locals, mod_locals, instrs, &exprs[0]),
        l => {
            // Compile and insert drop instructions after each
            // expr
            let exprs_prefix = &exprs[..l - 1];
            exprs_prefix.iter().for_each(|expr| {
                compile_expr(cx, locals, mod_locals, instrs, expr);
                let x = stacksize(&cx, expr.t);
                for _ in 0..x {
                    instrs.drop();
                }
            });
            // Then compile the last one.
            compile_expr(cx, locals, mod_locals, instrs, &exprs[l - 1]);
        }
    }
}

/// Generates code to evaluate the given expr, inserting the instructions into
/// the given instruction list, leaving values on the stack.
/// Returns how many values it leaves on the stack, so we know how many
/// values to get rid of if the return val is ignored.
fn compile_expr(
    cx: &Cx,
    locals: &mut HashMap<VarSym, LocalVar>,
    mod_locals: &mut w::ModuleLocals,
    instrs: &mut w::InstrSeqBuilder,
    expr: &ir::TypedExpr<TypeSym>,
) {
    use ir::Expr as E;
    use ir::*;
    match &expr.e {
        E::Lit { val } => match val {
            Literal::Integer(i) => {
                assert!(*i < (std::i32::MAX as i64));
                instrs.i32_const(*i as i32);
            }
            Literal::Bool(b) => {
                instrs.i32_const(if *b { 1 } else { 0 });
            }
            // noop
            Literal::Unit => (),
        },
        E::Var { name } => {
            let ldef = locals
                .get(name)
                .expect(&format!("Unknown local {:?}; should never happen", name));
            instrs.local_get(ldef.local_idx);
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
            compile_expr(cx, locals, mod_locals, instrs, lhs);
            compile_expr(cx, locals, mod_locals, instrs, rhs);
            let op = compile_binop(op);
            instrs.binop(op);
        }
        E::UniOp { op, rhs } => match op {
            // We just implement this as 0 - thing.
            // By definition this only works on signed integers anyway.
            ir::UOp::Neg => {
                instrs.i32_const(0);
                compile_expr(cx, locals, mod_locals, instrs, rhs);
                instrs.binop(w::ir::BinaryOp::I32Sub);
            }
        },
        // This is pretty much just a list of expr's by now.
        // However, functions/etc must not leave extra values
        // on the stack, so this needs to insert drop's as appropriate
        E::Block { body } => {
            compile_exprs(cx, locals, mod_locals, instrs, body);
        }
        E::Let {
            varname,
            typename,
            init,
        } => {
            // Declare local var storage
            let wasm_type = compile_typesym(cx, *typename);
            let local_idx = mod_locals.add(wasm_type);
            let l = LocalVar {
                local_idx,
                wasm_type,
            };
            locals.insert(*varname, l);
            // Compile init expression
            compile_expr(cx, locals, mod_locals, instrs, init);
            // Store result of expression
            instrs.local_set(local_idx);
        }
        E::If { cases, falseblock } => {
            compile_ifcase(cx, locals, mod_locals, cases, instrs, falseblock);
        }

        E::Loop { body } => todo!(),
        E::Lambda { signature, body } => todo!(),
        E::Funcall { func, params } => {
            for param in params {
                compile_expr(cx, locals, mod_locals, instrs, param);
            }
            //compile_expr(
            todo!()
        }
        E::Break => todo!(),
        E::Return { retval } => {
            compile_expr(cx, locals, mod_locals, instrs, retval);
            instrs.return_();
        }
    };
}

/*
// Notes from earlier IR
// Expand cases out into nested if ... else if ... else if ... else
// TODO: Really this should be lowered into a match expr, but we don't
// have those yet, so.
fn unheck_if(ifcases: &[ast::IfCase], elsecase: &[ast::Expr]) -> Expr {
    match ifcases {
        [] => panic!("If statement with no condition; should never happen"),
        [single] => {
            let nelsecase = lower_exprs(elsecase);
            If {
                condition: Box::new(lower_expr(&*single.condition)),
                trueblock: lower_exprs(single.body.as_slice()),
                falseblock: nelsecase,
            }
        }
        [itm, rst @ ..] => {
            let res = unheck_if(rst, elsecase);
            If {
                condition: Box::new(lower_expr(&*itm.condition)),
                trueblock: lower_exprs(itm.body.as_slice()),
                falseblock: vec![res],
            }
        }
    }
}
unheck_if(cases.as_slice(), falseblock.as_slice())
*/

/// hokay, so.  For each case, we have to:
/// compile the test
/// generate the code for the if-part
/// if there's another if-case, recurse.
fn compile_ifcase(
    cx: &Cx,
    locals: &mut HashMap<VarSym, LocalVar>,
    mod_locals: &mut w::ModuleLocals,
    cases: &[(ir::TypedExpr<TypeSym>, Vec<ir::TypedExpr<TypeSym>>)],
    instrs: &mut w::InstrSeqBuilder,
    falseblock: &[ir::TypedExpr<TypeSym>],
) {
    assert_ne!(cases.len(), 0);
    // First off we just compile the if test,
    // regardless of what.
    let (test, thenblock) = &cases[0];
    compile_expr(cx, locals, mod_locals, instrs, &test);
    let rettype = compile_typesym(&cx, test.t);
    let grr = RefCell::new(locals);
    let mgrr = RefCell::new(mod_locals);

    // The redundancy here is kinda screwy, trying to refactor
    // out the closures leads to weird lifetime errors.
    if cases.len() == 1 {
        // We are done, just compile the test...
        instrs.if_else(
            rettype,
            |then| {
                let locals = &mut grr.borrow_mut();
                let mod_locals = &mut mgrr.borrow_mut();
                compile_exprs(cx, locals, mod_locals, then, &thenblock);
            },
            |else_| {
                let locals = &mut grr.borrow_mut();
                let mod_locals = &mut mgrr.borrow_mut();
                compile_exprs(cx, locals, mod_locals, else_, falseblock);
            },
        );
    } else {
        // Compile the first if case, then recurse with the rest of them
        instrs.if_else(
            rettype,
            |then| {
                let locals = &mut grr.borrow_mut();
                let mod_locals = &mut mgrr.borrow_mut();
                compile_exprs(cx, locals, mod_locals, then, &thenblock);
            },
            |else_| {
                let locals = &mut grr.borrow_mut();
                let mod_locals = &mut mgrr.borrow_mut();
                compile_ifcase(cx, locals, mod_locals, &cases[1..], else_, falseblock);
            },
        );
    }
}

fn compile_typesym(cx: &Cx, t: TypeSym) -> w::ValType {
    let tdef = cx.fetch_type(t);
    compile_type(cx, &tdef)
}

fn compile_type(_cx: &Cx, t: &TypeDef) -> w::ValType {
    match t {
        TypeDef::SInt(4) => w::ValType::I32,
        TypeDef::SInt(_) => todo!(),
        TypeDef::Bool => w::ValType::I32,
        TypeDef::Tuple(_) => todo!(),
        TypeDef::Lambda(_, _) => todo!(),
    }
}

/// The number of slots a type will consume when left on the wasm stack.
/// Not the same as words used in memory.
/// unit = 0
/// bool = 1
/// I32 = 1
/// eventually, tuple >= 1
fn stacksize(cx: &Cx, t: TypeSym) -> usize {
    let tdef = cx.fetch_type(t);
    match &*tdef {
        TypeDef::SInt(4) => 1,
        TypeDef::SInt(_) => todo!(),
        TypeDef::Bool => 1,
        TypeDef::Tuple(t) if t.len() == 0 => 0,
        TypeDef::Tuple(_) => todo!(),
        TypeDef::Lambda(_, _) => todo!(),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use walrus as w;

    use crate::backend::*;
    use crate::ir::{self, plz, Expr as E};

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
        let mod_locals = &mut bcx.m.locals;
        let mut fb = w::FunctionBuilder::new(&mut bcx.m.types, &paramtype, &rettype);
        let instrs = &mut fb.func_body();
        let locals = &mut HashMap::new();

        // Can we compile the let?
        let expr = ir::TypedExpr {
            t: unit_t,
            e: E::Let {
                varname: varname,
                typename: i32_t,
                init: Box::new(ir::TypedExpr {
                    t: i32_t,
                    e: ir::Expr::int(9),
                }),
            },
        };
        compile_expr(cx, locals, mod_locals, instrs, &expr);

        assert_eq!(locals.len(), 1);
        assert!(locals.get(&varname).is_some());

        // Can we then compile the var lookup?
        let expr = ir::TypedExpr {
            t: i32_t,
            e: E::Var { name: varname },
        };
        compile_expr(cx, locals, mod_locals, instrs, &expr);
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
