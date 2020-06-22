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

/// Entry point to turn the IR into a compiled wasm module
pub fn output(cx: &Cx, program: &ir::Ir<TypeSym>) -> Vec<u8> {
    let config = w::ModuleConfig::new();
    let m = &mut w::Module::with_config(config);
    let symbols = &mut Symbols::new();
    for decl in program.decls.iter() {
        compile_decl(&cx, m, symbols, decl);
    }
    //m.emit_wasm_file("/home/icefox/test.wasm");
    m.emit_wasm()
}

fn compile_decl(cx: &Cx, m: &mut w::Module, symbols: &mut Symbols, decl: &ir::Decl<TypeSym>) {
    use ir::*;
    match decl {
        Decl::Function {
            name,
            signature,
            body,
        } => {
            compile_function(cx, m, symbols, *name, signature, body);
            let name_str = cx.fetch(*name);
            let function_id = symbols.get_function(*name).unwrap();
            m.exports.add(&name_str, function_id);
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

fn function_signature(cx: &Cx, sig: &ir::Signature) -> (Vec<w::ValType>, Vec<w::ValType>) {
    let params: Vec<w::ValType> = sig
        .params
        .iter()
        .map(|(_varsym, typesym)| compile_typesym(&cx, *typesym))
        .collect();
    let rettype = compile_typesym(&cx, sig.rettype);
    (params, vec![rettype])
}
#[derive(Copy, Clone, Debug, PartialEq)]
struct LocalVar {
    name: VarSym,
    id: w::LocalId,
    wasm_type: w::ValType,
}

impl LocalVar {
    /// Creates a new local var, including wasm ID,
    /// without inserting it into the symbol table.
    fn new(cx: &Cx, locals: &mut w::ModuleLocals, name: VarSym, type_: TypeSym) -> LocalVar {
        let id = locals.add(compile_typesym(cx, type_));
        let p = LocalVar {
            name,
            id,
            wasm_type: compile_typesym(cx, type_),
        };
        p
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Binding {
    Local(LocalVar),

    GlobalVar {
        name: VarSym,
        //local_idx: w::LocalId,
        wasm_type: w::ValType,
    },

    Function {
        name: VarSym,
        id: w::FunctionId,
        params: Vec<LocalVar>,
    },
}

/// A scoped symbol table containing everything what's in the wasm module.
struct Symbols {
    bindings: Vec<HashMap<VarSym, Binding>>,
}

impl Symbols {
    fn new() -> Self {
        Self {
            bindings: vec![HashMap::default()],
        }
    }
    fn push_scope(&mut self) {
        self.bindings.push(HashMap::default());
    }
    fn pop_scope(&mut self) {
        assert!(self.bindings.len() > 1);
        self.bindings.pop();
    }

    /// Add an already-existing local binding to the top of the current scope.
    /// Create a new binding with LocalVar::new()
    fn add_local(&mut self, local: LocalVar) {
        self.bindings
            .last_mut()
            .unwrap()
            .insert(local.name, Binding::Local(local));
    }

    /// Insert a new function ID into the top scope
    fn new_function(&mut self, name: VarSym, id: w::FunctionId, params: &[LocalVar]) {
        self.bindings.last_mut().unwrap().insert(
            name,
            Binding::Function {
                name,
                id,
                params: params.to_owned(),
            },
        );
    }

    fn get_local(&mut self, name: VarSym) -> Option<&LocalVar> {
        for scope in self.bindings.iter().rev() {
            if let Some(Binding::Local(x)) = scope.get(&name) {
                return Some(x);
            }
        }
        None
    }

    fn get_function(&mut self, name: VarSym) -> Option<w::FunctionId> {
        for scope in self.bindings.iter().rev() {
            if let Some(Binding::Function { id, .. }) = scope.get(&name) {
                return Some(*id);
            }
        }
        None
    }
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
        Some(self.id.cmp(&other.id))
    }
}

/// Order by local index
impl cmp::Ord for LocalVar {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.id.cmp(&other.id)
    }
}

impl cmp::Eq for LocalVar {}

/// Compile a function, specifically -- wasm needs to know about its params and such.
fn compile_function(
    cx: &Cx,
    m: &mut w::Module,
    symbols: &mut Symbols,
    name: VarSym,
    sig: &ir::Signature,
    body: &[ir::TypedExpr<TypeSym>],
) -> w::FunctionId {
    let (paramtype, rettype) = function_signature(cx, sig);

    // add params
    // We do some shenanigans with scope here to create the locals and then
    let mut fn_params = vec![];
    for (pname, ptype) in sig.params.iter() {
        fn_params.push(LocalVar::new(cx, &mut m.locals, *pname, *ptype));
    }
    // So we have to create the function now, to get the FunctionId
    // and know what to call in case there's a recursion or such.
    let fb = w::FunctionBuilder::new(&mut m.types, &paramtype, &rettype);
    let fn_param_types: Vec<_> = fn_params.iter().map(|local| local.id).collect();
    let f_id = fb.finish(fn_param_types, &mut m.funcs);

    symbols.new_function(name, f_id, &fn_params);
    let defined_func = m.funcs.get_mut(f_id);
    match &mut defined_func.kind {
        w::FunctionKind::Local(lfunc) => {
            let instrs = &mut lfunc.builder_mut().func_body();
            // Handle scoping and add input params to it.
            symbols.push_scope();
            for local in fn_params.iter() {
                symbols.add_local(*local);
            }
            compile_exprs(&cx, &mut m.locals, symbols, instrs, &body);
            symbols.pop_scope();
        }
        _ => unreachable!("Can't happen!"),
    }
    f_id
}

/// Compile multiple exprs, making sure they don't leave unused
/// values on the stack by adding drop's as necessary.
/// Returns the number of values left on the stack at the end.
fn compile_exprs(
    cx: &Cx,
    m: &mut w::ModuleLocals,
    symbols: &mut Symbols,
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
        1 => compile_expr(cx, m, symbols, instrs, &exprs[0]),
        l => {
            // Compile and insert drop instructions after each
            // expr
            let exprs_prefix = &exprs[..l - 1];
            exprs_prefix.iter().for_each(|expr| {
                compile_expr(cx, m, symbols, instrs, expr);
                let x = stacksize(&cx, expr.t);
                for _ in 0..x {
                    instrs.drop();
                }
            });
            // Then compile the last one.
            compile_expr(cx, m, symbols, instrs, &exprs[l - 1]);
        }
    }
}

/// Generates code to evaluate the given expr, inserting the instructions into
/// the given instruction list, leaving values on the stack.
/// Returns how many values it leaves on the stack, so we know how many
/// values to get rid of if the return val is ignored.
fn compile_expr(
    cx: &Cx,
    m: &mut w::ModuleLocals,
    symbols: &mut Symbols,
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
            let ldef = symbols
                .get_local(*name)
                .expect(&format!("Unknown local {:?}; should never happen", name));
            instrs.local_get(ldef.id);
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

                    ir::BOp::Eq => w::ir::BinaryOp::I32Eq,
                    ir::BOp::Neq => w::ir::BinaryOp::I32Ne,
                    ir::BOp::Gt => w::ir::BinaryOp::I32GtS,
                    ir::BOp::Lt => w::ir::BinaryOp::I32LtS,
                    ir::BOp::Gte => w::ir::BinaryOp::I32GeS,
                    ir::BOp::Lte => w::ir::BinaryOp::I32LeS,

                    ir::BOp::And => w::ir::BinaryOp::I32And,
                    ir::BOp::Or => w::ir::BinaryOp::I32Or,
                    ir::BOp::Xor => w::ir::BinaryOp::I32Xor,
                }
            }
            compile_expr(cx, m, symbols, instrs, lhs);
            compile_expr(cx, m, symbols, instrs, rhs);
            let op = compile_binop(op);
            instrs.binop(op);
        }
        E::UniOp { op, rhs } => match op {
            // We just implement this as 0 - thing.
            // By definition this only works on signed integers anyway.
            ir::UOp::Neg => {
                instrs.i32_const(0);
                compile_expr(cx, m, symbols, instrs, rhs);
                instrs.binop(w::ir::BinaryOp::I32Sub);
            }
            ir::UOp::Not => {
                compile_expr(cx, m, symbols, instrs, rhs);
                instrs.unop(w::ir::UnaryOp::I32Eqz);
            }
        },
        // This is pretty much just a list of expr's by now.
        // However, functions/etc must not leave extra values
        // on the stack, so this needs to insert drop's as appropriate
        E::Block { body } => {
            compile_exprs(cx, m, symbols, instrs, body);
        }
        E::Let {
            varname,
            typename,
            init,
        } => {
            // Declare local var storage
            let local = LocalVar::new(cx, m, *varname, *typename);
            symbols.add_local(local);
            // Compile init expression
            compile_expr(cx, m, symbols, instrs, init);
            // Store result of expression
            instrs.local_set(local.id);
        }
        E::If { cases, falseblock } => {
            compile_ifcase(cx, m, symbols, cases, instrs, falseblock);
        }

        E::Loop { body } => todo!(),
        E::Lambda { .. } => {
            panic!("A lambda somewhere has not been lifted into a separate function.  Should never happen!");
        }
        E::Funcall { func, params } => {
            // OKAY.  SO.  Wasm does NOT do function calls through pointers on the stack.
            // The call is built into the instruction.  So, we can NOT do the functional-language
            // thing and trivially treat function pointers like any other type.
            // There IS the call_indirect instruction but it's kinda awful and shouldn't
            // be necessary currently, so.
            match **func {
                ir::TypedExpr {
                    e: E::Var { name },
                    t: _,
                } => {
                    for param in params {
                        compile_expr(cx, m, symbols, instrs, param);
                    }
                    let func = symbols.get_function(name).unwrap();
                    instrs.call(func);
                }
                _ => panic!("A funcall got something other than a var, which means a lambda somewhere hasn't been lowered"),
            }
        }
        E::Break => todo!(),
        E::Return { retval } => {
            compile_expr(cx, m, symbols, instrs, retval);
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
    m: &mut w::ModuleLocals,
    symbols: &mut Symbols,
    cases: &[(ir::TypedExpr<TypeSym>, Vec<ir::TypedExpr<TypeSym>>)],
    instrs: &mut w::InstrSeqBuilder,
    falseblock: &[ir::TypedExpr<TypeSym>],
) {
    assert_ne!(cases.len(), 0);
    // First off we just compile the if test,
    // regardless of what.
    let (test, thenblock) = &cases[0];
    compile_expr(cx, m, symbols, instrs, &test);
    let rettype = compile_typesym(&cx, test.t);
    let grr = RefCell::new(symbols);
    let mgrr = RefCell::new(m);

    // The redundancy here is kinda screwy, trying to refactor
    // out the closures leads to weird lifetime errors.
    if cases.len() == 1 {
        // We are done, just compile the test...
        instrs.if_else(
            rettype,
            |then| {
                let symbols = &mut grr.borrow_mut();
                let m = &mut mgrr.borrow_mut();
                compile_exprs(cx, m, symbols, then, &thenblock);
            },
            |else_| {
                let symbols = &mut grr.borrow_mut();
                let m = &mut mgrr.borrow_mut();
                compile_exprs(cx, m, symbols, else_, falseblock);
            },
        );
    } else {
        // Compile the first if case, then recurse with the rest of them
        instrs.if_else(
            rettype,
            |then| {
                let symbols = &mut grr.borrow_mut();
                let m = &mut mgrr.borrow_mut();
                compile_exprs(cx, m, symbols, then, &thenblock);
            },
            |else_| {
                let symbols = &mut grr.borrow_mut();
                let m = &mut mgrr.borrow_mut();
                compile_ifcase(cx, m, symbols, &cases[1..], else_, falseblock);
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
        // Essentially a pointer to a function
        TypeDef::Lambda(_, _) => w::ValType::I32,
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
        // Essentially a pointer to a function
        TypeDef::Lambda(_, _) => 1,
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
        let config = w::ModuleConfig::new();
        let mut m = w::Module::with_config(config);

        let varname = cx.intern("foo");

        let (paramtype, rettype) = function_signature(
            cx,
            &ir::Signature {
                params: vec![],
                rettype: cx.i32(),
            },
        );
        let symbols = &mut Symbols::new();
        let mut fb = w::FunctionBuilder::new(&mut m.types, &paramtype, &rettype);
        let instrs = &mut fb.func_body();

        // Can we compile the let?
        let expr = ir::TypedExpr {
            t: cx.unit(),
            e: E::Let {
                varname,
                typename: cx.i32(),
                init: Box::new(ir::TypedExpr {
                    t: cx.i32(),
                    e: ir::Expr::int(9),
                }),
            },
        };
        compile_expr(cx, &mut m.locals, symbols, instrs, &expr);

        assert_eq!(symbols.bindings.last().unwrap().len(), 1);
        assert!(symbols.get_local(varname).is_some());

        // Can we then compile the var lookup?
        let expr = ir::TypedExpr {
            t: cx.i32(),
            e: E::Var { name: varname },
        };
        compile_expr(cx, &mut m.locals, symbols, instrs, &expr);
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
