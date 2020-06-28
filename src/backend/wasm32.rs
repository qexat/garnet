//! Output code.
//!
//! For now, we're going to output WASM.  That should let us get interesting output
//! and maybe bootstrap stuff if we feel like it.

use std::cell::RefCell;
use std::collections::HashMap;

use walrus as w;

use crate::ir;
use crate::*;

/// Entry point to turn the IR into a compiled wasm module
pub(super) fn output(cx: &Cx, program: &ir::Ir<TypeSym>) -> Vec<u8> {
    let config = w::ModuleConfig::new();
    let m = &mut w::Module::with_config(config);
    let symbols = &mut Symbols::new();
    for decl in program.decls.iter() {
        predeclare_decl(&cx, m, symbols, decl);
    }
    make_heckin_function_table(m, symbols);
    for decl in program.decls.iter() {
        compile_decl(&cx, m, symbols, decl);
    }
    m.emit_wasm()
}

/// Ok, so.  To use indirect function calls, ie calling through a pointer,
/// we need to declare a table in our module.  Then we need to make an
/// element section full of our function id's, to init that table with.
fn make_heckin_function_table(m: &mut w::Module, symbols: &mut Symbols) {
    let mut table_members = vec![];
    for scope in &mut symbols.bindings {
        for binding in scope.values_mut() {
            if let Binding::Function(f) = binding {
                // Set the function's table index to its position in the table.
                f.table_offset = table_members.len();
                table_members.push(Some(f.id));
            }
        }
    }

    let table_size = table_members.len() as u32;
    let table = m
        .tables
        .add_local(table_size, Some(table_size), w::ValType::Funcref);
    symbols.function_table = Some(table);
    // Add elements segment...
    // What the heck is ElementKind::Active?  I don't see this shit anywhere in the spec.
    // TODO: Rummage through github design issues, see if we're doing it right.
    // This seems to execute fine, at least.
    m.elements.add(
        w::ElementKind::Active {
            table,
            offset: w::InitExpr::Value(w::ir::Value::I32(0).into()),
        },
        w::ValType::Funcref,
        table_members,
    );
}

/// Goes through top-level decl's and adds them all to the top scope of the symbol table,
/// so we don't need to do forward declarations in our source.
fn predeclare_decl(cx: &Cx, m: &mut w::Module, symbols: &mut Symbols, decl: &ir::Decl<TypeSym>) {
    use ir::*;
    match decl {
        Decl::Function {
            name,
            signature,
            ..
        } => {
            let (paramtype, rettype) = function_signature(cx, signature);
            // add params
            // We do some shenanigans with scope here to create the locals and then
            let mut fn_params = vec![];
            for (pname, ptype) in signature.params.iter() {
                fn_params.push(LocalVar::new(cx, &mut m.locals, *pname, *ptype));
            }
            // So we have to create the function now, to get the FunctionId
            // and know what to call in case there's a recursion or such.
            let fb = w::FunctionBuilder::new(&mut m.types, &paramtype, &rettype);
            let fn_param_types: Vec<_> = fn_params.iter().map(|local| local.id).collect();
            let f_id = fb.finish(fn_param_types, &mut m.funcs);

            symbols.new_function(*name, f_id, &fn_params);
            let name_str = cx.fetch(*name);
            let function_def = symbols.get_function(*name).unwrap();
            m.exports.add(&name_str, function_def.id);
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

fn compile_decl(cx: &Cx, m: &mut w::Module, symbols: &mut Symbols, decl: &ir::Decl<TypeSym>) {
    use ir::*;
    match decl {
        Decl::Function {
            name,
            body,
            ..
        } => {
            compile_function(cx, m, symbols, *name, body);
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

/// Generate walrus types representing the wasm representations of
/// a lambda's type.
fn lambda_signature(
    cx: &Cx,
    params: &[TypeSym],
    ret: TypeSym,
) -> (Vec<w::ValType>, Vec<w::ValType>) {
    let params: Vec<w::ValType> = params
        .iter()
        .map(|typesym| compile_typesym(&cx, *typesym))
        .flatten()
        .collect();
    let rettype = compile_typesym(&cx, ret);
    (params, rettype)
}

/// Same as `lambda_signature` but takes a `Signature`, which is part of
/// an expression, not a type.  Just calls `lambda_signature()` under the hood.
fn function_signature(cx: &Cx, sig: &ir::Signature) -> (Vec<w::ValType>, Vec<w::ValType>) {
    if let TypeDef::Lambda(params, ret) = &*cx.fetch_type(sig.to_type(cx)) {
        lambda_signature(cx, params, *ret)
    } else {
        unreachable!("Should never happen");
    }
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
        // TODO: Locals consisting of more than one value...
        let id = locals.add(compile_typesym(cx, type_)[0]);
        let p = LocalVar {
            name,
            id,
            wasm_type: compile_typesym(cx, type_)[0],
        };
        p
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Function {
    name: VarSym,
    id: w::FunctionId,
    params: Vec<LocalVar>,
    /// The location of the function in the function table
    /// We use a dummy table offset of 0 when we don't know,
    /// but that's also a valid offset, so we're just gonna
    /// suck it for now.
    table_offset: usize,
}

#[derive(Clone, Debug, PartialEq)]
enum Binding {
    Local(LocalVar),

    GlobalVar {
        name: VarSym,
        //local_idx: w::LocalId,
        wasm_type: w::ValType,
    },

    Function(Function),
}

/// A scoped symbol table containing everything what's in the wasm module.
struct Symbols {
    bindings: Vec<HashMap<VarSym, Binding>>,
    function_table: Option<w::TableId>,
}

impl Symbols {
    fn new() -> Self {
        Self {
            bindings: vec![HashMap::default()],
            function_table: None,
        }
    }
    /// TODO: See notes on IR symbol table scopes too.
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
    fn new_function(
        &mut self,
        name: VarSym,
        id: w::FunctionId,
        //type_id: w::TypeId,
        params: &[LocalVar],
    ) {
        self.bindings.last_mut().unwrap().insert(
            name,
            Binding::Function(Function {
                name,
                id,
                //type_id,
                params: params.to_owned(),
                table_offset: 0,
            }),
        );
    }

    /// Get a reference to a local var, if it exists.
    fn get_local(&mut self, name: VarSym) -> Option<&LocalVar> {
        if let Some(Binding::Local(f)) = self.get(name) {
            return Some(f);
        } else {
            None
        }
    }

    /// Get a reference to a function, if it exists
    fn get_function(&mut self, name: VarSym) -> Option<&Function> {
        if let Some(Binding::Function(f)) = self.get(name) {
            return Some(f);
        } else {
            None
        }
    }

    /// Get a reference to an arbitrary binding.
    fn get(&self, name: VarSym) -> Option<&Binding> {
        for scope in self.bindings.iter().rev() {
            if let Some(v) = scope.get(&name) {
                return Some(v);
            }
        }
        None
    }
}

/// Compile a function, specifically -- wasm needs to know about its params and such.
/// The function should already be predeclared.
fn compile_function(
    cx: &Cx,
    m: &mut w::Module,
    symbols: &mut Symbols,
    name: VarSym,
    body: &[ir::TypedExpr<TypeSym>],
) -> w::FunctionId {
    // This should already be here due to the function being predeclared.
    let func = symbols
        .get_function(name)
        .expect("Function was not predeclared, should never happen")
        .clone();
    let defined_func = m.funcs.get_mut(func.id);
    match &mut defined_func.kind {
        w::FunctionKind::Local(lfunc) => {
            let instrs = &mut lfunc.builder_mut().func_body();
            // Handle scoping and add input params to it.
            symbols.push_scope();
            for local in func.params.iter() {
                symbols.add_local(*local);
            }
            compile_exprs(&cx, &mut m.locals, &mut m.types, symbols, instrs, &body);
            symbols.pop_scope();
        }
        _ => unreachable!("Function source is not local, can't happen!"),
    }
    func.id
}

/// Compile multiple exprs, making sure they don't leave unused
/// values on the stack by adding drop's as necessary.
/// Returns the number of values left on the stack at the end.
fn compile_exprs(
    cx: &Cx,
    m: &mut w::ModuleLocals,
    t: &mut w::ModuleTypes,
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
        1 => compile_expr(cx, m, t, symbols, instrs, &exprs[0]),
        l => {
            // Compile and insert drop instructions after each
            // expr
            let exprs_prefix = &exprs[..l - 1];
            exprs_prefix.iter().for_each(|expr| {
                compile_expr(cx, m, t, symbols, instrs, expr);
                let x = stacksize(&cx, expr.t);
                for _ in 0..x {
                    instrs.drop();
                }
            });
            // Then compile the last one.
            compile_expr(cx, m, t, symbols, instrs, &exprs[l - 1]);
        }
    }
}

/// Generates code to evaluate the given expr, inserting the instructions into
/// the given instruction list, leaving values on the stack.
///
/// How many values are left on the stack is determined by the return type
/// attached to the input `TypedExpr`.
fn compile_expr(
    cx: &Cx,
    m: &mut w::ModuleLocals,
    t: &mut w::ModuleTypes,
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
            // Turn bools into integers.
            Literal::Bool(b) => {
                instrs.i32_const(if *b { 1 } else { 0 });
            }
        },
        E::Var { name } => {
            match symbols
                .get(*name)
                .expect(&format!("Unknown var {:?}; should never happen", name))
            {
                Binding::Local(ldef) => {
                    instrs.local_get(ldef.id);
                }
                Binding::Function(fdef) => {
                    instrs.i32_const(fdef.table_offset as i32);
                }
                _ => todo!("Globals"),
            }
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
            compile_expr(cx, m, t, symbols, instrs, lhs);
            compile_expr(cx, m, t, symbols, instrs, rhs);
            let op = compile_binop(op);
            instrs.binop(op);
        }
        E::UniOp { op, rhs } => match op {
            // We just implement this as 0 - thing.
            // By definition this only works on signed integers anyway.
            ir::UOp::Neg => {
                instrs.i32_const(0);
                compile_expr(cx, m, t, symbols, instrs, rhs);
                instrs.binop(w::ir::BinaryOp::I32Sub);
            }
            ir::UOp::Not => {
                compile_expr(cx, m, t, symbols, instrs, rhs);
                instrs.unop(w::ir::UnaryOp::I32Eqz);
            }
        },
        // This is pretty much just a list of expr's by now.
        // However, functions/etc must not leave extra values
        // on the stack, so this needs to use compile_exprs()
        // to insert drop's as appropriate
        E::Block { body } => {
            compile_exprs(cx, m, t, symbols, instrs, body);
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
            compile_expr(cx, m, t, symbols, instrs, init);
            // Store result of expression
            instrs.local_set(local.id);
        }
        E::If { cases, falseblock } => {
            compile_ifcase(cx, m, t, symbols, cases, instrs, falseblock);
        }

        E::Loop { .. } => todo!(),
        E::Lambda { .. } => {
            unreachable!("A lambda somewhere has not been lifted into a separate function.  Should never happen!");
        }
        E::Funcall { func, params } => {
            // OKAY.  SO.  Wasm does NOT do normal function calls through pointers on the stack.
            // The call is built into the instruction.
            //
            // So, to have function pointers we MUST use the `call_indirect` instruction,
            // which takes an index to a table entry on the stack.  So
            //
            // First, evaluate the args.
            for param in params {
                compile_expr(cx, m, t, symbols, instrs, param);
            }
            match &func.e {
                E::Var { name } => {
                    match symbols.get(*name) {
                        // If our var name goes directly to a function binding, it's easy, we can
                        // do a direct call. and we're done.
                        Some(Binding::Function(f)) => {
                            instrs.call(f.id);
                            return;
                        }
                        // Otherwise, it goes to a variable, so we have to look up the value for
                        // that and fetch it, which gives us a table index, then do an indirect
                        // call to that.
                        //
                        // This variable should always exist 'cause we've done our lambda lifting,
                        // so basically all lambda expressions have already been evaluated.
                        Some(Binding::Local(l)) => {
                            instrs.local_get(l.id);
                        }
                        _ => unreachable!(
                            "Backend could not resolve declared function {}!",
                            cx.fetch(*name)
                        ),
                    }
                }
                _expr => {
                    // We got some other expr, compile it and do an indirect call 'cause it heckin'
                    // better be a function
                    compile_expr(cx, m, t, symbols, instrs, func);
                }
            }
            // If we've gotten here we're doing an indirect call, and we've already emitted
            // whatever instructions leave the function table id on the stack.  So, we just
            // look up the function type, and call it.
            let expr_type = &*cx.fetch_type(func.t);
            let (params_types, return_types) = match expr_type {
                TypeDef::Lambda(params, ret) => lambda_signature(cx, params, *ret),
                _ => unreachable!("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
            };
            let function_type = t
                .find(&params_types, &return_types)
                .expect("Shouldn't happen");
            let table_id = symbols.function_table.expect("Can't happen");
            instrs.call_indirect(function_type, table_id);
        }
        E::Break => todo!(),
        E::Return { retval } => {
            compile_expr(cx, m, t, symbols, instrs, retval);
            instrs.return_();
        }
        E::TupleCtor { body } => {
            compile_exprs(cx, m, t, symbols, instrs, body);
        }
        E::TupleRef {
            expr: tuple_expr,
            elt,
        } => {
            // This is a pretty lame way to do it, but an optimization
            // pass to make it better is something that should happen before.

            // Compile init expression
            compile_expr(cx, m, t, symbols, instrs, &*tuple_expr);
            let tuple_type = match &*cx.fetch_type(tuple_expr.t) {
                TypeDef::Tuple(x) => x.clone(),
                _ => unreachable!(),
            };
            let tuple_len = tuple_type.len();
            // Drop however many values are above the element we want,
            // This is a little ghetto to avoid underflows.
            if *elt > 0 {
                for _ in 0..(elt - 1) {
                    instrs.drop();
                }
            }

            // copy the element into a new local,
            let varname = cx.gensym("tupleref");
            let local = LocalVar::new(cx, m, varname, expr.t);
            symbols.add_local(local);
            instrs.local_set(local.id);

            // drop all the things under it,
            for _ in (elt + 1)..tuple_len {
                instrs.drop();
            }
            // and fetch the local.
            instrs.local_get(local.id);
        }
    };
}

/// hokay, so.  For each case, we have to:
/// compile the test
/// generate the code for the if-part
/// if there's another if-case, recurse.
fn compile_ifcase(
    cx: &Cx,
    m: &mut w::ModuleLocals,
    t: &mut w::ModuleTypes,
    symbols: &mut Symbols,
    cases: &[(ir::TypedExpr<TypeSym>, Vec<ir::TypedExpr<TypeSym>>)],
    instrs: &mut w::InstrSeqBuilder,
    falseblock: &[ir::TypedExpr<TypeSym>],
) {
    assert_ne!(cases.len(), 0);
    // First off we just compile the if test,
    // regardless of what.
    let (test, thenblock) = &cases[0];
    compile_expr(cx, m, t, symbols, instrs, &test);
    let rettype = compile_typesym(&cx, test.t);
    let rettype = get_instrseqtype(t, &rettype);

    let grr = RefCell::new(symbols);
    let mgrr = RefCell::new(m);
    let tgrr = RefCell::new(t);

    // The redundancy here is kinda screwy, trying to refactor
    // out the closures leads to weird lifetime errors.
    if cases.len() == 1 {
        // We are done, just compile the test...
        instrs.if_else(
            rettype,
            |then| {
                let symbols = &mut grr.borrow_mut();
                let m = &mut mgrr.borrow_mut();
                let t = &mut tgrr.borrow_mut();
                compile_exprs(cx, m, t, symbols, then, &thenblock);
            },
            |else_| {
                let symbols = &mut grr.borrow_mut();
                let m = &mut mgrr.borrow_mut();
                let t = &mut tgrr.borrow_mut();
                compile_exprs(cx, m, t, symbols, else_, falseblock);
            },
        );
    } else {
        // Compile the first if case, then recurse with the rest of them
        instrs.if_else(
            rettype,
            |then| {
                let symbols = &mut grr.borrow_mut();
                let m = &mut mgrr.borrow_mut();
                let t = &mut tgrr.borrow_mut();
                compile_exprs(cx, m, t, symbols, then, &thenblock);
            },
            |else_| {
                let symbols = &mut grr.borrow_mut();
                let m = &mut mgrr.borrow_mut();
                let t = &mut tgrr.borrow_mut();
                compile_ifcase(cx, m, t, symbols, &cases[1..], else_, falseblock);
            },
        );
    }
}

/// Takes a `TypeSym`, looks it up, and gives us the
/// corresponding wasm type.
fn compile_typesym(cx: &Cx, t: TypeSym) -> Vec<w::ValType> {
    let tdef = cx.fetch_type(t);
    compile_type(cx, &tdef)
}

/// Takes a `TypeDef` and turns it into the appropriate
/// wasm type.
fn compile_type(cx: &Cx, t: &TypeDef) -> Vec<w::ValType> {
    match t {
        TypeDef::SInt(4) => vec![w::ValType::I32],
        TypeDef::SInt(_) => todo!(),
        TypeDef::Bool => vec![w::ValType::I32],
        TypeDef::Tuple(ts) => ts
            .iter()
            .map(|t| compile_typesym(cx, *t))
            .flatten()
            .collect(),
        // Essentially a pointer to a function
        TypeDef::Lambda(_, _) => vec![w::ValType::I32],
    }
}

/// Get/create an instrseqtype, which may be a single valtype or a a typeid
/// representing multiple.
fn get_instrseqtype(t: &mut w::ModuleTypes, def: &[w::ValType]) -> w::ir::InstrSeqType {
    if def.len() > 2 {
        w::ir::InstrSeqType::Simple(def.into_iter().cloned().next())
    } else {
        let typeid = if let Some(t) = t.find(&[], def) {
            t
        } else {
            t.add(&[], def)
        };
        w::ir::InstrSeqType::MultiValue(typeid)
    }
}

/// The number of slots a type will consume when left on the wasm stack.
/// Not the same as words used in memory.
/// unit = 0
/// bool = 1
/// I32 = 1
/// lambda = 1 (table index)
/// eventually, tuple >= 1
fn stacksize(cx: &Cx, t: TypeSym) -> usize {
    let tdef = cx.fetch_type(t);
    match &*tdef {
        TypeDef::SInt(4) => 1,
        TypeDef::SInt(_) => todo!(),
        TypeDef::Bool => 1,
        TypeDef::Tuple(ts) => ts.iter().map(|t| stacksize(cx, *t)).sum(),
        // Essentially a pointer to a function
        TypeDef::Lambda(_, _) => 1,
    }
}

#[cfg(test)]
mod tests {
    use walrus as w;

    use crate::backend::wasm32::*;
    use crate::ir::{self, Expr as E};

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
        compile_expr(cx, &mut m.locals, &mut m.types, symbols, instrs, &expr);

        assert_eq!(symbols.bindings.last().unwrap().len(), 1);
        assert!(symbols.get_local(varname).is_some());

        // Can we then compile the var lookup?
        let expr = ir::TypedExpr {
            t: cx.i32(),
            e: E::Var { name: varname },
        };
        compile_expr(cx, &mut m.locals, &mut m.types, symbols, instrs, &expr);
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
