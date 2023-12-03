//! Symbol table.
//!
//! Basically we need to make the symtable persistent,
//! so we only need to figure out all meta-info about it
//! once instead of needing to walk through scopes multiple
//! times and then throw the scope information away when
//! we're done with it.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use crate::*;

/// A symbol that has been renamed to be globally unique.
/// So basically instead of `foo` being scoped, it gets
/// renamed to `foo_1` where every mention of `foo_1`
/// refers to the same value.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct UniqueSym(Sym);

#[derive(Clone, Default, Debug)]
struct ScopeFrame {
    symbols: BTreeMap<Sym, UniqueSym>,
    _types: BTreeMap<Sym, Type>,
}

#[derive(Clone, Debug, Default)]
struct SymbolInfo {}

/// Basic symbol table that maps names to type ID's
/// and manages scope.
/// Looks ugly, works well.
#[derive(Clone, Debug)]
pub struct Symtbl {
    frames: Rc<RefCell<Vec<ScopeFrame>>>,
    /// A mapping from unique symbol names to whatever
    /// we need to know about them.
    unique_symbols: BTreeMap<UniqueSym, SymbolInfo>,

    /// Every time we generate a new symbol we increment
    /// this.
    ///
    /// TODO: It'd be kinda nice to have foo_1 , bar_1, foo_2,
    /// baz_1, foo_3 etc but don't need it for now.
    ///
    /// Also it might be nice to have each variable have a unique
    /// number someday.  ...though symbols already kinda do, hm.
    /// That can go in the SymbolInfo tho.
    gensym_increment: usize,
}

impl Default for Symtbl {
    /// We start with an empty toplevel scope existing,
    /// then add some builtin's to it.
    fn default() -> Self {
        Self {
            frames: Rc::new(RefCell::new(vec![ScopeFrame::default()])),
            unique_symbols: Default::default(),
            gensym_increment: 0,
        }
    }
}

pub struct ScopeGuard {
    scope: Symtbl,
}

impl Drop for ScopeGuard {
    fn drop(&mut self) {
        self.scope
            .frames
            .borrow_mut()
            .pop()
            .expect("Scope stack underflow");
    }
}

impl Symtbl {
    /// Creates a new symbol with the name of the given one
    /// and a unique numerical suffix.
    fn gensym(&mut self, base: Sym) -> UniqueSym {
        self.gensym_increment += 1;
        let name = format!("{}_{}", &*base.val(), self.gensym_increment);
        UniqueSym(Sym::new(name))
    }

    fn push_scope(&self) -> ScopeGuard {
        self.frames.borrow_mut().push(ScopeFrame::default());
        ScopeGuard {
            scope: self.clone(),
        }
    }

    /// Returns the UniqueSym that the given Sym is *currently*
    /// bound to, or None if DNE
    fn get_binding(&self, sym: Sym) -> Option<UniqueSym> {
        for scope in self.frames.borrow().iter().rev() {
            let v = scope.symbols.get(&sym).cloned();
            if v.is_some() {
                return v;
            }
        }
        None
    }

    /// Get a reference to the symbol most recently bound
    /// with that name, or panic if it does not exist
    fn really_get_binding(&self, sym: Sym) -> UniqueSym {
        let msg = format!("Symbol {} is not bound!", sym);
        self.get_binding(sym).expect(&msg)
    }

    /// Get the SymbolInfo for the given UniqueSym, or
    /// None if DNE
    fn get_info(&self, sym: UniqueSym) -> Option<&SymbolInfo> {
        self.unique_symbols.get(&sym)
    }

    fn binding_exists(&self, sym: Sym) -> bool {
        self.get_binding(sym).is_some()
    }

    fn unique_exists(&self, sym: UniqueSym) -> bool {
        self.get_info(sym).is_some()
    }

    /// Takes a symbol, generates a new UniqueSym for it,
    /// and stuffs it into the topmost scope, overwriting
    /// any symbol of the same name already there.
    fn bind_new_symbol(&mut self, sym: Sym) -> UniqueSym {
        let newsym = self.gensym(sym);
        self.frames
            .borrow_mut()
            .last_mut()
            .expect("Scope stack underflow")
            .symbols
            .insert(sym, newsym);
        self.unique_symbols.insert(newsym, SymbolInfo::default());
        newsym
    }

    /// Introduce a new symbol, or panic if that name is already
    /// defined (ie, it is a name conflict, like two functions with
    /// the same name)
    fn new_unique_symbol(&mut self, sym: Sym) -> UniqueSym {
        if self.binding_exists(sym) {
            panic!("Attempting to create duplicate symbol {}", sym);
        } else {
            self.bind_new_symbol(sym)
        }
    }

    /// Introduce a new symbol as long as it doesn't clash with one
    /// currently in the top scope.
    fn new_local_symbol(&mut self, sym: Sym) -> UniqueSym {
        // Does the symbol exist in the topmost scope?
        if self
            .frames
            .borrow_mut()
            .last_mut()
            .expect("Scope stack underflow")
            .symbols
            .contains_key(&sym)
        {
            panic!("Attempting to create duplicate global symbol {}", sym)
        } else {
            self.bind_new_symbol(sym)
        }
    }

    fn handle_exprs(&mut self, exprs: Vec<hir::ExprNode>) -> Vec<hir::ExprNode> {
        let f = &mut |e| self.handle_expr2(e);
        //exprs.into_iter().map(|e| e.map(f)).collect()
        passes::exprs_map_pre(exprs, f)
    }

    /// TODO: These can't *quite* be part of the Pass framework,
    /// because we need to handle different parts of the expr
    /// in different orders depending on what it is, pushing and
    /// popping scopes in different places as we go.  We *could*
    /// probably make it fit if we tried hard enough but I'm not
    /// sure it's worth it, so for now we'll just brute-force
    /// it and see how it looks at the end.
    fn handle_expr(&mut self, expr: hir::Expr) -> hir::Expr {
        use hir::Expr::*;
        // you know things are going well when you accidentally invent
        // the Y combinator
        let ff = &mut |e| self.handle_expr(e);
        match expr {
            Var { name } => {
                let new_name = self.really_get_binding(name);
                Var { name: new_name.0 }
            }
            BinOp { op, lhs, rhs } => BinOp {
                op,
                lhs: lhs.map(ff),
                rhs: rhs.map(ff),
            },
            /*
            UniOp { rhs, .. } => {
                self.handle_expr(rhs);
            }
            Block { body } => {
                let _guard = self.push_scope();
                self.handle_exprs(body);
            }
            Loop { .. } => todo!(),
            Funcall {
                func,
                params,
                type_params: _,
            } => {
                self.handle_expr(func);
                self.handle_exprs(params);
            }
            Let { varname, init, .. } => {
                // init expression cannot refer to the same
                // symbol name
                self.handle_expr(init);
                self.bind_new_symbol(*varname);
            }
            If { cases } => {
                for (test, body) in cases {
                    // The test cannot introduce a new scope unless
                    // it contains some cursed structure like a block
                    // or fn that introduces a new scope anyway, so.
                    self.handle_expr(test);
                    let _guard = self.push_scope();
                    self.handle_exprs(body);
                }
            }
            EnumCtor { name, .. } => {
                todo!()
            }
            TupleCtor { body } => {
                self.handle_exprs(body);
            }
            TupleRef { expr: e, .. } => {
                self.handle_expr(e);
            }
            StructCtor { body } => {
                for (_nm, expr) in body {
                    self.handle_expr(expr);
                }
            }
            StructRef { expr: e, .. } => {
                todo!()
            }
            Assign { .. } => {
                todo!()
            }
            Break => {}
            Lambda { .. } => {
                todo!()
            }
            Return { retval } => {
                self.handle_expr(retval);
            }
            TypeCtor {
                name,
                type_params: _,
                body,
            } => {
                self.bind_new_symbol(*name);
                self.handle_expr(body);
            }
            TypeUnwrap { expr: e } => {
                self.handle_expr(e);
            }
            SumCtor { .. } => {
                todo!()
            }
            ArrayCtor { .. } => {
                todo!()
            }
            ArrayRef { .. } => {
                todo!()
            }
            Typecast { .. } => {
                todo!()
            }
            Ref { .. } => {
                todo!()
            }
            Deref { .. } => {
                todo!()
            }
                            */
            other => other,
        }
    }

    fn handle_expr2(&mut self, expr: hir::ExprNode) -> hir::ExprNode {
        use hir::Expr::*;
        let f = &mut |e| match e {
            Var { name } => {
                let new_name = self.really_get_binding(name);
                Var { name: new_name.0 }
            }
            Block { body } => {
                let _guard = self.push_scope();
                Block {
                    body: self.handle_exprs(body),
                }
            }
            Loop { body } => {
                let _guard = self.push_scope();
                Loop {
                    body: self.handle_exprs(body),
                }
            }
            Let {
                varname,
                typename,
                init,
                mutable,
            } => {
                // init expression cannot refer to the same
                // symbol name
                let init = self.handle_expr2(init);
                let varname = self.bind_new_symbol(varname).0;
                Let {
                    varname,
                    typename,
                    init,
                    mutable,
                }
            }
            If { cases } => {
                let cases = cases
                    .into_iter()
                    .map(|(test, body)| {
                        // The test cannot introduce a new scope unless
                        // it contains some cursed structure like a block
                        // or fn that introduces a new scope anyway, so.
                        let test = self.handle_expr2(test);
                        let _guard = self.push_scope();
                        let body = self.handle_exprs(body);
                        (test, body)
                    })
                    .collect();
                If { cases }
            }
            /*
            EnumCtor { name, .. } => {
                todo!()
            }
            TupleCtor { body } => {
                self.handle_exprs(body);
            }
            TupleRef { expr: e, .. } => {
                self.handle_expr(e);
            }
            StructCtor { body } => {
                for (_nm, expr) in body {
                    self.handle_expr(expr);
                }
            }
            StructRef { expr: e, .. } => {
                todo!()
            }
            Assign { .. } => {
                todo!()
            }
            Break => {}
            Lambda { .. } => {
                todo!()
            }
            Return { retval } => {
                self.handle_expr(retval);
            }
            TypeCtor {
                name,
                type_params: _,
                body,
            } => {
                self.bind_new_symbol(*name);
                self.handle_expr(body);
            }
            TypeUnwrap { expr: e } => {
                self.handle_expr(e);
            }
            SumCtor { .. } => {
                todo!()
            }
            ArrayCtor { .. } => {
                todo!()
            }
            ArrayRef { .. } => {
                todo!()
            }
            Typecast { .. } => {
                todo!()
            }
            Ref { .. } => {
                todo!()
            }
            Deref { .. } => {
                todo!()
            }
                            */
            other => other,
        };
        expr.map(f)
    }
}

fn predeclare_decls(symtbl: &mut Symtbl, decls: &[hir::Decl]) {
    use hir::Decl::*;
    for d in decls {
        match d {
            Function {
                name,
                signature: _,
                body: _,
            } => {
                symtbl.new_unique_symbol(*name);
            }
            TypeDef {
                name: _,
                params: _,
                typedecl: _,
            } => {
                //todo!()
            }
            Const {
                name,
                typ: _,
                init: _,
            } => {
                symtbl.new_unique_symbol(*name);
            }
            Import { .. } => todo!(),
        }
    }
}

fn handle_decl(symtbl: &mut Symtbl, decl: hir::Decl) -> hir::Decl {
    use hir::Decl::*;
    match decl {
        Function {
            name,
            signature,
            body,
        } => {
            let _scope = symtbl.push_scope();
            let new_params = signature
                .params
                .into_iter()
                .map(|(sym, typ)| (symtbl.bind_new_symbol(sym).0, typ))
                .collect();
            let new_sig = hir::Signature {
                params: new_params,
                rettype: signature.rettype,
                typeparams: signature.typeparams,
            };
            let new_body = symtbl.handle_exprs(body);
            // Don't rename function named "main"
            let fn_name = if name == Sym::new("main") {
                name
            } else {
                symtbl.really_get_binding(name).0
            };
            Function {
                name: fn_name,
                signature: new_sig,
                body: new_body,
            }
        }
        TypeDef {
            name: _,
            params: _,
            typedecl: _,
        } => {
            //todo!()
            decl
        }
        Const {
            name: _,
            typ: _,
            init: _,
        } => {
            todo!()
        }
        Import { .. } => todo!(),
    }
}

/// Takes all value and type names and renames them all to
/// remove scope dependence.
pub fn resolve_symbols(ir: hir::Ir) -> (hir::Ir, Symtbl) {
    let mut s = Symtbl::default();
    predeclare_decls(&mut s, &ir.decls);
    let mut ir = ir;
    ir.decls = ir
        .decls
        .into_iter()
        .map(|d| handle_decl(&mut s, d))
        .collect();
    (ir, s)
}
