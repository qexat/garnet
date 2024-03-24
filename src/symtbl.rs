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

use anymap::Map;

use crate::*;

/// A symbol that has been renamed to be globally unique.
/// So basically instead of `foo` being scoped, it gets
/// renamed to `foo_1` where every mention of `foo_1`
/// refers to the same value.
///
/// TODO: The typeck code needs to construct/retrieve its own
/// UniqueSym's, which is why this is public.  It's a bit of
/// a pickle though, since really the goal is to have these
/// make non-unique syms unrepresentable.
///
/// But after the alpha-renaming pass all Sym's are unique already
/// anyway, soooooo...  idk.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct UniqueSym(pub Sym);

#[derive(Clone, Default, Debug)]
struct ScopeFrame {
    symbols: BTreeMap<Sym, UniqueSym>,
    types: BTreeMap<Sym, UniqueSym>,
}

/// A shortcut for a cloneable AnyMap
type CloneMap = Map<dyn anymap::any::CloneAny>;

/// Basic symbol table that maps names to type ID's
/// and manages scope.
/// Looks ugly, works well.
#[derive(Debug, Clone)]
pub struct Symtbl {
    frames: Rc<RefCell<Vec<ScopeFrame>>>,
    /// A mapping from unique symbol names to whatever
    /// we need to know about them.  It's an AnyMap, so
    /// we can just stuff whatever data we need into it.
    unique_symbols: BTreeMap<UniqueSym, CloneMap>,

    /// Same as unique_symbols but for types.
    unique_types: BTreeMap<UniqueSym, CloneMap>,

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
            unique_types: Default::default(),
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
    fn add_builtins(&mut self) {
        for builtin in &*builtins::BUILTINS {
            let _new = self.new_unchanged_symbol(builtin.name);
        }
        // Add another builtin called "main" so the main function
        // doesn't get renamed.
        self.new_unchanged_symbol(Sym::new("main"));
    }

    /// Creates a new symbol with the name of the given one
    /// and a unique numerical suffix.
    fn gensym(&mut self, base: Sym) -> UniqueSym {
        self.gensym_increment += 1;
        let name = format!("{}_{}", &*base.val(), self.gensym_increment);
        UniqueSym(Sym::new(name))
    }

    pub fn push_scope(&self) -> ScopeGuard {
        self.frames.borrow_mut().push(ScopeFrame::default());
        // TODO: This clone is a little cursed 'cause
        // it'll clone the whole `unique_symbols` map,
        // We should probably just wrap the whole symtbl in
        // a Rc<RefCell<>> intead of just the scopeframe.
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
    ///
    /// ...I kinda miss Elixir's convention of having
    /// a function `foo` return a result and `foo!` panic
    /// on error.
    fn really_get_binding(&self, sym: Sym) -> UniqueSym {
        let msg = format!("Symbol {} is not bound!", sym);
        self.get_binding(sym).expect(&msg)
    }

    fn get_type_binding(&self, sym: Sym) -> Option<UniqueSym> {
        for scope in self.frames.borrow().iter().rev() {
            let v = scope.types.get(&sym).cloned();
            if v.is_some() {
                return v;
            }
        }
        None
    }

    fn really_get_type_binding(&self, sym: Sym) -> UniqueSym {
        let msg = format!("Type {} is not bound!", sym);
        self.get_type_binding(sym).expect(&msg)
    }

    /// Get the specified info for the given UniqueSym, or
    /// None if DNE
    pub fn get_info<T>(&self, sym: UniqueSym) -> Option<&T>
    where
        T: anymap::any::CloneAny,
    {
        self.unique_symbols
            .get(&sym)
            .and_then(|anymap| anymap.get::<T>())
    }

    /// Adds the given info struct to the symbol, or
    /// panics if it already exists.
    pub fn put_info<T>(&mut self, sym: UniqueSym, info: T)
    where
        T: anymap::any::CloneAny,
    {
        self.unique_symbols
            .get_mut(&sym)
            .and_then(|anymap| anymap.insert(info));
    }

    pub fn type_exists(&self, sym: Sym) -> bool {
        if sym == Sym::new("Tuple") {
            return true;
        }
        let thing = self.unique_types.get(&UniqueSym(sym));
        // fuck combinators
        match thing {
            Some(_anymap) => true,
            None => false,
        }
    }

    fn get_type_info<T>(&self, sym: UniqueSym) -> Option<&T>
    where
        T: anymap::any::CloneAny,
    {
        // dbg!(&self.unique_types);
        self.unique_types
            .get(&sym)
            .and_then(|anymap| anymap.get::<T>())
    }

    /// BUGGO: See discussion on UniqueSym type
    pub fn get_type_info2<T>(&self, sym: Sym) -> Option<&T>
    where
        T: anymap::any::CloneAny,
    {
        self.get_type_info(UniqueSym(sym))
    }

    /// Adds the given type struct to the symbol, or
    /// panics if it already exists.
    fn put_type_info<T>(&mut self, sym: UniqueSym, info: T)
    where
        T: anymap::any::CloneAny + std::fmt::Debug,
    {
        let anymap = self.unique_types.entry(sym).or_insert_with(CloneMap::new);
        // slightly weird error message formatting 'cause we apparently can't
        // clone `info` easily???
        let info_dbg = format!("{:?}", info);
        if let Some(x) = anymap.insert(info) {
            panic!(
                "Type info for {} already exists: {:?}\nAttempting to replace it with:    {}",
                sym.0, x, info_dbg
            );
        }
    }
    /// BUGGO: same as get_type_info2()
    pub fn put_type_info2<T>(&mut self, sym: Sym, info: T)
    where
        T: anymap::any::CloneAny + std::fmt::Debug,
    {
        self.put_type_info(UniqueSym(sym), info)
    }

    fn binding_exists(&self, sym: Sym) -> bool {
        self.get_binding(sym).is_some()
    }

    fn _unique_exists(&self, sym: UniqueSym) -> bool {
        self.unique_symbols.get(&sym).is_some()
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
        self.unique_symbols.insert(newsym, Map::new());
        newsym
    }

    /// Takes a name and generates a new type param name for it
    pub(crate) fn bind_new_type(&mut self, sym: Sym) -> UniqueSym {
        let newsym = self.gensym(sym);
        self.frames
            .borrow_mut()
            .last_mut()
            .expect("Scope stack underflow")
            .types
            .insert(sym, newsym);
        self.unique_types.insert(newsym, Map::new());
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
    fn _new_local_symbol(&mut self, sym: Sym) -> UniqueSym {
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

    /// Insert a new symbol that is actually identical to the given one,
    fn new_unchanged_symbol(&mut self, sym: Sym) -> UniqueSym {
        if self.binding_exists(sym) {
            panic!("Attempting to create duplicate symbol {}", sym);
        } else {
            let new_sym = UniqueSym(sym);
            self.frames
                .borrow_mut()
                .last_mut()
                .expect("Scope stack underflow")
                .symbols
                .insert(sym, new_sym);
            self.unique_symbols.insert(new_sym, Map::new());
            new_sym
        }
    }

    fn handle_types(&mut self, tys: Vec<Type>) -> Vec<Type> {
        tys.into_iter().map(|t| self.handle_type(t)).collect()
    }

    /// Replace any generic names in the type with unique ones
    fn handle_type(&mut self, ty: Type) -> Type {
        use crate::Type::*;
        match ty {
            Named(nm, type_params) if &*nm.val() == "Tuple" => {
                let new_type_params = self.handle_types(type_params);
                Named(nm, new_type_params)
            }
            Named(nm, type_params) => {
                let nm = self
                    .get_type_binding(nm)
                    .unwrap_or_else(|| panic!("Could not find declared type {}", nm))
                    .0;
                let _guard = self.push_scope();
                let new_type_params = self.handle_types(type_params);
                Named(nm, new_type_params)
            }
            Func(args, rettype, type_params) => {
                let _guard = self.push_scope();
                let new_args = self.handle_types(args);
                let new_rettype = Box::new(self.handle_type(*rettype));
                let new_type_params = self.handle_types(type_params);
                Func(new_args, new_rettype, new_type_params)
            }
            Struct(body, type_params) => {
                let _guard = self.push_scope();
                let new_type_params = self.handle_types(type_params);
                let new_body = body
                    .into_iter()
                    .map(|(nm, ty)| (nm, self.handle_type(ty)))
                    .collect();
                Struct(new_body, new_type_params)
            }
            Sum(body, type_params) => {
                let _guard = self.push_scope();
                let new_type_params = self.handle_types(type_params);
                let new_body = body
                    .into_iter()
                    .map(|(nm, ty)| (nm, self.handle_type(ty)))
                    .collect();
                Sum(new_body, new_type_params)
            }
            Array(t, size) => Array(Box::new(self.handle_type(*t)), size),
            Uniq(inner) => Uniq(Box::new(self.handle_type(*inner))),

            // all these have no subtypes
            Prim(_) => ty,
            Never => ty,
            Enum(_) => ty,
        }
    }

    fn handle_exprs(&mut self, exprs: Vec<hir::ExprNode>) -> Vec<hir::ExprNode> {
        trace!("Handling exprs");
        //let f = &mut |e| self.handle_expr2(e);
        exprs.into_iter().map(|e| self.handle_expr(e)).collect()
        //passes::exprs_map_pre(exprs, f)
    }

    /// TODO: These can't *quite* be part of the Pass framework,
    /// because we need to handle different parts of the expr
    /// in different orders depending on what it is, pushing and
    /// popping scopes in different places as we go.  We *could*
    /// probably make it fit if we tried hard enough but I'm not
    /// sure it's worth it, so for now we'll just brute-force
    /// it and see how it looks at the end.
    fn handle_expr(&mut self, expr: hir::ExprNode) -> hir::ExprNode {
        use hir::Expr::*;
        let f = &mut |e| match e {
            Lit { val } => Lit { val },
            Var { name } => {
                let new_name = self.really_get_binding(name);
                Var { name: new_name.0 }
            }
            BinOp { op, lhs, rhs } => BinOp {
                op,
                lhs: self.handle_expr(lhs),
                rhs: self.handle_expr(rhs),
            },
            UniOp { rhs, op } => UniOp {
                op,
                rhs: self.handle_expr(rhs),
            },
            Block { body } => {
                let _guard = self.push_scope();
                let new_body = self.handle_exprs(body);
                Block { body: new_body }
            }
            Loop { body } => Loop {
                body: self.handle_exprs(body),
            },
            Funcall {
                func,
                params,
                type_params,
            } => {
                let func = self.handle_expr(func);
                let params = self.handle_exprs(params);
                let type_params = self.handle_types(type_params);
                Funcall {
                    func,
                    params,
                    type_params,
                }
            }
            Let {
                varname,
                init,
                typename,
                mutable,
            } => {
                // init expression cannot refer to the same
                // symbol name
                let init = self.handle_expr(init);
                let varname = self.bind_new_symbol(varname).0;
                let typename = typename.map(|t| self.handle_type(t));
                Let {
                    varname,
                    init,
                    typename,
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
                        let t = self.handle_expr(test);
                        let _guard = self.push_scope();
                        let b = self.handle_exprs(body);
                        (t, b)
                    })
                    .collect();
                If { cases }
            }
            EnumCtor {
                name,
                variant,
                value,
            } => {
                // Translate the type name
                // No need to translate the variants 'cause
                // they are namespaced anyway
                let new_name = self.really_get_type_binding(name).0;
                EnumCtor {
                    name: new_name,
                    variant,
                    value,
                }
            }
            TupleCtor { body } => TupleCtor {
                body: self.handle_exprs(body),
            },
            TupleRef { expr, elt } => TupleRef {
                expr: self.handle_expr(expr),
                elt,
            },
            StructCtor { body } => {
                let body = body
                    .into_iter()
                    .map(|(nm, expr)| (nm, self.handle_expr(expr)))
                    .collect();
                StructCtor { body }
            }
            StructRef { expr, elt } => StructRef {
                expr: self.handle_expr(expr),
                elt,
            },
            Assign { lhs, rhs } => Assign {
                lhs: self.handle_expr(lhs),
                rhs: self.handle_expr(rhs),
            },
            Break => Break,
            Lambda { signature, body } => {
                let _scope = self.push_scope();
                // Here we DECLARE the types in the param
                // and rettype, similar to in handle_decl()
                let new_type_params = signature
                    .typeparams
                    .into_iter()
                    .map(|sym| self.bind_new_type(sym).0)
                    .collect();
                let new_rettype = self.handle_type(signature.rettype);

                // We have to do this AFTER adding the type params
                // because the function args can mention
                // type params.
                let new_params = signature
                    .params
                    .into_iter()
                    .map(|(sym, typ)| (self.bind_new_symbol(sym).0, self.handle_type(typ)))
                    .collect();
                let new_sig = hir::Signature {
                    params: new_params,
                    rettype: new_rettype,
                    typeparams: new_type_params,
                };
                let new_body = self.handle_exprs(body);
                Lambda {
                    signature: new_sig,
                    body: new_body,
                }
            }
            Return { retval } => Return {
                retval: self.handle_expr(retval),
            },
            TypeCtor {
                name,
                type_params,
                body,
            } => {
                // Translate the type name
                let new_name = self.really_get_type_binding(name).0;
                // Translate the type params it's called with
                let new_params = self.handle_types(type_params);
                let body = self.handle_expr(body);
                TypeCtor {
                    name: new_name,
                    type_params: new_params,
                    body,
                }
            }
            TypeUnwrap { expr } => TypeUnwrap {
                expr: self.handle_expr(expr),
            },
            SumCtor {
                name,
                variant,
                body,
            } => {
                let new_name = self.really_get_type_binding(name).0;
                SumCtor {
                    name: new_name,
                    variant,
                    body: self.handle_expr(body),
                }
            }
            ArrayCtor { body } => ArrayCtor {
                body: self.handle_exprs(body),
            },
            ArrayRef { expr, idx } => ArrayRef {
                expr: self.handle_expr(expr),
                idx: self.handle_expr(idx),
            },
            Typecast { .. } => {
                todo!()
            }
            Ref { .. } => {
                todo!()
            }
            Deref { .. } => {
                todo!()
            }
        };
        expr.map(f)
    }
}

pub(crate) fn predeclare_decls(symtbl: &mut Symtbl, decls: &[hir::Decl]) {
    use hir::Decl::*;
    for d in decls {
        match d {
            Function {
                name,
                signature: _,
                body: _,
            } => {
                // Don't rename functions already declared as builtins, like "main"
                // Little hacky but does what we want.
                symtbl
                    .get_binding(*name)
                    .unwrap_or_else(|| symtbl.new_unique_symbol(*name));
            }
            TypeDef {
                name,
                params: _,
                typedecl: _,
            } => {
                // Let's rename all types, which I think doesn't
                // quite matter yet but will in the future when we
                // have more powerful modules and such.
                symtbl.bind_new_type(*name);
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
            // This has to happen before we push the scope, so
            // if we have a function arg with the same name as the
            // function it doesn't rename the function.
            let fn_name = symtbl.really_get_binding(name).0;

            let _scope = symtbl.push_scope();
            // These must be handled first 'cause types in the
            // function arguments can refer to them.
            let new_type_params = signature
                .typeparams
                .into_iter()
                .map(|sym| symtbl.bind_new_type(sym).0)
                .collect();
            let new_params = signature
                .params
                .into_iter()
                .map(|(sym, typ)| (symtbl.bind_new_symbol(sym).0, symtbl.handle_type(typ)))
                .collect();
            let new_rettype = symtbl.handle_type(signature.rettype);
            let new_sig = hir::Signature {
                params: new_params,
                rettype: new_rettype,
                typeparams: new_type_params,
            };
            let new_body = symtbl.handle_exprs(body);
            Function {
                name: fn_name,
                signature: new_sig,
                body: new_body,
            }
        }
        TypeDef {
            name,
            params,
            typedecl,
        } => {
            // Ok this is exactly like resolving variables in a function.
            // We bind the declared type params, check that all the types
            // named in the body exist, and rename the ones in the params.
            let ty_name = symtbl.really_get_type_binding(name).0;
            let _scope = symtbl.push_scope();
            let new_params = params
                .into_iter()
                .map(|sym| (symtbl.bind_new_type(sym).0))
                .collect();
            TypeDef {
                name: ty_name,
                params: new_params,
                typedecl: symtbl.handle_type(typedecl),
            }
        }
        Const { name, typ, init } => {
            let new_name = symtbl.really_get_binding(name).0;
            let new_type = symtbl.handle_type(typ);
            let new_body = symtbl.handle_expr(init);
            Const {
                name: new_name,
                typ: new_type,
                init: new_body,
            }
        }
        Import { .. } => todo!(),
    }
}

/// Takes all value and type names and renames them all to
/// remove scope dependence.
///
/// aka alpha-renaming
pub fn resolve_symbols(ir: hir::Ir) -> (hir::Ir, Symtbl) {
    let mut s = Symtbl::default();
    s.add_builtins();
    predeclare_decls(&mut s, &ir.decls);
    let mut ir = ir;
    ir.decls = ir
        .decls
        .into_iter()
        .map(|d| handle_decl(&mut s, d))
        .collect();
    (ir, s)
}
