use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::*;

/// Type checking engine
#[derive(Default)]
struct Tck {
    /// Used to generate unique IDs
    id_counter: usize,
    /// Binding from type vars to what we know about the type
    vars: HashMap<TypeId, TypeInfo>,
    /// What we know about the type of each node in the AST.
    types: HashMap<ast::AstId, TypeId>,
}

impl Tck {
    /// Save the type associated with the given expr
    fn set_expr_type(&mut self, expr: &ast::ExprNode, ty: TypeId) {
        assert!(
            self.types.get(&expr.id).is_none(),
            "Redefining known type, not suuuuure if this is bad or not"
        );
        self.types.insert(expr.id, ty);
    }

    fn get_expr_type(&mut self, expr: &ast::ExprNode) -> TypeId {
        *self.types.get(&expr.id).unwrap()
    }

    /// Create a new type term with whatever we have about its type
    pub fn insert(&mut self, info: TypeInfo) -> TypeId {
        // Generate a new ID for our type term
        self.id_counter += 1;
        let id = TypeId(self.id_counter);
        assert!(self.vars.get(&id).is_none(), "Can't happen");
        self.vars.insert(id, info);
        id
    }

    /// Make the types of two type terms equivalent (or produce an error if
    /// there is a conflict between them)
    pub fn unify(&mut self, a: TypeId, b: TypeId) -> Result<(), String> {
        use TypeInfo::*;
        match (self.vars[&a].clone(), self.vars[&b].clone()) {
            // Follow any references
            (Ref(a), _) => self.unify(a, b),
            (_, Ref(b)) => self.unify(a, b),

            // When we don't know anything about either term, assume that
            // they match and make the one we know nothing about reference the
            // one we may know something about
            (Unknown, _) => {
                self.vars.insert(a, TypeInfo::Ref(b));
                Ok(())
            }
            (_, Unknown) => {
                self.vars.insert(b, TypeInfo::Ref(a));
                Ok(())
            }

            // Primitives are trivial to unify
            (Num, Num) => Ok(()),
            (Bool, Bool) => Ok(()),

            // When unifying complex types, we must check their sub-types. This
            // can be trivially implemented for tuples, sum types, etc.
            (Func(a_i, a_o), Func(b_i, b_o)) => {
                if a_i.len() != b_i.len() {
                    return Err(String::from("Arg lists are not same length"));
                }
                for (arg_a, arg_b) in a_i.iter().zip(b_i) {
                    self.unify(*arg_a, arg_b)?;
                }
                self.unify(a_o, b_o)
            }
            (NamedGeneric(s1), NamedGeneric(s2)) if s1 == s2 => Ok(()),
            (NamedGeneric(_s), _other) => {
                // TODO: Make sure the name doesn't refer to
                // itself, such as T = List<T>
                self.vars.insert(a, TypeInfo::Ref(b));
                self.unify(a, b)
            }
            (_other, NamedGeneric(_s)) => {
                self.vars.insert(b, TypeInfo::Ref(a));
                self.unify(a, b)
            }
            // If no previous attempts to unify were successful, raise an error
            (a, b) => Err(format!("Conflict between {:?} and {:?}", a, b)),
        }
    }

    /// Attempt to reconstruct a concrete type from the given type term ID. This
    /// may fail if we don't yet have enough information to figure out what the
    /// type is.
    pub fn reconstruct(&self, id: TypeId) -> Result<Type, String> {
        use TypeInfo::*;
        match &self.vars[&id] {
            Unknown => Err(format!("Cannot infer")),
            Ref(id) => self.reconstruct(*id),
            Num => Ok(Type::Num),
            Bool => Ok(Type::Bool),
            Func(args, rettype) => {
                let real_args: Result<Vec<Type>, String> =
                    args.into_iter().map(|arg| self.reconstruct(*arg)).collect();
                Ok(Type::Func(
                    real_args?,
                    Box::new(self.reconstruct(*rettype)?),
                ))
            }
            NamedGeneric(name) => Ok(Type::Generic(name.to_owned())),
        }
    }

    /// Kinda the opposite of reconstruction; takes a concrete type
    /// and generates a new type with unknown's (type variables) for the generic types (type
    /// parameters)
    ///
    /// The named_types is a *local* binding of generic type names to type variables.
    /// We use this to make multiple mentions of the same type name, such as
    /// `id :: T -> T`, all refer to the same type variable.
    /// Feels Weird but it works.
    fn instantiate(&mut self, named_types: &mut HashMap<String, TypeId>, t: &Type) -> TypeId {
        let typeinfo = match t {
            Type::Num => TypeInfo::Num,
            Type::Bool => TypeInfo::Bool,
            Type::Generic(s) => {
                if let Some(ty) = named_types.get(s) {
                    TypeInfo::Ref(*ty)
                } else {
                    let tid = self.insert(TypeInfo::Unknown);
                    named_types.insert(s.clone(), tid);
                    TypeInfo::Ref(tid)
                }
            }
            Type::Func(args, rettype) => {
                let inst_args: Vec<_> = args
                    .iter()
                    .map(|t| self.instantiate(named_types, t))
                    .collect();
                let inst_ret = self.instantiate(named_types, rettype);
                TypeInfo::Func(inst_args, inst_ret)
            }
        };
        self.insert(typeinfo)
    }
}

/// Basic symbol table that maps names to type ID's
/// and manages scope.
// Looks ugly, works well.
#[derive(Clone)]
struct Symtbl {
    symbols: Rc<RefCell<Vec<HashMap<String, TypeId>>>>,

    /// Function-scoped generics we may have
    generic_vars: Rc<RefCell<Vec<HashMap<String, TypeId>>>>,
}

impl Default for Symtbl {
    /// We start with an empty toplevel scope existing.
    fn default() -> Self {
        Self {
            symbols: Rc::new(RefCell::new(vec![HashMap::new()])),
            generic_vars: Rc::new(RefCell::new(vec![HashMap::new()])),
        }
    }
}

pub struct ScopeGuard {
    scope: Symtbl,
}

impl Drop for ScopeGuard {
    fn drop(&mut self) {
        self.scope
            .symbols
            .borrow_mut()
            .pop()
            .expect("Scope stack underflow");
        self.scope
            .generic_vars
            .borrow_mut()
            .pop()
            .expect("Generic scope stack underflow");
    }
}

impl Symtbl {
    fn push_scope(&self) -> ScopeGuard {
        self.symbols.borrow_mut().push(HashMap::new());
        self.generic_vars.borrow_mut().push(HashMap::new());
        ScopeGuard {
            scope: self.clone(),
        }
    }

    fn add_var(&self, var: impl AsRef<str>, ty: TypeId) {
        self.symbols
            .borrow_mut()
            .last_mut()
            .expect("Scope stack underflow")
            .insert(var.as_ref().to_owned(), ty);
    }

    /// Checks whether the var exists in the currently alive scopes
    fn get_var_binding(&self, var: impl AsRef<str>) -> Option<TypeId> {
        for scope in self.symbols.borrow().iter().rev() {
            let v = scope.get(var.as_ref());
            if v.is_some() {
                return v.cloned();
            }
        }
        return None;
    }

    fn lookup_generic(&self, name: &str) -> TypeId {
        for scope in self.generic_vars.borrow().iter().rev() {
            let v = scope.get(name);
            return v.unwrap().clone();
        }
        panic!("No generic found, aieee");
    }

    fn add_generic(&mut self, name: &str, typeid: TypeId) {
        self.generic_vars
            .borrow_mut()
            .last_mut()
            .expect("Scope stack underflow")
            .insert(name.to_owned(), typeid);
    }
}

fn infer_lit(lit: &ast::Literal) -> TypeInfo {
    match lit {
        ast::Literal::Integer(_) => TypeInfo::Num,
        ast::Literal::Bool(_) => TypeInfo::Bool,
    }
}

fn typecheck_expr(
    tck: &mut Tck,
    symtbl: &mut Symtbl,
    expr: &ast::ExprNode,
) -> Result<TypeId, String> {
    use ast::Expr::*;
    match &*expr.node {
        Lit { val } => {
            let lit_type = infer_lit(val);
            let typeid = tck.insert(lit_type);
            tck.set_expr_type(expr, typeid);
            Ok(typeid)
        }
        Var { name } => {
            let ty = symtbl
                .get_var_binding(name)
                .unwrap_or_else(|| panic!("unbound var: {:?}", name));
            tck.set_expr_type(expr, ty);
            Ok(ty)
        }
        Let {
            varname,
            typename,
            init,
        } => {
            typecheck_expr(tck, symtbl, init)?;
            let init_expr_type = tck.get_expr_type(init);
            let var_type = tck.insert(typename.clone());
            tck.unify(init_expr_type, var_type)?;

            // TODO: Make this expr return unit instead of the
            // type of `init`
            let this_expr_type = init_expr_type;
            tck.set_expr_type(expr, this_expr_type);

            symtbl.add_var(varname, var_type);
            Ok(var_type)
        }
        Lambda {
            signature: _,
            body: _,
        } => todo!("idk mang"),
        Funcall { func, params } => {
            // Oh, defined generics are "easy".
            // Each time I call a function I create new type
            // vars for its generic args.

            let func_type = typecheck_expr(tck, symtbl, func)?;
            // We know this will work because we require full function signatures
            // on our functions.
            let actual_func_type = tck.reconstruct(func_type)?;
            match &actual_func_type {
                Type::Func(_args, _rettype) => {
                    println!("Calling function {:?} is {:?}", func, actual_func_type);
                }
                _ => panic!("Tried to call something not a function"),
            }

            // Synthesize what we know about the function
            // from the call.
            let mut params_list = vec![];
            for param in params {
                typecheck_expr(tck, symtbl, param)?;
                let param_type = tck.get_expr_type(param);
                params_list.push(param_type);
            }
            // We don't know what the expected return type of the function call
            // is yet; we make a type var that will get resolved when the enclosing
            // expression is.
            let rettype_var = tck.insert(TypeInfo::Unknown);
            let funcall_var = tck.insert(TypeInfo::Func(params_list, rettype_var));

            // Now I guess this is where we make a copy of the function
            // with new generic types.
            // Is this "instantiation"???
            // Yes it is.  Differentiate "type parameters", which are the
            // types a function takes as input (our `Generic` or `NamedGeneric`
            // things I suppose), from "type variables" which are the TypeId
            // we have to solve for.
            let named_types = &mut HashMap::new();
            let heck = tck.instantiate(named_types, &actual_func_type);
            //tck.unify(func_type, funcall_var)?;
            tck.unify(heck, funcall_var)?;

            tck.set_expr_type(expr, rettype_var);
            Ok(rettype_var)
        }
    }
}

// # Example usage
// In reality, the most common approach will be to walk your AST, assigning type
// terms to each of your nodes with whatever information you have available. You
// will also need to call `engine.unify(x, y)` when you know two nodes have the
// same type, such as in the statement `x = y;`.
pub fn typecheck(ast: &ast::Ast) {
    let mut tck = Tck::default();
    let mut symtbl = Symtbl::default();
    for decl in &ast.decls {
        use ast::Decl::*;

        match decl {
            Function {
                name,
                signature,
                body,
            } => {
                // Ok, for generic types as function inputs... we collect
                // all the types that are generics, and save each of them
                // in our scope.  We save them as... something, and when
                // we unify them we should be able to follow references to
                // them as normal?
                //
                // Things get a little weird when we define vs. call.
                // When we call a function with generics we are providing
                // types for it like they are function args.  So we bind
                // them to a scope the same way we bind function args.
                //
                // To start with let's just worry about defining.

                // Insert info about the function signature
                let mut params = vec![];
                for (_paramname, paramtype) in &signature.params {
                    let p = tck.insert(paramtype.clone());
                    params.push(p);
                    if let Some(name) = paramtype.generic_name() {
                        symtbl.add_generic(name, p);
                    }
                }
                let rettype = tck.insert(signature.rettype.clone());
                let f = tck.insert(TypeInfo::Func(params, rettype));
                symtbl.add_var(name, f);

                // Add params to function's scope
                let _guard = symtbl.push_scope();
                for (paramname, paramtype) in &signature.params {
                    let p = tck.insert(paramtype.clone());
                    symtbl.add_var(paramname, p);
                }

                // Typecheck body
                for expr in body {
                    typecheck_expr(&mut tck, &mut symtbl, expr).expect("Typecheck failure");
                    // TODO here: unit type for expressions and such
                }
                let last_expr = body.last().expect("empty body, aieeee");
                let last_expr_type = tck.get_expr_type(last_expr);
                tck.unify(last_expr_type, rettype)
                    .expect("Unification of function body failed, aieeee");

                println!("Typechecked {}, types are", name);
                let mut vars_report: Vec<_> = tck.vars.iter().collect();
                vars_report.sort_by(|(k1, _), (k2, _)| k1.cmp(k2));
                for (k, v) in vars_report.iter() {
                    print!("  ${} => {:?}\n", k.0, v);
                }
            }
        }
    }
    // Print out toplevel symbols
    for (name, id) in symtbl.symbols.borrow().last().unwrap() {
        println!("fn {} type is {:?}", name, tck.reconstruct(*id));
    }
}
