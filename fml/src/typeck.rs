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

    /// Check whether the type contains any references to the given
    /// typeid.  Used for making sure we don't construct nonsense
    /// things like `T = List<T>`.
    ///
    /// TODO: This is made redundant by the assertion at the beginning
    /// of unify, I think?
    fn _contains_type(&self, a: TypeId, b: TypeId) -> bool {
        use TypeInfo::*;
        match self.vars[&a] {
            Unknown => false,
            Ref(t) => b == t || self._contains_type(t, b),
            Named(_, ref args) => args.iter().any(|arg| self._contains_type(*arg, b)),
            Func(ref args, rettype) => {
                b == rettype || args.iter().any(|arg| self._contains_type(*arg, b))
            }
            TypeParam(_) => false,
            Struct(_) => todo!(),
        }
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

    /// Create a new type term out of a known type, such as if we
    /// declare a var's type.
    pub fn insert_known(&mut self, t: &Type) -> TypeId {
        let tinfo = match t {
            Type::Named(s, args) => {
                let new_args = args.iter().map(|t| self.insert_known(t)).collect();
                TypeInfo::Named(s.clone(), new_args)
            }
            Type::Func(args, rettype) => {
                let new_args = args.iter().map(|t| self.insert_known(t)).collect();
                let new_rettype = self.insert_known(rettype);
                TypeInfo::Func(new_args, new_rettype)
            }
            Type::Generic(s) => TypeInfo::TypeParam(s.to_string()),
            Type::Struct(body) => {
                let new_body = body
                    .iter()
                    .map(|(nm, t)| (nm.clone(), self.insert_known(t)))
                    .collect();
                TypeInfo::Struct(new_body)
            }
        };
        self.insert(tinfo)
    }

    /// Make the types of two type terms equivalent (or produce an error if
    /// there is a conflict between them)
    pub fn unify(&mut self, symtbl: &Symtbl, a: TypeId, b: TypeId) -> Result<(), String> {
        assert_ne!(a, b, "Tried to unify a type with itself!  This means our typechecking state has gotten invalid data into it somehow?");
        use TypeInfo::*;
        match (self.vars[&a].clone(), self.vars[&b].clone()) {
            // Follow any references
            (Ref(a), _) => self.unify(symtbl, a, b),
            (_, Ref(b)) => self.unify(symtbl, a, b),

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

            // For type constructors, if their names are the same we try
            // to unify their args
            (Named(n1, args1), Named(n2, args2)) if n1 == n2 && args1.len() == args2.len() => {
                for (arg1, arg2) in args1.iter().zip(args2.iter()) {
                    self.unify(symtbl, *arg1, *arg2)?;
                }
                Ok(())
            }

            // When unifying complex types, we must check their sub-types. This
            // can be trivially implemented for tuples, sum types, etc.
            (Func(a_i, a_o), Func(b_i, b_o)) => {
                if a_i.len() != b_i.len() {
                    return Err(String::from("Arg lists are not same length"));
                }
                for (arg_a, arg_b) in a_i.iter().zip(b_i) {
                    self.unify(symtbl, *arg_a, arg_b)?;
                }
                self.unify(symtbl, a_o, b_o)
            }
            (Struct(body1), Struct(body2)) => {
                for (nm, t1) in body1.iter() {
                    let t2 = body2[nm];
                    self.unify(symtbl, *t1, t2)?;
                }
                // Now we just do it again the other way around
                // which is a dumb but effective way of making sure
                // struct2 doesn't have any fields that struct1 doesn't.
                for (nm, t2) in body2.iter() {
                    let t1 = body1[nm];
                    self.unify(symtbl, t1, *t2)?;
                }
                Ok(())
            }
            (TypeParam(s1), TypeParam(s2)) if s1 == s2 => Ok(()),
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
            Unknown => Err(format!("Cannot infer type for type ID {:?}", id)),
            Ref(id) => self.reconstruct(*id),
            Named(s, args) => {
                let arg_types: Result<Vec<_>, _> =
                    args.iter().map(|x| self.reconstruct(*x)).collect();
                Ok(Type::Named(s.clone(), arg_types?))
            }
            Func(args, rettype) => {
                let real_args: Result<Vec<Type>, String> =
                    args.into_iter().map(|arg| self.reconstruct(*arg)).collect();
                Ok(Type::Func(
                    real_args?,
                    Box::new(self.reconstruct(*rettype)?),
                ))
            }
            TypeParam(name) => Ok(Type::Generic(name.to_owned())),
            Struct(body) => {
                let real_body: Result<HashMap<_, _>, String> = body
                    .iter()
                    .map(|(nm, t)| {
                        let new_t = self.reconstruct(*t)?;
                        Ok((nm.clone(), new_t))
                    })
                    .collect();
                Ok(Type::Struct(real_body?))
            }
        }
    }

    fn print_types(&self) {
        let mut vars_report: Vec<_> = self.vars.iter().collect();
        vars_report.sort_by(|(k1, _), (k2, _)| k1.cmp(k2));
        for (k, v) in vars_report.iter() {
            print!("  ${} => {:?}\n", k.0, v);
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
    ///
    /// This has to actually be an empty hashtable on the first instantitaion
    /// instead of the symtbl, since the symtbl is full of type parameter names from the
    /// enclosing function and those are what we explicitly want to get away from.
    fn instantiate(&mut self, named_types: &mut HashMap<String, TypeId>, t: &Type) -> TypeId {
        let typeinfo = match t {
            Type::Named(s, args) => {
                let inst_args: Vec<_> = args
                    .iter()
                    .map(|t| self.instantiate(named_types, t))
                    .collect();
                TypeInfo::Named(s.clone(), inst_args)
            }
            Type::Generic(s) => {
                // If we know this is is a particular generic, match wiht it
                if let Some(ty) = named_types.get(s) {
                    TypeInfo::Ref(*ty)
                } else {
                    panic!("Referred to unknown generic named {}", s);
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
            Type::Struct(body) => {
                let inst_body = body
                    .iter()
                    .map(|(nm, ty)| (nm.clone(), self.instantiate(named_types, ty)))
                    .collect();
                TypeInfo::Struct(inst_body)
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
    types: Rc<RefCell<Vec<HashMap<String, Type>>>>,
}

impl Default for Symtbl {
    /// We start with an empty toplevel scope existing.
    fn default() -> Self {
        Self {
            symbols: Rc::new(RefCell::new(vec![HashMap::new()])),
            types: Rc::new(RefCell::new(vec![HashMap::new()])),
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
            .types
            .borrow_mut()
            .pop()
            .expect("Scope stack underflow");
    }
}

impl Symtbl {
    fn push_scope(&self) -> ScopeGuard {
        self.symbols.borrow_mut().push(HashMap::new());
        self.types.borrow_mut().push(HashMap::new());
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

    fn add_type(&self, name: impl AsRef<str>, ty: &Type) {
        self.types
            .borrow_mut()
            .last_mut()
            .expect("Scope stack underflow")
            .insert(name.as_ref().to_owned(), ty.to_owned());
    }

    fn get_type(&self, ty: impl AsRef<str>) -> Option<Type> {
        for scope in self.types.borrow().iter().rev() {
            let v = scope.get(ty.as_ref());
            if v.is_some() {
                return v.cloned();
            }
        }
        return None;
    }
}

fn infer_lit(lit: &ast::Literal) -> TypeInfo {
    match lit {
        ast::Literal::Integer(_) => TypeInfo::Named("I32".to_string(), vec![]),
        ast::Literal::Bool(_) => TypeInfo::Named("Bool".to_string(), vec![]),
    }
}
fn typecheck_func_body(
    name: Option<&str>,
    tck: &mut Tck,
    symtbl: &mut Symtbl,
    signature: &ast::Signature,
    body: &[ast::ExprNode],
) -> Result<TypeId, String> {
    // Insert info about the function signature
    let mut params = vec![];
    for (_paramname, paramtype) in &signature.params {
        let p = tck.insert_known(paramtype);
        params.push(p);
    }
    let rettype = tck.insert_known(&signature.rettype);
    let f = tck.insert(TypeInfo::Func(params, rettype));
    // If we have a name (ie, are not a lambda), bind the function's type to its name
    // A gensym might make this easier/nicer someday, but this works for now.
    //
    // Note we do this *before* pushing the scope and checking its body,
    // so this will add the function's name to the outer scope.
    if let Some(n) = name {
        symtbl.add_var(n, f);
    }

    // Add params to function's scope
    let _guard = symtbl.push_scope();
    for (paramname, paramtype) in &signature.params {
        let p = tck.insert_known(paramtype);
        symtbl.add_var(paramname, p);
    }

    // Typecheck body
    for expr in body {
        typecheck_expr(tck, symtbl, expr)?;
        // TODO here: unit type for expressions and such
    }
    let last_expr = body.last().expect("empty body, aieeee");
    let last_expr_type = tck.get_expr_type(last_expr);
    tck.unify(symtbl, last_expr_type, rettype)?;

    println!(
        "Typechecked function {}, types are",
        name.unwrap_or("(lambda)")
    );
    tck.print_types();
    Ok(f)
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
            // Does our let decl have a type attached to it?
            let var_type = if let Some(t) = typename {
                tck.insert_known(t)
            } else {
                tck.insert(TypeInfo::Unknown)
            };
            tck.unify(symtbl, init_expr_type, var_type)?;

            // TODO: Make this expr return unit instead of the
            // type of `init`
            let this_expr_type = init_expr_type;
            tck.set_expr_type(expr, this_expr_type);

            symtbl.add_var(varname, var_type);
            Ok(var_type)
        }
        Lambda { signature, body } => {
            let t = typecheck_func_body(None, tck, symtbl, signature, body)?;
            tck.set_expr_type(expr, t);
            Ok(t)
        }
        StructRef { e, name } => {
            typecheck_expr(tck, symtbl, e)?;
            let struct_type = tck.get_expr_type(e);
            // TODO: Not sure this reconstruct does the Right Thing,
            // especially where type params are involved,
            // but it has the type signature we need.
            //
            // TODO: We really need an operator that "unwraps" a
            // Named type to its contents, the opposite of a type
            // constructor.
            // I suspect when we have that we'll also need to
            // instantiate the named type's generics, as we do
            // with TypeCtor.
            // But right now, we just make a hack that lets you reach
            // inside $ types that are structs by doing thing.name
            match tck.reconstruct(struct_type)? {
                Type::Struct(body) => Ok(tck.insert_known(&body[name])),
                Type::Named(s, _args) => {
                    let hopefully_a_struct = symtbl.get_type(s).unwrap();
                    match hopefully_a_struct {
                        Type::Struct(body) => Ok(tck.insert_known(&body[name])),
                        _other => Err(format!("Yeah I know this is wrong bite me")),
                    }
                }
                other => Err(format!(
                    "Tried to get field named {} but it is an {:?}, not a struct",
                    name, other
                )),
            }
        }
        TupleCtor { body } => {
            let body_types: Result<Vec<_>, _> = body
                .iter()
                .map(|expr| typecheck_expr(tck, symtbl, expr))
                .collect();
            let body_types = body_types?;
            let tuple_type = TypeInfo::Named("Tuple".to_string(), body_types);
            let typeid = tck.insert(tuple_type);
            tck.set_expr_type(expr, typeid);
            Ok(typeid)
        }
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
                    // So when we call a function we need to know what its
                    // type params are.  Then we bind those type parameters
                    // to things.
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
            let funcall_var = tck.insert(TypeInfo::Func(params_list.clone(), rettype_var));

            // Now I guess this is where we make a copy of the function
            // with new generic types.
            // Is this "instantiation"???
            // Yes it is.  Differentiate "type parameters", which are the
            // types a function takes as input (our `Generic` or `TypeParam`
            // things I suppose), from "type variables" which are the TypeId
            // we have to solve for.
            //
            // So we go through the generics the function declares and create
            // new type vars for each of them.
            let named_types = &mut HashMap::new();
            let function_type_params = actual_func_type.get_generic_names();
            for name in function_type_params.iter() {
                let tid = tck.insert(TypeInfo::Unknown);
                named_types.insert(name.clone(), tid);
            }
            let heck = tck.instantiate(named_types, &actual_func_type);
            tck.unify(symtbl, heck, funcall_var)?;

            tck.set_expr_type(expr, rettype_var);
            Ok(rettype_var)
        }

        StructCtor { body } => {
            let body_types: Result<HashMap<_, _>, _> = body
                .iter()
                .map(|(name, expr)| {
                    // ? in map doesn't work too well...
                    match typecheck_expr(tck, symtbl, expr) {
                        Ok(t) => Ok((name.to_string(), t)),
                        Err(s) => Err(s),
                    }
                })
                .collect();
            let body_types = body_types?;
            let struct_type = TypeInfo::Struct(body_types);
            let typeid = tck.insert(struct_type);
            tck.set_expr_type(expr, typeid);
            Ok(typeid)
        }
        TypeCtor {
            name,
            type_params,
            body,
        } => {
            let named_type = symtbl.get_type(name).expect("Unknown type constructor");
            println!("Got type named {}: is {:?}", name, named_type);
            // Ok if we have declared type params we gotta instantiate them
            // to match the type's generics.
            let type_param_names = named_type.get_generic_names();
            assert_eq!(type_params.len(), type_param_names.len());
            let mut type_mapping = HashMap::new();
            for (name, ty) in type_param_names.iter().zip(type_params.iter()) {
                let tid = tck.insert_known(ty);
                type_mapping.insert(name.clone(), tid);
            }
            let tid = tck.instantiate(&mut type_mapping, &named_type);

            //let tid = tck.insert_known(&named_type);
            let body_type = typecheck_expr(tck, symtbl, body)?;
            println!("Expected type is {:?}, body type is {:?}", tid, body_type);
            tck.unify(symtbl, tid, body_type)?;
            println!("Done unifying type ctor");
            // The type the expression returns
            let constructed_type =
                tck.insert_known(&Type::Named(name.clone(), type_params.clone()));
            tck.set_expr_type(expr, constructed_type);
            Ok(tid)
        }
    }
}

// # Example usage
// In reality, the most common approach will be to walk your AST, assigning type
// terms to each of your nodes with whatever information you have available. You
// will also need to call `engine.unify(x, y)` when you know two nodes have the
// same type, such as in the statement `x = y;`.
pub fn typecheck(ast: &ast::Ast) {
    let tck = &mut Tck::default();
    let symtbl = &mut Symtbl::default();
    for decl in &ast.decls {
        use ast::Decl::*;

        match decl {
            Function {
                name,
                signature,
                body,
            } => {
                let t = typecheck_func_body(Some(name), tck, symtbl, signature, body);
                t.unwrap_or_else(|e| {
                    tck.print_types();
                    panic!("Error while typechecking function {}:\n{}", name, e)
                });
            }
            TypeDef { name, params, ty } => {
                // Make sure that there are no unbound generics in the typedef
                // that aren't mentioned in the params.
                let generic_names: HashSet<String> = ty.get_generic_names().into_iter().collect();
                let param_names: HashSet<String> = params.iter().cloned().collect();
                let difference: Vec<_> = generic_names
                    .symmetric_difference(&param_names)
                    // gramble gramble &String
                    .map(|s| s.as_str())
                    .collect();
                if difference.len() != 0 {
                    let differences = difference.join(", ");
                    panic!("Error in typedef {}: Type params do not match generics mentioned in body.  Unmatched types: {}", name, differences);
                }

                // Remember that we know about a type with this name
                symtbl.add_type(name, ty)
            }
            ConstDef { name, ty, init } => {
                // The init expression is typechecked in its own
                // scope, since it may theoretically be a `let` or
                // something that introduces new names inside it.
                let init_type = {
                    let _guard = symtbl.push_scope();
                    let t = typecheck_expr(tck, symtbl, init).unwrap();
                    t
                };
                let decl_type = tck.insert_known(ty);
                tck.unify(&symtbl, decl_type, init_type).unwrap();
                symtbl.add_var(name, decl_type);
            }
        }
    }
    // Print out toplevel symbols
    for (name, id) in symtbl.symbols.borrow().last().unwrap() {
        println!("fn {} type is {:?}", name, tck.reconstruct(*id));
    }
}
