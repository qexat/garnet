use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::*;

/// Type checking engine
#[derive(Default)]
struct Tck {
    /// Used to generate unique IDs
    id_counter: usize,
    vars: HashMap<TypeId, TypeInfo>,
    types: HashMap<ast::AstId, TypeId>,
}

impl Tck {
    /// Save the type associated with the given expr
    fn set_expr_type(&mut self, expr: &ast::ExprNode, ty: TypeId) {
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
            Func(i, o) => {
                let is: Result<Vec<Type>, String> =
                    i.iter().copied().map(|arg| self.reconstruct(arg)).collect();
                Ok(Type::Func(is?, Box::new(self.reconstruct(*o)?)))
            }
        }
    }
}

/// Basic symbol table that maps names to type ID's
/// and manages scope.
#[derive(Clone)]
struct Symtbl {
    symbols: Rc<RefCell<Vec<HashMap<String, TypeId>>>>,
}

impl Default for Symtbl {
    /// We start with an empty toplevel scope existing.
    fn default() -> Self {
        Self {
            symbols: Rc::new(RefCell::new(vec![HashMap::new()])),
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
    }
}

impl Symtbl {
    fn push_scope(&self) -> ScopeGuard {
        self.symbols.borrow_mut().push(HashMap::new());
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
}

fn infer_lit(lit: &ast::Literal) -> TypeInfo {
    match lit {
        ast::Literal::Integer(_) => TypeInfo::Num,
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
        Lambda { signature, body } => todo!("idk mang"),
        Funcall { func, params } => {
            typecheck_expr(tck, symtbl, func)?;
            let func_type = tck.get_expr_type(func);

            // Synthesize what we know about the function
            // from the call.
            let mut params_list = vec![];
            for param in params {
                typecheck_expr(tck, symtbl, param)?;
                let param_type = tck.get_expr_type(param);
                params_list.push(param_type);
            }
            let rettype_var = tck.insert(TypeInfo::Unknown);
            let funcall_var = tck.insert(TypeInfo::Func(params_list, rettype_var));
            tck.unify(func_type, funcall_var)?;

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
                // Insert info about the function signature
                let mut params = vec![];
                for (_paramname, paramtype) in &signature.params {
                    let p = tck.insert(paramtype.clone());
                    params.push(p);
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
                }
                let last_expr = body.last().expect("empty body, aieeee");
                let last_expr_type = tck.get_expr_type(last_expr);
                tck.unify(last_expr_type, rettype)
                    .expect("Unification of function body failed, aieeee");
            }
        }
    }
    // Print out toplevel symbols
    for (name, id) in symtbl.symbols.borrow().last().unwrap() {
        println!("fn {} type is {:?}", name, tck.reconstruct(*id));
    }
}
/*
pub fn typecheck2() {
    let mut tck = Tck::default();

    // A function with an unknown input
    let i = tck.insert(TypeInfo::Unknown);
    let o = tck.insert(TypeInfo::Num);
    let f0 = tck.insert(TypeInfo::Func(vec![i, o.clone()], o));

    // A function with an unknown output
    let i = tck.insert(TypeInfo::Bool);
    let o = tck.insert(TypeInfo::Unknown);
    let f1 = tck.insert(TypeInfo::Func(vec![i, o.clone()], o));

    // Unify them together...
    tck.unify(f0, f1).unwrap();

    // An instance of the aforementioned function
    let thing = tck.insert(TypeInfo::Ref(f1));

    // ...and compute the resulting type
    println!("Final type = {:?}", tck.reconstruct(thing));
}
*/
