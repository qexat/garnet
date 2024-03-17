//! Closure conversion.
//!
//! Takes function expressions and rewrites
//! them to make any captured variables (and type
//! parameters!) explicit.
//!
//! From https://matt.might.net/articles/closure-conversion/
//! the key points are to rewrite a lambda term into
//! a closure-creation term that returns both a closure
//! and an environment?  Then rewrite function calls calling
//! the variable to include the environment.
//!
//! Right now we have no borrowing or anything, so all values
//! are *copied* into their environment.
//!
//! So for a first stab at this, let us merely worry about type
//! parameters.

use std::collections::HashSet;

use crate::hir::{Decl, Expr, Ir};
use crate::passes::*;

// This might be the time to split the Symtbl into the
// scope bit and the renamed table.
use crate::symtbl::{self, Symtbl};

/// A dumb and simple scope for type params.
/// Will eventually get folded into Symtbl but I want to
/// be able to figure out what shape it needs to be first without
/// needing to alter anything that touches Symtbl.
#[derive(Debug, Default)]
struct ScopeThing {
    type_params: Vec<HashSet<Sym>>,
}

struct ScopeGuard<'a> {
    scope: &'a mut ScopeThing,
}

impl<'a> Drop for ScopeGuard<'a> {
    fn drop(&mut self) {
        self.scope.pop_scope()
    }
}

impl ScopeThing {
    fn push_scope(&mut self) -> ScopeGuard<'_> {
        self.type_params.push(Default::default());
        ScopeGuard { scope: self }
    }

    fn pop_scope(&mut self) {
        self.type_params
            .pop()
            .expect("Scope stack underflow aieeee");
    }

    fn add_type_params(&mut self, params: &[Sym]) {
        let top = self.type_params.last_mut().expect("Scope stack underflow");
        top.extend(params.iter().cloned());
    }

    fn is_type_param(&self, ty: Sym) -> bool {
        let msg = format!("Walking down scope {:?}", self);
        dbg!(msg);
        for scope in self.type_params.iter().rev() {
            let msg = format!("is {} a type param? {}", ty, scope.contains(&ty));
            dbg!(&msg);
            if scope.contains(&ty) {
                return true;
            }
        }
        return false;
    }
}

fn cc_type(scope: &mut ScopeThing, ty: Type) -> Type {
    match &ty {
        Type::Struct(fields, params) => ty,
        Type::Sum(fields, generics) => ty,
        Type::Array(ty, len) => *ty.clone(),
        Type::Func(args, rettype, typeparams) => ty,
        // Not super sure whether this is necessary, but can't hurt.
        Type::Named(nm, tys) => ty,
        Type::Prim(_) => ty,
        Type::Never => ty,
        Type::Enum(_) => ty,
        Type::Uniq(t) => ty,
    }
}

/// Returns a list of type parameters that are not defined in the given scope.
fn get_free_type_params(expr: &ExprNode, scope: &mut ScopeThing) -> HashSet<Sym> {
    let mut accm = HashSet::new();
    /// Take a type and collect any named types that aren't in the symbol table
    ///  into an array
    ///
    /// TODO: Eventually we can probably refactor this into using type_iter()
    /// but handling type params is weird enough that for now I want to keep
    /// it separate.
    fn collect_free_types(t: &Type, scope: &mut ScopeThing, accm: &mut HashSet<Sym>) {
        use Type::*;
        match t {
            Named(nm, params) => {
                if !scope.is_type_param(*nm) {
                    accm.insert(nm.clone());
                }
                for ty in params {
                    collect_free_types(ty, scope, accm);
                }
            }
            Func(params, rettype, typeparams) => {
                for ty in params {
                    collect_free_types(ty, scope, accm);
                }
                collect_free_types(rettype, scope, accm);
                for ty in typeparams {
                    collect_free_types(ty, scope, accm);
                }
            }
            Struct(body, typeparams) | Sum(body, typeparams) => {
                for (_nm, ty) in body {
                    collect_free_types(ty, scope, accm);
                }
                for ty in typeparams {
                    collect_free_types(ty, scope, accm);
                }
            }
            Array(ty, _) | Uniq(ty) => {
                collect_free_types(ty, scope, accm);
            }
            // No type parameters for these types
            Prim(_) | Never | Enum(_) => (),
        }
    }
    use hir::Expr::*;
    match &*expr.e {
        BinOp { lhs, rhs, .. } => {
            accm.extend(get_free_type_params(lhs, scope));
            accm.extend(get_free_type_params(rhs, scope));
        }
        UniOp { rhs, .. } => accm.extend(get_free_type_params(rhs, scope)),
        Block { body } | Loop { body } => {
            for expr in body {
                accm.extend(get_free_type_params(expr, scope));
            }
        }
        Funcall {
            func: _,
            params: _,
            type_params,
        } => {
            // Can't raise this closure definition above the match
            // because the lifetimes of scope get squirrelly when
            // we borrow it.
            //
            // We could make ScopeThing use internal mutability
            // but it's not really worth it yet.
            let f = &mut |t| collect_free_types(t, scope, &mut accm);
            // These are type params *passed into* the funcall
            type_params.iter().for_each(f);
        }
        Let {
            typename: None,
            init: _,
            ..
        } => (),
        Let {
            typename: Some(ty),
            init: _,
            ..
        } => {
            collect_free_types(ty, scope, &mut accm);
        }
        If { cases } => {
            for (test, exprs) in cases {
                accm.extend(get_free_type_params(test, scope));
                for expr in exprs {
                    accm.extend(get_free_type_params(expr, scope));
                }
            }
        }
        Lambda { signature, body: _ } => {
            // These are type params *defined by* the funcall
            let guard = scope.push_scope();
            guard.scope.add_type_params(&signature.typeparams);
            let ty = signature.to_type();
            collect_free_types(&ty, guard.scope, &mut accm);

            // TODO: uhhhhhh this might not recurse properly because
            // we have to have the push/pop *around* the body
            // traversal.
            // would a scope guard work?
        }
        TypeCtor { .. } => todo!(),
        SumCtor { .. } => todo!(),
        Typecast { .. } => todo!(),
        Lit { .. } | Var { .. } | EnumCtor { .. } => (),
        _other => todo!(),
        // other => expr_iter(expr, &mut |e| {
        //     let res = get_free_type_params(e, scope);
        //     accm.extend(res);
        // }),
    }
    accm
}

fn expr_contains_type_param(expr: &ExprNode, scope: &ScopeThing) -> bool {
    false
}

fn exprs_contains_type_param(expr: &[ExprNode], scope: &ScopeThing) -> bool {
    let f = |e| expr_contains_type_param(e, scope);
    expr.iter().any(f)
}

fn cc_expr(symtbl: &mut Symtbl, scope: &mut ScopeThing, expr: ExprNode) -> ExprNode {
    let result = &mut |e| match e {
        Expr::Lambda { signature, body } => {
            // Ok, so what we need to do is look at the signature
            // for types that are declared in the type params of
            // upper scopes, and if we see any of them in the
            // signature wwe will need to add type params to this
            // lambda.
            //
            // We will also need to look for mentions of it
            // in the function body, such as in let statements
            // or function calls or further lambdas
            //
            // ....hmmm, this is kinda screwy 'cause the type param
            // may not be mentioned at the toplevel of the type.
            // For example it could be ^[]T or such.
            // See Type::get_toplevel_type_params() and uncurse
            // it as appropriate.
            //
            // Well what we really need to check is whether a mentioned
            // type is a type param, so we just recurse through any type
            // we see and do that...
            // We can do that with type_map(), right?

            if exprs_contains_type_param(&body, scope) {
                // ...which is not mentioned in the lambda's type params...
                // Then rewrite the signature to contain the type params the body uses.
                Expr::Lambda { signature, body }
            } else {
                // no change necessary
                Expr::Lambda { signature, body }
            }
        }
        x => x,
    };
    expr.map(result)
}

fn cc_decl(symtbl: &mut Symtbl, decl: Decl) -> Decl {
    //let type_params: &mut Vec<HashSet<Sym>> = &mut Default::default();
    let scope = &mut ScopeThing::default();
    scope.push_scope();
    match decl {
        D::Function {
            name,
            signature,
            body,
        } => {
            let _guard = symtbl.push_scope();
            // Currently, let's just look at the type parameters
            // in the signature, since that's the only way to
            // introduce new types inside the body of the closure.
            let _renamed_type_params: Vec<_> = signature
                .typeparams
                .clone()
                .into_iter()
                // TODO: The name here is kinda useless..
                .map(|sym| symtbl.bind_new_type(sym).0)
                .collect();

            scope.add_type_params(&signature.typeparams);

            D::Function {
                name,
                signature,
                body: exprs_map(body, &mut |e| cc_expr(symtbl, scope, e), &mut |e| e),
            }
        }
        D::Const { name, typ, init } => D::Const {
            name,
            typ,
            init: cc_expr(symtbl, scope, init),
        },
        D::TypeDef {
            name,
            params,
            typedecl,
        } => D::TypeDef {
            name,
            params,
            typedecl,
        },
        D::Import { .. } => decl,
    }
}

/// I am somewhat dissatisfied that we have to do symbol table
/// scope-wrangling stuff for this, and then do it again for
/// the symtbl alpha-renaming, but oh well.  They're different
/// things, this adds information and the renaming just
/// transforms it into something more convenient.
pub fn closure_convert(ir: Ir) -> Ir {
    let symtbl = &mut Symtbl::default();
    symtbl::predeclare_decls(symtbl, &ir.decls);
    let new_decls: Vec<Decl> = ir
        .decls
        .into_iter()
        .map(|decl| cc_decl(symtbl, decl))
        .collect();
    Ir {
        decls: new_decls,
        ..ir
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_free_type_param() {
        let test_cases = vec![
            ("3", vec![]),
            ("let foo = bar", vec![]),
            ("let foo I32 = bar", vec![]),
            ("let foo T = bar", vec![Sym::new("T")]),
            (
                "let foo T(Thing) = bar",
                vec![Sym::new("T"), Sym::new("Thing")],
            ),
            ("fn(x I32) I32 = x end", vec![]),
            ("fn(x T) T = x end", vec![Sym::new("T")]),
            ("fn(|T| x T) T = x end", vec![]),
            (
                r#"fn(|A| x A) A = 
    let f = fn(x A) A = x end
    f(x)
end"#,
                vec![],
            ),
        ];
        let scope = &mut ScopeThing::default();
        for (src, expected) in test_cases {
            let ir = compile_to_hir_expr(src);
            let frees = get_free_type_params(&ir, scope);
            use std::iter::FromIterator;
            let expected: HashSet<Sym> = HashSet::from_iter(expected);
            assert_eq!(frees, expected);
        }
    }
}
