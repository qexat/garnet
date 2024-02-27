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

#[derive(Debug, Default)]
struct ScopeThing {
    type_params: Vec<HashSet<Sym>>,
}

impl ScopeThing {
    fn push_scope(&mut self) {
        self.type_params.push(Default::default())
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
        for scope in self.type_params.iter().rev() {
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

fn get_mentioned_type_params(expr: &ExprNode, scope: &ScopeThing) -> Vec<Sym> {
    let mut accm = vec![];
    fn add_mentioned_type(t: &Type, scope: &ScopeThing, accm: &mut Vec<Sym>) {
        match t {
            Type::Named(nm, params) => {
                if scope.is_type_param(*nm) {
                    accm.push(nm.clone());
                }
                for ty in params {
                    add_mentioned_type(ty, scope, accm);
                }
            }
            _ => todo!(),
        }
    }
    let inner = &mut |expr: &ExprNode| {
        let f = &mut |t| add_mentioned_type(t, scope, &mut accm);
        use hir::Expr::*;
        match &*expr.e {
            Funcall {
                func,
                params,
                type_params,
            } => {
                type_params.iter().for_each(f);
            }
            Let {
                typename: None,
                init,
                ..
            } => todo!(),
            Let {
                typename: Some(typename),
                init,
                ..
            } => todo!(),
            Lambda { .. } => todo!(),
            TypeCtor { .. } => todo!(),
            SumCtor { .. } => todo!(),
            Typecast { .. } => todo!(),
            _ => (),
        }
    };
    expr_iter(expr, inner);
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
