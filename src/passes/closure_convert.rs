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

fn cc_expr(symtbl: &mut Symtbl, expr: ExprNode) -> ExprNode {
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
            Expr::Lambda { signature, body }
        }
        x => x,
    };
    expr.map(result)
}

fn cc_decl(symtbl: &mut Symtbl, decl: Decl) -> Decl {
    //let type_params: &mut Vec<HashSet<Sym>> = &mut Default::default();
    match decl {
        D::Function {
            name,
            signature,
            body,
        } => {
            let _guard = symtbl.push_scope();
            // Currently, let's just look at the type parameters
            // in the signature, since that's the only way to
            // introduce new types inside the wossname.
            //
            let _renamed_type_params: Vec<_> = signature
                .typeparams
                .clone()
                .into_iter()
                // TODO: The name here is kinda useless..
                .map(|sym| symtbl.bind_new_type(sym).0)
                .collect();

            D::Function {
                name,
                signature,
                body: exprs_map(body, &mut |e| cc_expr(symtbl, e), &mut |e| e),
            }
        }
        D::Const { name, typ, init } => D::Const {
            name,
            typ,
            init: cc_expr(symtbl, init),
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
