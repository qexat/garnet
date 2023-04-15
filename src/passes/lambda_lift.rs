//! Lambda lifting.
//!
//! Basically take any function expression and turn it into
//! a standalone function.
//!
//! Note this does NOT do closure conversion yet; this will break
//! closures. We will need a spearate pass operating before this to take a
//! closure, define an environment type for it, and turn it into
//! a function taking an arg of that environment type.

use crate::passes::*;
use crate::*;

/// Lambda lift a single expr, creating a new toplevel function if necessary.
fn lambda_lift_expr(expr: ExprNode, output_funcs: &mut Vec<D>) -> ExprNode {
    let result = &mut |e| match e {
        E::Lambda { signature, body } => {
            // This is actually the only important bit.
            // TODO: Make a more informative name, maybe including the file and line number or
            // such.
            let lambda_name = INT.gensym("lambda");
            let function_decl = D::Function {
                name: lambda_name,
                signature,
                body: exprs_map(body, &mut |e| lambda_lift_expr(e, output_funcs)),
            };
            output_funcs.push(function_decl);
            E::Var { name: lambda_name }
        }
        x => x,
    };
    expr.map(result)
}

/// A transformation pass that removes lambda expressions and turns
/// them into function decl's.
/// TODO: Doesn't handle actual closures yet though.  That should be a
/// separate pass that happens first?
///
/// TODO: Output is a HIR tree that does not have any lambda expr's in it, which
/// I would like to make un-representable, but don't see a good way
/// to do yet.  Add a tag type of some kind to the Ir?
/// Eeeeeeeeh we might just have to check on it later one way or another,
/// either when doing codegen or when transforming our HIR to a lower-level IR.
pub(super) fn lambda_lift(ir: Ir) -> Ir {
    let mut new_functions = vec![];
    let new_decls: Vec<D> = ir
        .decls
        .into_iter()
        .map(|decl| {
            decl_map(
                decl,
                &mut |e| lambda_lift_expr(e, &mut new_functions),
                &mut |t| t,
            )
        })
        .collect();
    new_functions.extend(new_decls.into_iter());
    Ir {
        decls: new_functions,
    }
}
