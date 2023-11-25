//! Sanity checks the results of the type checker.
//! Doesn't actually change anything, just walks through
//! the entire IR tree and makes sure that every expression
//! has a real type.
//!
//! Technically unnecessary but occasionally useful for
//! debugging stuff.

use crate::hir::*;
use crate::passes::*;
use crate::*;

fn check_expr(expr: ExprNode, tck: &mut typeck::Tck) -> ExprNode {
    let expr_typeid = tck.get_expr_type(&expr);
    let expr_type = tck
        .reconstruct(expr_typeid)
        .unwrap_or_else(|e| panic!("Typechecker couldn't reconstruct something: {}", e));
    match &expr_type {
        Type::Prim(PrimType::UnknownInt) => panic!("Unknown int in expr {:?}", expr.e),
        Type::Prim(PrimType::AnyPtr) => panic!("should be unused"),
        _ => (),
    }

    expr
}

pub(super) fn double_typeck(ir: Ir, tck: &mut typeck::Tck) -> Ir {
    let type_map = &mut |t| t;
    let new_decls = ir
        .decls
        .into_iter()
        .map(|d| decl_map_pre(d, &mut |e| check_expr(e, tck), type_map))
        .collect();
    Ir {
        decls: new_decls,
        ..ir
    }
}
