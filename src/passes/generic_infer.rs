//! A pass that takes Named type parameters and turns them into Generics
//! if they are not declared.
//! We could do this in the typechecker but I think it will be simpler as a preprocessing pass.

use crate::hir::Ir;
use crate::passes::*;
use crate::*;

fn generic_infer_expr(expr: ExprNode) -> ExprNode {
    expr
}

pub(super) fn generic_infer(ir: Ir) -> Ir {
    let generic_infer_expr = &mut |expr| generic_infer_expr(expr);
    let generic_infer_ty = &mut |ty| ty;
    let new_decls: Vec<D> = ir
        .decls
        .into_iter()
        .map(|decl| decl_map_pre(decl, generic_infer_expr, generic_infer_ty))
        .collect();
    Ir {
        decls: new_decls,
        ..ir
    }
}
