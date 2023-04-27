//! Placeholder, since monomorph is a little cursed
//! and also not very easy.  So!
//! To get things working for now and let me move on
//! to more interesting things for the moment, we will
//! just turn all functions taking generics into ones
//! taking void pointers.
//!
//! This can't go wrong, right?
//! Right!

use crate::passes::*;

fn type_erase_expr(expr: ExprNode, tck: &mut typeck::Tck) -> ExprNode {
    expr
}

fn type_erase_type(typ: Type) -> Type {
    typ
}

pub(super) fn type_erasure(ir: Ir, tck: &mut typeck::Tck) -> Ir {
    let typeerase_expr = &mut |e| type_erase_expr(e, tck);
    let new_ir = ir
        .decls
        .into_iter()
        .map(|d| decl_map(d, typeerase_expr, &mut type_erase_type))
        .collect();
    Ir { decls: new_ir }
}
