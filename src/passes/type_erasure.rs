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

fn type_erase_type(typ: Type) -> Type {
    match typ {
        Type::Prim(_) => typ,
        Type::Enum(_) => typ,
        Type::Named(_, _) => todo!(),
        Type::Func(_, _, _) => todo!(),
        Type::Struct(_, _) => todo!(),
        Type::Sum(_, _) => todo!(),
        Type::Array(_, _) => todo!(),
        Type::Generic(_) => todo!(),
    }
}

fn type_erase_expr(expr: ExprNode, tck: &mut typeck::Tck) -> ExprNode {
    let expr_typeid = tck.get_expr_type(&expr);
    let expr_type = tck.reconstruct(expr_typeid).expect("Should never happen");
    let new_contents = match (*expr.e).clone() {
        E::Let {
            varname,
            typename,
            init,
            mutable,
        } => {
            todo!()
        }
        E::StructCtor { body } => todo!(),
        E::StructRef {
            expr: inner_expr,
            elt,
        } => {
            todo!()
        }
        other => other,
    };
    // Change the type of the struct literal expr node to the new tuple
    // We need to do this to any expression because its type may be a struct.
    // And we need to do this after the above because we've changed the
    let new_t = type_erase_type(expr_type.clone());
    if new_t != expr_type {
        info!("Replaced generic type {:?} with {:?}", expr_type, new_t);
        trace!("Expression was rewritten into {:?}", new_contents);
    }
    tck.replace_expr_type(&expr, &new_t);
    expr.map(&mut |_| new_contents.clone())
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
