//! Enum to integer transformation
//!
//! Basically so that we don't have to follow Rust's rules for enums, iirc.

use crate::passes::*;
use crate::*;

/// Takes an arbitrary type and recursively turns
/// structs into tuples.
fn intify_type(typ: Type) -> Type {
    match typ {
        Type::Enum(_fields) => Type::i32(),
        other => other,
    }
}

/// Take an expression and turn any enum literals into integers.
fn intify_expr(expr: ExprNode, tck: &mut typeck::Tck) -> ExprNode {
    let expr_typeid = tck.get_expr_type(&expr);
    let expr_type = tck.reconstruct(expr_typeid).expect("Should never happen?");
    let new_contents = match &*expr.e {
        E::EnumCtor {
            name: _,
            variant: _,
            value,
        } => E::Lit {
            val: hir::Literal::SizedInteger {
                vl: *value as i128,
                bytes: 4,
            },
            // match &expr_type {
            //     Type::Enum(body) => {
            //         let res = body
            //             .iter()
            //             .find(|(sym, _vl)| sym == valname)
            //             .expect("Enum didn't have the field we selected, should never happen");
            //         // Right now enums are always repr(i32)
            //     }
            //     _other => unreachable!("Should never happen?  {:?}", _other),
            //_other => *expr.e.clone(),
        },
        _other => *expr.e.clone(),
    };

    let new_t = intify_type(expr_type);
    tck.replace_expr_type(&expr, &new_t);
    expr.map(&mut |_| new_contents.clone())
}

/// Takes any enum types or values and replaces them with integers.
fn enum_to_int(ir: Ir, tck: &mut typeck::Tck) -> Ir {
    let mut new_decls = vec![];
    let intify_expr = &mut |e| intify_expr(e, tck);

    for decl in ir.decls.into_iter() {
        let res = decl_map(decl, intify_expr, &mut intify_type);
        new_decls.push(res);
    }
    Ir { decls: new_decls }
}
