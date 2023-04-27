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

/// This is a little squirrelly 'cause we don't actually
/// change a type's "generics" field...  I think leaving it
/// alone should be okay 'cause it will be irrelevant
/// for further parts?
fn type_erase_type(typ: Type) -> Type {
    let t = match typ {
        Type::Generic(_) => Type::Prim(PrimType::AnyPtr),
        other => other,
    };
    t
}

fn type_erase_expr(expr: ExprNode, tck: &mut typeck::Tck) -> ExprNode {
    let expr_typeid = tck.get_expr_type(&expr);
    let expr_type = tck.reconstruct(expr_typeid).expect("Should never happen");
    // What expr's do we actually have to change?
    // Just ones that mention types explicitly.
    // Lambda's will be lifted,
    let new_contents = match (*expr.e).clone() {
        E::Let {
            varname,
            typename: Some(tn),
            init,
            mutable,
        } => {
            let new_type = type_map(tn, &mut type_erase_type);
            E::Let {
                varname,
                typename: Some(new_type),
                init,
                mutable,
            }
        }
        E::Funcall {
            func,
            params,
            type_params,
        } => {
            let new_types = type_params
                .iter()
                .map(|t| type_map(t.clone(), &mut type_erase_type))
                .collect();
            E::Funcall {
                func,
                params,
                type_params: new_types,
            }
        }
        E::TypeCtor {
            name,
            type_params,
            body,
        } => {
            let new_types = type_params
                .iter()
                .map(|t| type_map(t.clone(), &mut type_erase_type))
                .collect();
            E::TypeCtor {
                name,
                type_params: new_types,
                body,
            }
        }
        other => other,
    };
    // Change the type of the struct literal expr node to the new tuple
    // We need to do this to any expression because its type may be a struct.
    // And we need to do this after the above because we've changed the
    let new_t = type_map(expr_type.clone(), &mut type_erase_type);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::passes::*;
    use crate::*;

    #[test]
    fn test_basic() {
        let t = Type::Generic(Sym::new("foo"));
        let t2 = type_map(t.clone(), &mut |t| type_erase_type(t));
        assert_ne!(t, t2);
    }

    #[test]
    fn test_tuple_recursive() {
        let t = Type::tuple(vec![Type::i32(), Type::Generic(Sym::new("Foo"))]);
        let t2 = type_map(t.clone(), &mut |t| type_erase_type(t));

        let expected = Type::tuple(vec![Type::i32(), Type::Prim(PrimType::AnyPtr)]);
        assert_ne!(t, t2);
        assert_eq!(t2, expected);
    }

    #[test]
    fn test_func_recursive() {
        let g1 = Type::Generic(Sym::new("Foo"));
        let g2 = Type::Generic(Sym::new("Bar"));
        let t = Type::Func(
            vec![Type::i32(), g1.clone(), g2.clone()],
            Box::new(Type::tuple(vec![g1.clone()])),
            vec![g1, g2],
        );
        let t2 = type_map(t.clone(), &mut |t| type_erase_type(t));

        let expected = Type::Func(
            vec![Type::i32(), Type::anyptr(), Type::anyptr()],
            Box::new(Type::tuple(vec![Type::anyptr()])),
            vec![Type::anyptr(), Type::anyptr()],
        );
        assert_ne!(t, t2);
        assert_eq!(t2, expected);
    }

    #[test]
    fn test_func_decl() {
        let g1 = Type::Generic(Sym::new("Foo"));
        let sig = ast::Signature {
            params: vec![(Sym::new("x"), g1.clone())],
            rettype: g1.clone(),
            typeparams: vec![g1.clone()],
        };
        // fn foo(x: @Foo | @Foo) @Foo
        let d = D::Function {
            name: Sym::new("test"),
            signature: sig,
            body: vec![],
        };

        let typeerase_expr = &mut |e| e;
        let res = decl_map(d.clone(), typeerase_expr, &mut type_erase_type);
        let expected_sig = ast::Signature {
            params: vec![(Sym::new("x"), Type::anyptr())],
            rettype: Type::anyptr(),
            typeparams: vec![Type::anyptr()],
        };
        // fn foo(x: AnyPtr | AnyPtr) AnyPtr
        let expected_d = D::Function {
            name: Sym::new("test"),
            signature: expected_sig,
            body: vec![],
        };

        assert_eq!(res, expected_d);
    }
}
