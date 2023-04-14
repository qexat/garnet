//! Struct to tuples.
//!
//! Takes all structs and turns them into tuples.  Basically
//! so that we don't need to generate names for anonymous structs
//! and juggle which ones we do and don't know about.
//! Not sure if this is the best approach for that, but it
//! seems to work fine so far.  IIRC doing it the other way around
//! caused issues with generating type constructors and such
//! in Rust.

use crate::passes::*;
use crate::*;

/// Takes a struct and a field name and returns what tuple
/// offset the field should correspond to.
fn offset_of_field(fields: &BTreeMap<Sym, Type>, name: Sym) -> usize {
    // This is one of those times where an iterator chain is
    // way stupider than just a normal for loop, alas.
    // TODO someday: make this not O(n), or at least make it
    // cache results.
    for (i, (nm, _ty)) in fields.iter().enumerate() {
        if *nm == name {
            return i;
        }
    }
    panic!(
        "Invalid struct name {} in struct got through typechecking, should never happen",
        &*name.val()
    );
}

/// Takes an arbitrary type and recursively turns
/// structs into tuples.
fn tuplize_type(typ: Type) -> Type {
    match typ {
        Type::Struct(fields, _generics) => {
            // This is why structs contain a BTreeMap,
            // so that this ordering is always consistent based
            // on the field names.

            // TODO: What do we do with generics?  Anything?
            let tuple_fields = fields
                .iter()
                .map(|(_sym, ty)| type_map(ty.clone(), &mut tuplize_type))
                .collect();
            Type::tuple(tuple_fields)
        }
        other => other,
    }
}

/// Takes an arbitrary expr and if it is a struct ctor
/// or ref turn it into a tuple ctor or ref.
///
/// TODO: Do something with the expr types?
fn tuplize_expr(expr: ExprNode, tck: &mut typeck::Tck) -> ExprNode {
    trace!("Tuplizing expr {:?}", expr);
    let expr_typeid = tck.get_expr_type(&expr);
    let struct_type = tck.reconstruct(expr_typeid).expect("Should never happen?");
    let new_contents = match (*expr.e).clone() {
        E::Let {
            varname,
            typename: Some(t),
            init,
            mutable,
        } => {
            let new_type = tuplize_type(t.clone());
            E::Let {
                varname,
                init,
                mutable,
                typename: Some(new_type),
            }
        }
        E::StructCtor { body } => match &struct_type {
            // TODO: Generics?
            Type::Struct(type_body, _generics) => {
                let mut ordered_body: Vec<_> = body
                    .into_iter()
                    .map(|(ky, vl)| (offset_of_field(&type_body, ky), vl))
                    .collect();
                ordered_body.sort_by(|a, b| a.0.cmp(&b.0));
                let new_body = ordered_body
                    .into_iter()
                    .map(|(_i, expr)| expr.clone())
                    .collect();

                E::TupleCtor { body: new_body }
            }
            _other => {
                unreachable!("StructCtor expression type is not a struct, should never happen")
            }
        },
        E::StructRef {
            expr: inner_expr,
            elt,
        } => {
            let inner_typeid = tck.get_expr_type(&inner_expr);
            let struct_type = tck.reconstruct(inner_typeid).expect("Should never happen?");
            match struct_type {
                Type::Struct(type_body, _generics) => {
                    let offset = offset_of_field(&type_body, elt);
                    E::TupleRef {
                        expr: inner_expr.clone(),
                        elt: offset,
                    }
                }
                other => unreachable!(
                    "StructRef passed type checking but expr type is {:?}.  Should never happen!",
                    other
                ),
            }
        }
        other => other,
    };
    // Change the type of the struct literal expr node to the new tuple
    // We need to do this to any expression because its type may be a struct.
    // And we need to do this after the above because we've changed the
    let new_t = tuplize_type(struct_type.clone());
    if new_t != struct_type {
        info!("Replaced struct type {:?} with {:?}", struct_type, new_t);
        trace!("Expression was rewritten into {:?}", new_contents);
    }
    tck.replace_expr_type(&expr, &new_t);
    expr.map(&mut |_| new_contents.clone())
}

/// Takes any struct types and replaces them with tuples.
/// This is necessary because we have anonymous structs but
/// Rust doesn't.
///
/// I WAS going to turn them into named structs but then declaring them gets
/// kinda squirrelly, because we don't really have a decl that translates
/// directly into a Rust struct without being wrapped in a typedef first.  So I
/// think I will translate them to tuples after all.
pub(super) fn struct_to_tuple(ir: Ir, tck: &mut typeck::Tck) -> Ir {
    let mut new_decls = vec![];
    let tuplize_expr = &mut |e| tuplize_expr(e, tck);

    for decl in ir.decls.into_iter() {
        let res = decl_map(decl, tuplize_expr, &mut tuplize_type);
        new_decls.push(res);
    }
    let new_ir = Ir { decls: new_decls };
    // TODO BUGGO: shiiiiit our replace_expr_type() call in tuplize_expr()
    // doesn't work correctly in all cases.
    // For example if we have a function call with a struct as an arg,
    // it will rewrite the struct's type correctly, but the type of the
    // function will not get rewritten.  I have *no idea* how to fix this
    // in the general case, it might involve having a version of
    // passes::expr_map that does a post-traversal instead of a pre-traversal?
    //
    // So for now we just throw away all type info and regenerate it!
    let new_tck =
        typeck::typecheck(&new_ir).expect("Generated monomorphized IR that doesn't typecheck");
    *tck = new_tck;
    new_ir
}

/* I'm just gonna stash the unused anonymous-to-named struct
code here in case I ever want it later.


//////  Turn all anonymous types into named ones //////


/// Whether or not something is an anonymous tuple or struct or such.
fn type_is_anonymous(typ: &Type) -> bool {
    match typ {
        Type::Prim(_) | Type::Named(_, _) | Type::Array(_, _) | Type::Generic(_) => false,
        _ => true,
    }
}

/// Takes a type and, if it is anonymous, generates a unique name for it
/// and inserts it into the given set of typedecls.
/// Returns the renamed type, or the old type if it doesn't need to be altered.
fn nameify_type(typ: Type, known_types: &mut BTreeMap<Sym, D>) -> Type {
    if type_is_anonymous(&typ) {
        let new_type_name = Sym::new(generate_type_name(&typ));
        let generic_names = typ.collect_generic_names();
        let generics: Vec<_> = generic_names.iter().map(|s| Type::Generic(*s)).collect();
        let named_type = Type::Named(new_type_name, generics.clone());
        // TODO: entry()
        if !known_types.contains_key(&new_type_name) {
            known_types.insert(
                new_type_name,
                D::TypeDef {
                    name: new_type_name,
                    typedecl: typ,
                    params: generic_names,
                },
            );
        }
        named_type
    } else {
        typ
    }
}

fn nameify_expr(
    expr: ExprNode,
    tck: &mut typeck::Tck,
    known_types: &mut BTreeMap<Sym, D>,
) -> ExprNode {
    let expr_typeid = tck.get_expr_type(&expr);
    let expr_type = tck.reconstruct(expr_typeid).expect("Should never happen?");
    let named_type = nameify_type(expr_type, known_types);
    tck.replace_expr_type(&expr, &named_type);
    expr
}

fn nameify(ir: Ir, tck: &mut typeck::Tck) -> Ir {
    let mut new_decls = vec![];
    let known_types = &mut BTreeMap::new();

    for decl in ir.decls.into_iter() {
        let res = match decl {
            D::Const {
                name,
                typ: typename,
                init,
            } => {
                // gramble gramble can't have two closures borrowing known_types at the same time
                let new_type = type_map(typename, &mut |t| nameify_type(t, known_types));
                let new_init = expr_map(init, &mut |e| nameify_expr(e, tck, known_types));
                D::Const {
                    name,
                    typ: new_type,
                    init: new_init,
                }
            }
            D::Function {
                name,
                signature,
                body,
            } => D::Function {
                name,
                signature: signature_map(signature, &mut |t| nameify_type(t, known_types)),
                body: exprs_map(body, &mut |e| nameify_expr(e, tck, known_types)),
            },
            D::TypeDef {
                name,
                params,
                typedecl,
            } => D::TypeDef {
                name,
                params,
                typedecl: type_map(typedecl, &mut |t| nameify_type(t, known_types)),
            },
        };
        // Can't do this 'cause borrowing known_types twice is no bueno
        // Sad.
        // let res = decl_map(decl, &mut |e| nameify_expr(e, tck, known_types), &mut |t| {
        //     nameify_type(t, known_types)
        // });
        new_decls.push(res);
    }
    new_decls.extend(known_types.into_iter().map(|(_k, v)| v.clone()));
    Ir { decls: new_decls }
}

*/

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;
    #[test]
    fn test_enumize() {
        let mut body = BTreeMap::new();
        body.insert(Sym::new("foo"), Type::i32());
        body.insert(Sym::new("bar"), Type::i64());

        let desired = tuplize_type(Type::Struct(body.clone(), vec![]));
        let inp = Type::Struct(body, vec![]);
        let out = type_map(inp.clone(), &mut tuplize_type);
        assert_eq!(out, desired);

        let desired2 = Type::Array(Box::new(out.clone()), 3);
        let inp2 = Type::Array(Box::new(inp.clone()), 3);
        let out2 = type_map(inp2.clone(), &mut tuplize_type);
        assert_eq!(out2, desired2);
    }
}
