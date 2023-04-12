//!  Monomorphization

use crate::passes::*;
use crate::*;

fn monomorphize_expr(
    expr: ExprNode,
    tck: &mut typeck::Tck,
    functioncalls: &mut BTreeSet<(Sym, Vec<Type>)>,
) -> ExprNode {
    let mut new_expr = |e| match &e {
        E::Funcall {
            func,
            params: _,
            type_params,
        } => {
            // Figure out what types the function has been called with
            let ftypeid = tck.get_expr_type(&expr);
            let ftype = tck
                .reconstruct(ftypeid)
                .expect("Should never fail, 'cause typechecking succeeded if we got here");
            dbg!(&ftype);
            let generics = ftype.collect_generic_names();
            dbg!(&generics);
            if generics.len() > 0 {
                match &*func.e {
                    E::Var { name } => {
                        let substs: Vec<_> = type_params.values().cloned().collect();
                        let new_func_name = mangle_generic_name(*name, &substs);
                        functioncalls.insert((*name, substs.clone()));
                        E::Var { name: new_func_name }
                    },
                    _ => unreachable!("Function calls should always be named 'cause we've done lambda lifting, right?   Right?  Nope, but we gotta start somewhere.")
                }
            } else {
                // No generics in the function call, nothing to monomorphize
                // TODO: Sort your shit out with type_params
                e
            }
        }
        _ => e,
    };
    expr.clone().map(&mut new_expr)
}

/// takes a symbol S and type params listed and generates a new symbol
/// for it.
/// Someday we should probably make a real name mangling scheme that
/// we have actually thought about.
fn mangle_generic_name(s: Sym, tys: &[Type]) -> Sym {
    let mut new_str = format!("__mono_{}_", s);
    for t in tys {
        new_str += &generate_type_name(t);
        new_str += "_";
    }
    Sym::new(new_str)
}

/// Ok, so what we have to do here is this:
/// FIRST, we scan through the entire program and find
/// all functions with generic params that are called,
/// and figure out what their generics are.  (The expr's
/// type should have this info.)
///
/// Then we have to go through and find where each of those functions
/// is defined, which should be at the toplevel since we've done
/// lambda-lifting.  We make a copy of them with the types
/// substituted, with a new name.
///
/// Then we have to rewrite the function call names to point at the monomorphized
/// functions.  We can do this while collecting them, since we generate the
/// names from the signature.
///
/// That probably won't get us *all* of the way there, we still will have
/// generic type constructors in structs and stuff.  But it's a start.
///
/// ...ok so what do we do if we have:
/// foo.baz = some_func
/// foo.baz()
/// We need to change the `some_func`, not the `foo.baz()`.
/// So...  every function call has some list of type substitutions
/// stuck into it.  So we don't monomorphize just function calls, we have
/// to do it to every function object, with the types we figure out at
/// the call...  Our type-checking should have that info, does it?
pub(super) fn monomorphize(ir: Ir, tck: &mut typeck::Tck) -> Ir {
    let functioncalls: &mut BTreeSet<(Sym, Vec<Type>)> = &mut Default::default();
    let mut functions_to_mono: BTreeSet<Sym> = Default::default();

    let mut new_decls = vec![];
    for decl in ir.decls.into_iter() {
        let res = match decl.clone() {
            D::Function {
                name,
                signature,
                body,
            } => {
                let sigtype = signature.to_type();
                let sig_generics = sigtype.collect_generic_names();
                if sig_generics.len() > 0 {
                    // This function is a candidate for monomorph
                    functions_to_mono.insert(name);
                }
                let new_body = exprs_map(body, &mut |e| monomorphize_expr(e, tck, functioncalls));
                D::Function {
                    name,
                    signature,
                    body: new_body,
                }
            }
            D::Const {
                name: _,
                typ: _,
                init: _,
            } => {
                decl

                // TODO:
                //unimplemented!("We cannot monomorphize a const expr!");
                /*
                let generics = typ.collect_generic_names();
                if generics.len() > 0 {
                    D::Const {
                        name,
                        typ,
                        init: monomorphize_expr(init, tck),
                    }
                } else {
                    D::Const {
                        name,
                        typ,
                        init: init,
                    }
                }
                */
            }
            // No need to touch anything here, huzzah
            D::TypeDef { .. } => decl,
        };
        new_decls.push(res);
    }
    Ir { decls: new_decls }
}
