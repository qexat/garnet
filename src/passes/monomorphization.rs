//!  Monomorphization

use std::collections::{BTreeMap, BTreeSet};

use crate::passes::*;
use crate::*;

/// A map from function name to all the
/// specializations we have found for it.
/// Operates as a set, so adding the same
/// specializations for it multiple times is a no-op.
#[derive(Default, Debug)]
struct FunctionSpecs {
    specializations: BTreeMap<Sym, BTreeSet<BTreeMap<Sym, Type>>>,
}

impl FunctionSpecs {
    fn insert(&mut self, s: Sym, substs: BTreeMap<Sym, Type>) {
        self.specializations.entry(s).or_default().insert(substs);
    }
}
/// Take any type annotations with generics in them and
/// replaces them with different (hopefully concrete) types.
///
/// This generates new, non-typechecked expressions, since we've
/// changed their types!
fn subst_expr(expr: ExprNode, substs: &BTreeMap<Sym, Type>) -> ExprNode {
    let e = *expr.e;
    let ne = ExprNode::new(e.clone());

    match e {
        // Nodes with no recursive expressions.
        // I list all these out explicitly instead of
        // using a catchall so it doesn't fuck up if we change
        // the type of Expr.
        E::Lit { .. } => ne,
        E::Var { .. } => ne,
        E::Break => ne,
        E::EnumCtor { .. } => ne,
        E::Let {
            varname,
            typename,
            init,
            mutable,
        } => {
            if let Some(Type::Generic(nm)) = typename {
                let new_t = substs.get(&nm).unwrap();
                ExprNode::new(E::Let {
                    varname,
                    typename: Some(new_t.clone()),
                    init,
                    mutable,
                })
            } else {
                ne
            }
        }
        _other => todo!(),
    }
}

fn monomorphize_expr(
    expr: ExprNode,
    tck: &mut typeck::Tck,
    functioncalls: &mut FunctionSpecs,
) -> ExprNode {
    let mut new_expr = |e| match &e {
        E::Funcall {
            func,
            params,
            type_params,
        } => {
            match &*func.e {
                E::Var { name } => {
                    // Get the type we know of the function expression
                    let ftypeid = tck.get_expr_type(func);
                    let ftype = tck
                        .reconstruct(ftypeid)
                        .expect("Should never fail, 'cause typechecking succeeded if we got here");
                    // If the function in question has no generics,
                    // bail early.
                    if ftype.collect_generic_names().len() == 0 {
                        return e;
                    }

                    // Figure out what types the function has been called with
                    let param_types: Vec<_> = params
                        .iter()
                        .map(|p| tck.reconstruct(tck.get_expr_type(p)).unwrap())
                        .collect();
                    // Get this expr's return type
                    let rettype = tck.reconstruct(tck.get_expr_type(&expr)).unwrap();
                    let called_type = Type::Func(param_types, Box::new(rettype));
                    let substitutions = &mut Default::default();
                    // Do our substitution of generics to real types
                    let _concrete_type = ftype.substitute(&called_type, substitutions);
                    functioncalls.insert(*name, substitutions.clone());

                    // Ok we replace the var being called with a reference to
                    // the substituted function's name
                    let new_name = mangle_generic_name(*name, &substitutions);
                    let new_call = func.clone().map(&mut |_func| E::Var { name: new_name });
                    E::Funcall {
                        func: new_call,
                        params: params.clone(),
                        type_params: type_params.clone(),
                    }
                }
                // what the shit how the fuck do I tell if this is a local var or not
                _ => todo!("How the hell do we know what function to monomorphize here???"),
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
fn mangle_generic_name(s: Sym, substs: &BTreeMap<Sym, Type>) -> Sym {
    let mut new_str = format!("__mono_{}", s);
    for (nm, ty) in substs.iter() {
        new_str += "__";
        new_str += &*nm.val();
        new_str += "_";
        new_str += &generate_type_name(ty);
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
    let functioncalls: &mut FunctionSpecs = &mut Default::default();
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
    // Now we actually create the new functions
    for (nm, specs) in &functioncalls.specializations {
        trace!("Specializing {}", nm);
        dbg!(nm);
        let old_func = new_decls
            .iter()
            .find(|d| matches!(d, D::Function { name, .. } if name == nm))
            .unwrap()
            .clone();
        for subst in specs {
            let mangled_name = mangle_generic_name(*nm, subst);
            trace!("  Specialized name: {}", mangled_name);
            match &old_func {
                D::Function {
                    name: _,
                    signature: old_sig,
                    body: old_body,
                } => {
                    let sig_type = old_sig.to_type();
                    let new_sig_type = sig_type.apply_substitutions(subst);
                    let new_sig = old_sig.map_type(&new_sig_type);
                    let new_body = exprs_map(old_body.clone(), &mut |e| subst_expr(e, subst));
                    let new_decl = D::Function {
                        name: mangled_name,
                        signature: new_sig,
                        // I don't think we have to actually change
                        // any of the contents...
                        // Maybe we do if there's type annotations?
                        // ...yep, yep we totally do.
                        //
                        // We shouldn't have to typecheck though, since
                        // we know we're generating valid code.
                        // Nope, nope, wait, we totally do, because
                        // our codegen relies on having types for everything.
                        body: new_body,
                    };
                    new_decls.push(new_decl);
                }
                _ => unreachable!(),
            }
        }
    }
    // TODO: Typecheck whole program with new Tck, I suppose.
    let new_ir = Ir { decls: new_decls };
    let new_tck =
        typeck::typecheck(&new_ir).expect("Generated monomorphized IR that doesn't typecheck");
    *tck = new_tck;
    new_ir
}
