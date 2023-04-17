//!  Monomorphization

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::passes::*;
use crate::*;

type Specialization = (Sym, BTreeMap<Sym, Type>);

/// A map from function name to all the
/// specializations we have found for it.
/// Operates as a set, so adding the same
/// specializations for it multiple times is a no-op.
#[derive(Default, Debug)]
struct FunctionSpecs {
    specializations: BTreeMap<Sym, BTreeSet<BTreeMap<Sym, Type>>>,

    specs: BTreeSet<Specialization>,
}

impl FunctionSpecs {
    fn insert(&mut self, s: Sym, substs: BTreeMap<Sym, Type>) {
        self.specializations
            .entry(s)
            .or_default()
            .insert(substs.clone());
        self.specs.insert((s, substs));
    }

    fn exists(&self, s: &Specialization) -> bool {
        self.specs.contains(s)
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
        // We only care about nodes that can have a type named
        // in them somewhere.
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
        E::Funcall {
            func,
            params,
            type_params,
        } => {
            let new_type_params = type_params
                .into_iter()
                .map(|(nm, t)| (nm, t.apply_substs(substs)))
                .collect();
            ExprNode::new(E::Funcall {
                func,
                params,
                type_params: new_type_params,
            })
        }
        E::TypeCtor {
            name,
            type_params,
            body,
        } => {
            let new_type_params = type_params
                .into_iter()
                .map(|t| t.apply_substs(substs))
                .collect();
            ExprNode::new(E::TypeCtor {
                name,
                type_params: new_type_params,
                body,
            })
        }
        E::Lambda { .. } => unreachable!("These have all been lambda-lifted away"),
        _other => ne,
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
            match &*func.e {
                E::Var { name } => {
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
                    info!(
                        "Finding substitutions to turn {:?} into {:?}",
                        ftype, &called_type
                    );
                    ftype.find_substs(&called_type, substitutions);
                    trace!("Subst for {} is {:?}", name, substitutions);
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

fn monomorphize2_expr(
    expr: ExprNode,
    tck: &mut typeck::Tck,
    functions: &BTreeMap<Sym, hir::Decl>,
    specs: &mut FunctionSpecs,
    worklist: &mut VecDeque<Specialization>,
) -> ExprNode {
    let mut new_expr = |e| match &e {
        // After lambda-lifting,
        // All function calls will end up with a variable name somewhere.
        // So to find the types of functions that are called we just look up the
        // types of these variables.
        E::Var { name } => {
            let expr_type = tck.reconstruct(tck.get_expr_type(&expr)).unwrap();
            if matches!(&expr_type, Type::Func(_, _)) {
                // Lookup the generic func
                let generic_function = functions.get(name).unwrap();
                let generic_function_sig = match generic_function {
                    D::Function { signature, .. } => signature,
                    _ => unreachable!(),
                };
                let generic_function_type = generic_function_sig.to_type();
                if generic_function_type.collect_generic_names().len() == 0 {
                    // Not calling a generic function, bail with expression
                    // unchanged.
                    return e;
                }
                info!(
                    "Finding substitutions to turn {:?} into {:?}",
                    generic_function_type, expr_type
                );
                let mut substitutions = Default::default();
                generic_function_type.find_substs(&expr_type, &mut substitutions);
                // Ok we register this as a new substitution that needs to
                // be instantiated.
                let new_name = mangle_generic_name(*name, &substitutions);
                worklist.push_back((*name, substitutions));

                // And we replace the var with one containing
                // the substituted function
                E::Var { name: new_name }
            } else {
                e
            }
        }
        _ => e,
    };
    /*
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
                        info!(
                            "Finding substitutions to turn {:?} into {:?}",
                            ftype, &called_type
                        );
                        ftype.find_substs(&called_type, substitutions);
                        trace!("Subst for {} is {:?}", name, substitutions);
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
    */
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
///
/// ...ok, this is gonna have to get a little more radical, 'cause
/// if we have a generic function foo(A, B) and it is called from
/// another generic function bar(B) like foo(true, B), and then bar()
/// is called like bar(some_concrete_type), then we have to chain
/// the specializations down from the source.  
/// So to do this I suppose we just have to walk the entire program
/// in order.
pub(super) fn monomorphize(ir: Ir, tck: &mut typeck::Tck) -> Ir {
    let specializations: &mut FunctionSpecs = &mut Default::default();
    // A function and type param substitutions
    type Specialization = (Sym, BTreeMap<Sym, Type>);
    // The specializations we have yet to deal with.
    // We may add things to this as we walk our program and discover
    // new things that need instantiation.
    let worklist: &mut VecDeque<Specialization> = &mut VecDeque::new();
    // We put finished specializations here so we can quickly check
    // whether they are duplicates.
    let donelist: &mut BTreeSet<Specialization> = &mut Default::default();

    let new_functions: &mut Vec<D> = &mut vec![];

    // A map from name to decl for all of our functions, so we can
    // do random access lookups more easily to walk the call tree.
    let functions: &mut BTreeMap<Sym, hir::Decl> = &mut Default::default();
    for decl in ir.decls.iter() {
        match decl {
            D::Function {
                name,
                signature: _,
                body: _,
            } => {
                functions.insert(*name, decl.clone());
            }
            _ => (),
        }
    }

    // We start from the entry point and just walk the call tree.
    worklist.push_front((Sym::new("main"), BTreeMap::default()));
    while !worklist.is_empty() {
        // Get a function to work on and grab its definition
        let current_subst = worklist.pop_back().unwrap();
        // Make sure this isn't a subst that already exists.
        // We just fill the worklist up with all the functions and then
        // skip duplicates, rather than checking for duplicates when inserting
        // new work items.  Not sure which is better, so just sticking with this.
        if donelist.contains(&current_subst) {
            continue;
        }

        let (current_name, current_substs) = current_subst;
        let current_func = functions.get(&current_name).unwrap();
        // Ok now we go through the body of the function and, for each
        // function variable it contains:
        //  * Check if its definition is generic.  If so,
        //  * Get the arg types and construct a substitution for it
        //  * Check if the instantiation has already been done
        //  * If not, add it to the worklist
        match current_func {
            D::Function {
                name,
                signature,
                body,
            } => {
                let substed_body = exprs_map(body.clone(), &mut |e| {
                    monomorphize2_expr(e, tck, functions, specializations, worklist)
                });
            }
            _ => unreachable!(),
        }

        donelist.insert((current_name, current_substs));
    }

    let mut new_decls = vec![];
    for decl in ir.decls.into_iter() {
        let res = match decl.clone() {
            D::Function {
                name,
                signature,
                body,
            } => {
                //let sigtype = signature.to_type();
                //let sig_generics = sigtype.collect_generic_names();
                let new_body = exprs_map(body, &mut |e| monomorphize_expr(e, tck, specializations));
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
            D::Import { .. } => todo!(),
        };
        new_decls.push(res);
    }
    // Now we actually create the new functions
    for (nm, specs) in &specializations.specializations {
        trace!("Specializing {}", nm);
        let old_func = new_decls
            .iter()
            .find(|d| matches!(d, D::Function { name, .. } if name == nm))
            .expect("Can't happen, this function name had to come from somewhere.")
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
                    let new_sig_type = sig_type.apply_substs(subst);
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
    // Typecheck whole program with new Tck, I suppose.
    // This feels slightly insane but works for the moment.
    // We need the type info for the code we've generated
    // so that we can output it correctly.
    let new_ir = Ir { decls: new_decls };
    let new_tck =
        typeck::typecheck(&new_ir).expect("Generated monomorphized IR that doesn't typecheck");
    *tck = new_tck;
    new_ir
}
