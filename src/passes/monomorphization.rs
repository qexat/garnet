//! Monomorphization
//!
//! From the Austral compiler docs:
//!
//! Monomorphization of generic types works recursively from the bottom up. To transform a type to a monomorphic type, [walk the expression tree and for each type encountered]:
//!
//!  * If encountering a type with no type arguments, leave it alone.
//!  * If encountering a generic type with (monomorphic) type arguments applied to it, retrieve or add a monomorph for the given type and arguments, and replace this type with the monomorph.
//!
//! Monomorphization of functions is analogous: starting from the main function (which has no generic type parameters), recur down the body.  If we encounter a call to a generic function f, look at the concrete type arguments the function is being called with. These define a mapping from the function’s type parameters ... to the type arguments [variables]...
//!

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::passes::*;
use crate::*;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Instance {
    ty: typeck::TypeId,
    referenced_from: hir::Eid,
}

#[derive(Default, Debug)]
struct MonoCtx {
    instances: BTreeSet<Instance>,
}

impl MonoCtx {}

/// Walk up every expression tree from leaf to root and, when encountering a call
/// to a generic function, look at the concrete type arguments the function is being
/// called with.  
fn mono_expr(expr: ExprNode, tck: &typeck::Tck) -> ExprNode {
    let f = &mut |e| match &e {
        E::Funcall {
            func,
            params,
            type_params,
        } => {
            // Get the type of `func` and if it has generic parameters that are concrete, create a monomorph for it.
            let fn_typevar = tck.get_expr_type(&func);
            let concrete_type = tck.reconstruct(fn_typevar).unwrap();
            trace!(
                "mono_expr function call:\n  func {}\n  type {}\n  params {:?}\n  type_params {:?}",
                func,
                concrete_type,
                params,
                type_params
            );
            let declared_type_params = concrete_type.get_toplevel_type_params();
            let _given_type_params = type_params;
            // Hmmmm, this needs to be where we actually *use* Tck.instances I think
            if declared_type_params.is_empty() {
                // Function is not polymorphic, we don't need to do anything.
                trace!("Not polymorphic, don't need to do anything");
                return e;
            }
            // Function is polymorphic
            trace!("Need to monomorph {}", func);
            // Get instantiated type of this particular expression
            let instance = tck.instances_rev.get(&func.id).unwrap();
            trace!(
                "instantiating type {}",
                tck.reconstruct(*instance).unwrap().get_name()
            );

            e
        }
        E::Lambda { .. } => {
            panic!("Should never happen, lambda-lifting has to happen first")
        }
        other => other.clone(),
    };
    expr.map(f)
}

/// Takes the given function decl and, for all functions it calls, generate
/// instances for them as necessary.  
///
/// I think this implies that the current function *must* not have type
/// parameters, but since `main()` doesn't have type params and we start
/// from there we basically always ensure this invariant is upheld.
fn mono_function_decl(tck: &typeck::Tck, nm: Sym, sig: &Signature, body: &[ExprNode]) {
    assert!(
        sig.typeparams.is_empty(),
        "Tried to monomorphize function that isn't itself monomorphic yet: {} {:?}",
        nm,
        sig
    );
    let f = &mut |e| mono_expr(e, tck);
    let _ = exprs_map_post(body.to_vec(), f);
}

/// I want to be able to look up functions easily by name, so
/// we'll just do things this way.  We've played with that a bit
/// in the past and ended up the list format the IR currently uses,
/// but I guess we'll just make both and translate as needed for
/// whatever we need.
#[derive(Debug, Clone, Default)]
struct SplitIr {
    /// signature, body
    functions: BTreeMap<Sym, (Signature, Vec<ExprNode>)>,
    /// type, init val
    consts: BTreeMap<Sym, (Type, ExprNode)>,
    /// params, typedecl
    typedefs: BTreeMap<Sym, (Vec<Sym>, Type)>,
    /// name, localname
    imports: BTreeMap<Sym, Sym>,
    filename: String,
    modulename: String,
}

impl From<&Ir> for SplitIr {
    fn from(ir: &hir::Ir) -> Self {
        let mut sir = SplitIr {
            filename: ir.filename.clone(),
            modulename: ir.modulename.clone(),
            ..SplitIr::default()
        };
        for decl in ir.decls.iter().cloned() {
            use hir::Decl::*;
            match decl {
                Function {
                    name,
                    signature,
                    body,
                } => {
                    let _ = sir.functions.insert(name, (signature, body));
                }
                Const { name, typ, init } => {
                    let _ = sir.consts.insert(name, (typ, init));
                }
                TypeDef {
                    name,
                    params,
                    typedecl,
                } => {
                    let _ = sir.typedefs.insert(name, (params, typedecl));
                }
                Import { name, localname } => {
                    let _ = sir.imports.insert(name, localname);
                }
            }
        }

        sir
    }
}

impl From<&SplitIr> for Ir {
    fn from(sir: &SplitIr) -> Self {
        let mut ir = Ir {
            filename: sir.filename.clone(),
            modulename: sir.modulename.clone(),
            ..Ir::default()
        };
        for (nm, (sig, body)) in &sir.functions {
            let d = hir::Decl::Function {
                name: *nm,
                signature: sig.clone(),
                body: body.clone(),
            };
            ir.decls.push(d);
        }
        for (nm, (typ, init)) in &sir.consts {
            let d = hir::Decl::Const {
                name: *nm,
                typ: typ.clone(),
                init: init.clone(),
            };
            ir.decls.push(d);
        }

        for (nm, (params, typedecl)) in &sir.typedefs {
            let d = hir::Decl::TypeDef {
                name: *nm,
                params: params.clone(),
                typedecl: typedecl.clone(),
            };
            ir.decls.push(d);
        }

        for (nm, localname) in &sir.imports {
            let d = hir::Decl::Import {
                name: *nm,
                localname: *localname,
            };
            ir.decls.push(d);
        }

        ir
    }
}

/// So to start off, we just walk the AST and, from bottom to top, find all the places a
/// generic type is instantiated or a generic function is called with type params.
pub(super) fn monomorphize(ir: Ir, _symtbl: &symtbl::Symtbl, tck: &mut typeck::Tck) -> Ir {
    // let type_map = &mut |t| t;
    //let expr_map = &mut |e| mono_expr(e, tck);

    // hmmm, we really need to start from `main()` and walk to the functions it calls.
    let sir = SplitIr::from(&ir);
    // We know the main function exists and has typechecked correctly
    let entry = sir
        .functions
        .get(&Sym::new("main"))
        .expect("No main function found, should never happen");
    // Do a depth-first search through call graph
    let mut functions_to_mono = VecDeque::new();
    functions_to_mono.push_back((Sym::new("main"), entry));
    let mut functions_monoed: BTreeSet<Sym> = BTreeSet::new();
    loop {
        // Get next function to mono
        if let Some((nm, (sig, body))) = functions_to_mono.pop_front() {
            if functions_monoed.contains(&nm) {
                // already did that func, carry on to next
                continue;
            }
            // *do magic here*
            mono_function_decl(tck, nm, sig, body);

            functions_monoed.insert(nm);
        } else {
            // no work left to do
            break;
        }
    }
    // let new_decls = ir
    //     .decls
    //     .into_iter()
    //     .map(|d| decl_map_post(d, expr_map, type_map))
    //     .collect();
    Ir {
        // decls: new_decls,
        ..ir
    }
}

#[cfg(test)]
mod tests {
    // use super::*;
    // use crate::hir::*;
}
