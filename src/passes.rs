//! Transformation/optimization passes that function on the IR.
//! May function on typed or untyped IR, either way.
//! For now each is a function from `IR -> IR`, rather than
//! having a visitor and mutating stuff or anything like that,
//! which may be less efficient but is IMO simpler to think about.
//!
//! TODO: A pass that might be useful would be "alpha renaming".
//! Essentially you walk through your entire compilation unit and
//! rename everything to a globally unique name.  This means that
//! scope vanishes, for example, and anything potentially
//! ambiguous becomes unambiguous.

use crate::hir::{Decl as D, Expr as E, Ir, TypedExpr};
use crate::*;

type Pass = fn(Ir) -> Ir;

pub fn run_passes(ir: Ir) -> Ir {
    let passes: &[Pass] = &[lambda_lifting];
    passes.iter().fold(ir, |prev_ir, f| f(prev_ir))
}

/// Lambda lift a single expr.
fn lambda_lift_expr(expr: TypedExpr, output_funcs: &mut Vec<D>) -> TypedExpr {
    let result = match expr.e {
        E::BinOp { op, lhs, rhs } => {
            let nlhs = lambda_lift_expr(*lhs, output_funcs);
            let nrhs = lambda_lift_expr(*rhs, output_funcs);
            E::BinOp {
                op,
                lhs: Box::new(nlhs),
                rhs: Box::new(nrhs),
            }
        }
        E::UniOp { op, rhs } => {
            let nrhs = lambda_lift_expr(*rhs, output_funcs);
            E::UniOp {
                op,
                rhs: Box::new(nrhs),
            }
        }
        E::Block { body } => E::Block {
            body: lambda_lift_exprs(body, output_funcs),
        },
        E::Let {
            varname,
            typename,
            init,
            mutable,
        } => E::Let {
            varname,
            typename,
            init: Box::new(lambda_lift_expr(*init, output_funcs)),
            mutable,
        },
        E::If { cases } => {
            let new_cases = cases
                .into_iter()
                .map(|(test, case)| {
                    let new_test = lambda_lift_expr(test, output_funcs);
                    let new_cases = lambda_lift_exprs(case, output_funcs);
                    (new_test, new_cases)
                })
                .collect();
            E::If { cases: new_cases }
        }

        E::Loop { body } => E::Loop {
            body: lambda_lift_exprs(body, output_funcs),
        },
        E::Return { retval } => E::Return {
            retval: Box::new(lambda_lift_expr(*retval, output_funcs)),
        },
        E::Funcall {
            func,
            params,
            generic_types,
        } => {
            let new_func = Box::new(lambda_lift_expr(*func, output_funcs));
            let new_params = lambda_lift_exprs(params, output_funcs);
            let new_generics = generic_types.to_vec();
            E::Funcall {
                func: new_func,
                params: new_params,
                generic_types: new_generics,
            }
        }
        E::Lambda { signature, body } => {
            // This is actually the only important bit.
            // TODO: Make a more informative name, maybe including the file and line number or
            // such.
            let lambda_name = INT.gensym("lambda");
            let function_decl = D::Function {
                name: lambda_name,
                signature,
                body: lambda_lift_exprs(body, output_funcs),
            };
            output_funcs.push(function_decl);
            E::Var { name: lambda_name }
        }
        x => x,
    };
    hir::TypedExpr {
        e: result,
        id: expr.id,
    }
}

/// Lambda lift a list of expr's.
fn lambda_lift_exprs(exprs: Vec<TypedExpr>, output_funcs: &mut Vec<D>) -> Vec<TypedExpr> {
    exprs
        .into_iter()
        .map(|e| lambda_lift_expr(e, output_funcs))
        .collect()
}

/// A transformation pass that removes lambda expressions and turns
/// them into function decl's.
///
/// TODO: Output is an IR that does not have any lambda expr's in it, which
/// I would like to make un-representable, but don't see a good way
/// to do yet.  Add a tag type of some kind to the Ir?
fn lambda_lifting(ir: Ir) -> Ir {
    let mut new_functions = vec![];
    let new_decls: Vec<D> = ir
        .decls
        .into_iter()
        .map(|decl| match decl {
            D::Function {
                name,
                signature,
                body,
            } => {
                let new_body = lambda_lift_exprs(body, &mut new_functions);
                D::Function {
                    name,
                    signature,
                    body: new_body,
                }
            }
            x => x,
        })
        .collect();
    new_functions.extend(new_decls.into_iter());
    Ir {
        decls: new_functions,
    }
}

fn _enum_to_int_expr(_expr: TypedExpr, _output_funcs: &mut Vec<D>) -> TypedExpr {
    todo!()
}

fn _enum_to_int(ir: Ir) -> Ir {
    let mut new_functions = vec![];
    let new_decls: Vec<D> = ir
        .decls
        .into_iter()
        .map(|decl| match decl {
            D::Function {
                name,
                signature,
                body,
            } => {
                let new_body = body
                    .into_iter()
                    .map(|e| _enum_to_int_expr(e, &mut new_functions))
                    .collect();
                D::Function {
                    name,
                    signature,
                    body: new_body,
                }
            }
            x => x,
        })
        .collect();
    new_functions.extend(new_decls.into_iter());
    Ir {
        decls: new_functions,
    }
}

/// Takes an IR containing compound types (currently just tuples)
/// treated as value types, and turns it into one containing
/// only reference types -- ie, anything with subdividable
/// fields and/or bigger than a machine register (effectively
/// 64-bits for wasm) is only referred to through a pointer.
///
/// We might be able to get rid of TupleRef's by turning
/// them into pointer arithmatic, too.
fn _pointerification(_ir: Ir) -> Ir {
    todo!()
}
