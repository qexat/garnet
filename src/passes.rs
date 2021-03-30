//! Transformation/optimization passes that function on the IR.
//! May function on typed or untyped IR, either way.
//! For now each is a function from `IR<T> -> IR<T>`, rather than
//! having a visitor and mutating stuff or anything like that,
//! which may be less efficient but is IMO simpler to think about.

use crate::hir::{plz, Decl as D, Expr as E, Ir, TypedExpr};
use crate::*;

type Pass<T> = fn(Ir<T>) -> Ir<T>;

pub fn run_passes(ir: Ir<()>) -> Ir<()> {
    let passes: &[Pass<()>] = &[lambda_lifting];
    passes.iter().fold(ir, |prev_ir, f| f(prev_ir))
}
/* Handy match templates
 *

 fn compile_decl(cx: &Cx, decl: &ir::Decl<TypeSym>) {
 match decl {
 D::Function {
 name,
 signature,
 body,
 } => {
 }
 D::Const {
/*
name,
typename,
init,
*/
..
} => {
}
}
}

fn compile_expr(
    cx: &Cx,
    expr: &ir::TypedExpr<TypeSym>,
) {
    match &expr.e {
        E::Lit { val } => match val {
            Literal::Integer(i) => {
            }
            Literal::Bool(b) => {
            }
            // noop
            Literal::Unit => (),
        },
        E::Var { name } => {
        }
        E::BinOp { op, lhs, rhs } => {
        }
        E::UniOp { op, rhs } => match op {
            ir::UOp::Neg => {
            }
            ir::UOp::Not => {
            }
        },
        E::Block { body } => {
        }
        E::Let {
            varname,
            typename,
            init,
        } => {
        }
        E::If { cases, falseblock } => {
        }

        E::Loop { body } => todo!(),
        E::Lambda { signature, body } => todo!(),
        E::Funcall { func, params } => {
        }
        E::Break => todo!(),
        E::Return { retval } => {
        }
    };
}
*/

/* TODO: Try again to make this work
fn walks<T1, T2>(
    cx: &Cx,
    exprs: Vec<TypedExpr<T1>>,
    f: impl Fn(&Cx, TypedExpr<T1>) -> TypedExpr<T2>,
) -> Vec<TypedExpr<T2>> {
    exprs.into_iter().map(|e| walk(cx, e, f)).collect()
}

/// The idea is that you pass it a function that matches on some expr types
/// and overrides them, and on the rest it just falls through to this which
/// walks the tree.
fn walk<T1, T2>(
    cx: &Cx,
    expr: TypedExpr<T1>,
    f: impl Fn(&Cx, TypedExpr<T1>) -> TypedExpr<T2>,
) -> TypedExpr<T2> {
    f(cx, expr)
    let result = match expr.e {
        E::BinOp { op, lhs, rhs } => {
            let nlhs = walk(cx, *lhs, f);
            let nrhs = walk(cx, *rhs, f);
            E::BinOp {
                op,
                lhs: Box::new(nlhs),
                rhs: Box::new(nrhs),
            }
        }
        E::UniOp { op, rhs } => {
            let nrhs = walk(cx, *rhs, f);
            E::UniOp {
                op,
                rhs: Box::new(nrhs),
            }
        }
        E::Block { body } => E::Block {
            body: walks(cx, body, f),
        },
        E::Let {
            varname,
            typename,
            init,
        } => E::Let {
            varname,
            typename,
            init: Box::new(walk(cx, *init, f)),
        },
        E::If { cases, falseblock } => {
            let new_cases = cases
                .into_iter()
                .map(|(test, case)| {
                    let new_test = walk(cx, test, f);
                    let new_cases = walks(cx, case, f);
                    (new_test, new_cases)
                })
                .collect();
            let new_falseblock = walks(cx, falseblock, f);
            E::If {
                cases: new_cases,
                falseblock: new_falseblock,
            }
        }

        E::Loop { body } => E::Loop {
            body: walks(cx, body, f),
        },
        E::Return { retval } => E::Return {
            retval: Box::new(walk(cx, *retval, f)),
        },
        E::Funcall { func, params } => {
            let new_func = Box::new(walk(cx, *func, f));
            let new_params = walks(cx, params, f);
            E::Funcall {
                func: new_func,
                params: new_params,
            }
        }
        E::Lambda { signature, body } => {
            // This is actually the only important bit.
            // TODO: Make a more informative name, maybe including the file and line number or
            // such.
            E::Lambda {
                signature,
                body: walks(cx, body, f),
            }
        }
    };
}
*/

/// Lambda lift a single expr.
fn lambda_lift_expr(expr: TypedExpr<()>, output_funcs: &mut Vec<D<()>>) -> TypedExpr<()> {
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
        E::If { cases, falseblock } => {
            let new_cases = cases
                .into_iter()
                .map(|(test, case)| {
                    let new_test = lambda_lift_expr(test, output_funcs);
                    let new_cases = lambda_lift_exprs(case, output_funcs);
                    (new_test, new_cases)
                })
                .collect();
            let new_falseblock = lambda_lift_exprs(falseblock, output_funcs);
            E::If {
                cases: new_cases,
                falseblock: new_falseblock,
            }
        }

        E::Loop { body } => E::Loop {
            body: lambda_lift_exprs(body, output_funcs),
        },
        E::Return { retval } => E::Return {
            retval: Box::new(lambda_lift_expr(*retval, output_funcs)),
        },
        E::Funcall { func, params } => {
            let new_func = Box::new(lambda_lift_expr(*func, output_funcs));
            let new_params = lambda_lift_exprs(params, output_funcs);
            E::Funcall {
                func: new_func,
                params: new_params,
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
    *plz(result)
}

/// Lambda lift a list of expr's
fn lambda_lift_exprs(
    exprs: Vec<TypedExpr<()>>,
    output_funcs: &mut Vec<D<()>>,
) -> Vec<TypedExpr<()>> {
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
/// to do yet.  Add a tag type of some kind to the Ir<T>?
fn lambda_lifting(ir: Ir<()>) -> Ir<()> {
    let mut new_functions = vec![];
    let new_decls: Vec<D<()>> = ir
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

/// Takes an IR containing compound types (currently just tuples)
/// treated as value types, and turns it into one containing
/// only reference types -- ie, anything with subdividable
/// fields and/or bigger than a machine register (effectively
/// 64-bits for wasm) is only referred to through a pointer.
///
/// We might be able to get rid of TupleRef's by turning
/// them into pointer arithmatic, too.
fn _pointerification(_ir: Ir<TypeSym>) -> Ir<TypeSym> {
    todo!()
}

/*
/// This isn't quite right, it returns true if ANY part of the subexpr
/// is const...
fn expr_is_const(expr: &TypedExpr<()>) -> bool {
eval_const_expr(expr).is_some()
}
*/

/*
    /// Returns a new expression that has evaluated all const expr's the
    /// given one contains, or None if the given expr is not const/is unmodified.
    fn eval_const_expr(_expr: &TypedExpr<TypeSym>) -> Option<TypedExpr<TypeSym>> {
    todo!()
/*
let new_expr = match &expr.e {
E::Lit { val } => Some(E::Lit { val: val.clone() }),
E::BinOp { op, lhs, rhs } => {
let elhs = eval_const_expr(lhs);
let erhs = eval_const_expr(rhs);
match (elhs, erhs) {
    // We have evaluated both subexpressions all the way to literals
    (Some(E::Lit { val: new_lhs }), Some(E::Lit { val: new_rhs })) => {
// Actually evaluating this stuff needs to happen AFTER typechecking,
// so we know the expressions are actually valid!
todo!()
}

// We have evaluated part of both subexpressions but can't go further.
(Some(new_lhs), Some(new_rhs)) => Some(E::BinOp {
op: *op,
lhs: Box::new(new_lhs),
rhs: Box::new(new_rhs),
}),
// We have evaluated part of one or the other side
(Some(new_lhs), None) => Some(E::BinOp {
op: *op,
lhs: Box::new(new_lhs),
rhs: rhs.clone(),
}),
(None, Some(new_rhs)) => Some(E::BinOp {
op: *op,
lhs: lhs.clone(),
rhs: Box::new(new_rhs),
}),
// We can't evaluate this binop further
(None, None) => None,
}
}

E::UniOp { .. } => None,
/*
E::Var { .. } => false,
E::Block { body } => body.iter().fold(true, |is_const, next_expr| {
is_const && expr_is_const(next_expr)
}),
E::Let { init, .. } => expr_is_const(init),
E::If { .. } => {
    // TODO: Could be true
    // if all cases and bodies are const
    false
    }
    E::Loop { .. } => {
// TODO: Could be true???
// Well currently this is an infinite loop, so, hah
false
}
E::Lambda { .. } => {
// TODO: Thonk about this one
false
}
E::Funcall { .. } => {
// TODO: True if the function is const
false
}
E::Break => false,
E::Return { .. } => false,
*/
_ => None,
  }?;
// TODO: We assume the type doesn't change, is that valid?
// Currently it is, but not if we get more powerful evaluation!
Some(*plz(new_expr))
    */
    }

/// Simple constant folding.
/// TODO: Everything.
/// Also, won't this need to have type info?  Maybe...
fn const_folding(_ir: Ir<()>) -> Ir<()> {
    /*
       let lowered_decls = ir
       .decls
       .into_iter()
       .map(|decl|
       eval_const_expr(decl))
       .collect::<Result<Vec<ir::Decl<TypeSym>>, TypeError>>()?;
       Ok(ir::Ir {
       decls: checked_decls,
       })
       */
    todo!()
}
*/
