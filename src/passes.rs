//! Transformation/optimization passes that function on the IR.
//! May happen before or after typechecking, either way.
//! For now each is a function from `IR -> IR`, rather than
//! having a visitor and mutating stuff or anything like that,
//! which may be less efficient but is IMO simpler to think about.
//! But also maybe more tedious to write.  A proper recursion scheme
//! seems like the thing to do, but
//!
//! TODO: A pass that might be useful would be "alpha renaming".
//! Essentially you walk through your entire compilation unit and
//! rename everything to a globally unique name.  This means that
//! scope vanishes, for example, and anything potentially
//! ambiguous becomes unambiguous.

use crate::hir::{Decl as D, Expr as E, ExprNode, Ir};
use crate::*;

type Pass = fn(Ir) -> Ir;
type TckPass = fn(Ir, &typeck::Tck) -> Ir;

pub fn run_passes(ir: Ir) -> Ir {
    let passes: &[Pass] = &[lambda_lifting];
    passes.iter().fold(ir, |prev_ir, f| f(prev_ir))
}

pub fn run_typechecked_passes(ir: Ir, tck: &typeck::Tck) -> Ir {
    let passes: &[TckPass] = &[anon_struct_to_tuple];
    passes.iter().fold(ir, |prev_ir, f| f(prev_ir, tck))
}

/// Lambda lift a single expr.
fn lambda_lift_expr(expr: ExprNode, output_funcs: &mut Vec<D>) -> ExprNode {
    let result = match *expr.e {
        E::BinOp { op, lhs, rhs } => {
            let nlhs = lambda_lift_expr(lhs, output_funcs);
            let nrhs = lambda_lift_expr(rhs, output_funcs);
            E::BinOp {
                op,
                lhs: nlhs,
                rhs: nrhs,
            }
        }
        E::UniOp { op, rhs } => {
            let nrhs = lambda_lift_expr(rhs, output_funcs);
            E::UniOp { op, rhs: nrhs }
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
            init: lambda_lift_expr(init, output_funcs),
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
            retval: lambda_lift_expr(retval, output_funcs),
        },
        E::Funcall { func, params } => {
            let new_func = lambda_lift_expr(func, output_funcs);
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
    hir::ExprNode {
        e: Box::new(result),
        id: expr.id,
    }
}

/// Lambda lift a list of expr's.
fn lambda_lift_exprs(exprs: Vec<ExprNode>, output_funcs: &mut Vec<D>) -> Vec<ExprNode> {
    exprs
        .into_iter()
        .map(|e| lambda_lift_expr(e, output_funcs))
        .collect()
}

/// A transformation pass that removes lambda expressions and turns
/// them into function decl's.
/// TODO: Doesn't handle actual closures yet though.
///
/// TODO: Output is an IR that does not have any lambda expr's in it, which
/// I would like to make un-representable, but don't see a good way
/// to do yet.  Add a tag type of some kind to the Ir?
/// Eeeeeeeeh we might just have to check on it later one way or another,
/// either when doing codegen or when transforming our HIR to a lower-level IR.
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

fn _enum_to_int_expr(_expr: ExprNode, _output_funcs: &mut Vec<D>) -> ExprNode {
    todo!()
}

/// Takes any anonymous struct types and replaces them with
/// tuples.
/// This is necessary because we have anonymous structs but
/// Rust doesn't.
///
/// I WAS going to turn them into named structs but then declaring them gets
/// kinda squirrelly, because we don't really have a decl that translates
/// directly into a Rust struct without being wrapped in a typedef first.  So I
/// think I will translate them to tuples after all.

fn anon_struct_to_tuple(ir: Ir, tck: &typeck::Tck) -> Ir {
    use hir::Decl;
    // Takes a struct and a field name and returns what tuple
    // offset the field should correspond to.
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

    fn struct_to_tuple(fields: &BTreeMap<Sym, Type>) -> Type {
        // This is why structs contain a BTreeMap,
        // so that this ordering is always consistent based
        // on the field names.
        let tuple_fields = fields.iter().map(|(_sym, ty)| ty.clone()).collect();
        Type::tuple(tuple_fields)
    }

    let mut new_decls = vec![];

    for decl in ir.decls.into_iter() {
        match &decl {
            Decl::Const {
                name,
                typename,
                init,
            } => {
                // This is kinda the simplest case so we'll do it first.
                // If the type is an anonymous struct type,
                // we conjure forth a new named struct.
                match typename {
                    Type::Struct(fields, _generics) => {
                        let tuple = struct_to_tuple(fields);
                        todo!("Translate struct literal init to a tuple")
                    }
                    _ => new_decls.push(decl),
                }
            }
            Decl::Function {
                name,
                signature,
                body,
            } => todo!(),
            Decl::TypeDef {
                name,
                params,
                typedecl,
            } => todo!(),
        }
    }
    Ir { decls: new_decls }
}

/// Takes any enum typedef and values turns them into plain integers.
fn _enum_to_int(ir: Ir, tck: &typeck::Tck) -> Ir {
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
