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
//!
//! TODO: Passes that need to be done still:
//!  * Closure conversion
//!  * Monomorphization
//!  * Pointerification, turning large struct values into pointers?  Maybe.
//!  * Turning all enums into ints?  Maybe.

// Not 100% convinced that having a submodule for each pass is
// necessary, since many of the passes are pretty small, but
// oh well.

//mod enum_to_int;
mod lambda_lift;
mod monomorphization;
mod struct_to_tuple;

use crate::hir::{Decl as D, Expr as E, ExprNode, Ir};
use crate::*;

type Pass = fn(Ir) -> Ir;

/// Tck has to be mutable because we may change the return type
/// of expr nodes.
type TckPass = fn(Ir, &mut typeck::Tck) -> Ir;

pub fn run_passes(ir: Ir) -> Ir {
    // TODO: It may be more efficient to compose passes rather than fold them?
    // That way they don't need to build the full intermediate IR each time.
    // You can use the recursion scheme pattern for this.
    // That will take some nontrivial restructuring of expr_map though, will also need
    // a decl_map or something that can compose multiple passes together.
    // Probably not *difficult*, but tricksy.
    let passes: &[Pass] = &[lambda_lift::lambda_lift];
    passes.iter().fold(ir, |prev_ir, f| f(prev_ir))
}

pub fn run_typechecked_passes(ir: Ir, tck: &mut typeck::Tck) -> Ir {
    // let passes: &[TckPass] = &[nameify, enum_to_int];
    //let passes: &[TckPass] = &[nameify, struct_to_tuple];
    let passes: &[TckPass] = &[
        struct_to_tuple::struct_to_tuple,
        monomorphization::monomorphize,
    ];
    let res = passes.iter().fold(ir, |prev_ir, f| f(prev_ir, tck));
    res
}

fn exprs_map(exprs: Vec<ExprNode>, f: &mut dyn FnMut(ExprNode) -> ExprNode) -> Vec<ExprNode> {
    exprs.into_iter().map(|e| expr_map(e, f)).collect()
}

/// This is the core recusion scheme that takes a function and applies
/// it to an expression, and all its subexpressions.
/// The function is `&mut FnMut` so if it needs to have access to more
/// data between calls besides the expression itself, just smuggle it
/// into the closure.  Trying to make this generic over different funcs
/// signatures is probably not worth the effort
///
/// To handle multiple expressions this will turn more into a fold()
/// than a map(), it will take an accumulator that gets threaded through
/// everything...  That gets *very weird* though and I manage to pour
/// my coffee on my cat this morning so let's give that a miss for now.
/// As it is, this can only transform subtrees into other subtrees.
///
/// We don't really need a version of this for decls, 'cause decls
/// can't be nested.
///
/// The function has to take and return an ExprNode, not an expr,
/// because they may have to look up the type of the expression.
///
/// HORRIBLE NUANCE: This does a *pre-traversal*, ie, it calls the given function on the
/// current node, and then on its sub-nodes.  This is because the function may hae
/// to look at the unaltered values/types of the subnodes to decide what to do!
/// I have *no idea* whether pre-traversal is always what we want!
fn expr_map(expr: ExprNode, f: &mut dyn FnMut(ExprNode) -> ExprNode) -> ExprNode {
    let thing = f(expr);
    let exprfn = &mut |e| match e {
        // Nodes with no recursive expressions.
        // I list all these out explicitly instead of
        // using a catchall so it doesn't fuck up if we change
        // the type of Expr.
        E::Lit { .. } => e,
        E::Var { .. } => e,
        E::Break => e,
        E::EnumCtor { .. } => e,
        E::TupleCtor { body } => E::TupleCtor {
            body: exprs_map(body, f),
        },
        E::StructCtor { body } => {
            let new_body = body
                .into_iter()
                .map(|(sym, vl)| (sym, expr_map(vl, f)))
                .collect();
            E::StructCtor { body: new_body }
        }
        E::TypeCtor {
            name,
            type_params,
            body,
        } => E::TypeCtor {
            name,
            type_params,
            body: expr_map(body, f),
        },
        E::SumCtor {
            name,
            variant,
            body,
        } => E::SumCtor {
            name,
            variant,
            body: expr_map(body, f),
        },
        E::ArrayCtor { body } => E::ArrayCtor {
            body: exprs_map(body, f),
        },
        E::TypeUnwrap { expr } => E::TypeUnwrap {
            expr: expr_map(expr, f),
        },
        E::TupleRef { expr, elt } => E::TupleRef {
            expr: expr_map(expr, f),
            elt,
        },
        E::StructRef { expr, elt } => E::StructRef {
            expr: expr_map(expr, f),
            elt,
        },
        E::ArrayRef { e, idx } => E::ArrayRef {
            e: expr_map(e, f),
            idx,
        },
        E::Assign { lhs, rhs } => E::Assign {
            lhs: expr_map(lhs, f), // TODO: Think real hard about lvalues
            rhs: expr_map(rhs, f),
        },
        E::BinOp { op, lhs, rhs } => E::BinOp {
            op,
            lhs: expr_map(lhs, f),
            rhs: expr_map(rhs, f),
        },
        E::UniOp { op, rhs } => E::UniOp {
            op,
            rhs: expr_map(rhs, f),
        },
        E::Block { body } => E::Block {
            body: exprs_map(body, f),
        },
        E::Let {
            varname,
            typename,
            init,
            mutable,
        } => E::Let {
            varname,
            typename,
            init: expr_map(init, f),
            mutable,
        },
        E::If { cases } => {
            let new_cases = cases
                .into_iter()
                .map(|(test, case)| {
                    let new_test = expr_map(test, f);
                    let new_cases = exprs_map(case, f);
                    (new_test, new_cases)
                })
                .collect();
            E::If { cases: new_cases }
        }
        E::Loop { body } => E::Loop {
            body: exprs_map(body, f),
        },
        E::Return { retval } => E::Return {
            retval: expr_map(retval, f),
        },
        E::Funcall {
            func,
            params,
            type_params,
        } => {
            let new_func = expr_map(func, f);
            let new_params = exprs_map(params, f);
            E::Funcall {
                func: new_func,
                params: new_params,
                type_params,
            }
        }
        E::Lambda { signature, body } => E::Lambda {
            signature,
            body: exprs_map(body, f),
        },
    };
    thing.map(exprfn)
}

/*
/// A pure version of the expr_map() that takes and returns state explicitly.
/// Seems to be strictly more of a pain in the ass than smuggling a closure
/// into expr_map(), so probably not worth the trouble.
fn expr_fold<S>(expr: ExprNode, state: S, f: &dyn Fn(E, S) -> (E, S)) -> (ExprNode, S)
where
    S: Clone,
{
    let res = |e, state: S| {
        let mut st = state;
        // TODO: Packing our mutable state away in a closure and just using
        // expr_map is in fact way more convenient than mongling it explicitly.
        // The cost is requiring that S is cloned on each node.  We could
        // probably do all the state-mongling ourself and change `expr_map()`
        // to be implemented in terms of `expr_fold()`, but that really doesn't
        // sound like much fun.
        let newfun = &mut |e| {
            let (new_expr, new_state) = f(e, st.clone());
            st = new_state;
            new_expr
        };
        let res = match e {
            // Nodes with no recursive expressions.
            // I list all these out explicitly instead of
            // using a catchall so it doesn't fuck up if we change
            // the type of Expr.
            E::Lit { .. } => e,
            E::Var { .. } => e,
            E::Break => e,
            E::TupleCtor { body } => E::TupleCtor {
                body: exprs_map(body, newfun),
            },
            E::StructCtor { body } => {
                let new_body = body
                    .into_iter()
                    .map(|(sym, vl)| (sym, expr_map(vl, newfun)))
                    .collect();
                E::StructCtor { body: new_body }
            }
            E::TypeCtor {
                name,
                type_params,
                body,
            } => E::TypeCtor {
                name,
                type_params,
                body: expr_map(body, newfun),
            },
            E::SumCtor {
                name,
                variant,
                body,
            } => E::SumCtor {
                name,
                variant,
                body: expr_map(body, newfun),
            },
            E::ArrayCtor { body } => E::ArrayCtor {
                body: exprs_map(body, newfun),
            },
            E::TypeUnwrap { expr } => E::TypeUnwrap {
                expr: expr_map(expr, newfun),
            },
            E::TupleRef { expr, elt } => E::TupleRef {
                expr: expr_map(expr, newfun),
                elt,
            },
            E::StructRef { expr, elt } => E::StructRef {
                expr: expr_map(expr, newfun),
                elt,
            },
            E::ArrayRef { e, idx } => E::ArrayRef {
                e: expr_map(e, newfun),
                idx,
            },
            E::Assign { lhs, rhs } => E::Assign {
                lhs: expr_map(lhs, newfun), // TODO: Think real hard about lvalues
                rhs: expr_map(rhs, newfun),
            },
            E::BinOp { op, lhs, rhs } => E::BinOp {
                op,
                lhs: expr_map(lhs, newfun),
                rhs: expr_map(rhs, newfun),
            },
            E::UniOp { op, rhs } => E::UniOp {
                op,
                rhs: rhs.map(newfun),
            },
            E::Block { body } => E::Block {
                body: exprs_map(body, newfun),
            },
            E::Let {
                varname,
                typename,
                init,
                mutable,
            } => E::Let {
                varname,
                typename,
                init: expr_map(init, newfun),
                mutable,
            },
            E::If { cases } => {
                let new_cases = cases
                    .into_iter()
                    .map(|(test, case)| {
                        let new_test = expr_map(test, newfun);
                        let new_cases = exprs_map(case, newfun);
                        (new_test, new_cases)
                    })
                    .collect();
                E::If { cases: new_cases }
            }
            E::Loop { body } => E::Loop {
                body: exprs_map(body, newfun),
            },
            E::Return { retval } => E::Return {
                retval: expr_map(retval, newfun),
            },
            E::Funcall { func, params } => {
                let new_func = expr_map(func, newfun);
                let new_params = exprs_map(params, newfun);
                E::Funcall {
                    func: new_func,
                    params: new_params,
                }
            }
            E::Lambda { signature, body } => E::Lambda {
                signature,
                body: exprs_map(body, newfun),
            },
        };
        f(res, st)
    };
    expr.fold(state, &res)
}

fn exprs_fold<S>(exprs: Vec<ExprNode>, state: S, f: &dyn Fn(E, S) -> (E, S)) -> (Vec<ExprNode>, S)
where
    S: Clone,
{
    let mut res = vec![];
    let mut new_state = state;
    for e in exprs {
        let (new_e, new_s) = expr_fold(e, new_state, f);
        new_state = new_s;
        res.push(new_e);
    }
    (res, new_state)
}
*/

/// Takes a decl and applies the given expr and type transformers to any
/// expression in it.
///
/// Technically not very necessary, since decls can't contain othe decls,
/// but happens often enough that it's worth having.
fn decl_map(
    decl: D,
    fe: &mut dyn FnMut(ExprNode) -> ExprNode,
    ft: &mut dyn FnMut(Type) -> Type,
) -> D {
    match decl {
        D::Function {
            name,
            signature,
            body,
        } => D::Function {
            name,
            signature: signature_map(signature, ft),
            body: exprs_map(body, fe),
        },
        D::Const { name, typ, init } => D::Const {
            name,
            typ: type_map(typ, ft),
            init: expr_map(init, fe),
        },
        D::TypeDef {
            name,
            params,
            typedecl,
        } => D::TypeDef {
            name,
            params,
            typedecl: type_map(typedecl, ft),
        },
    }
}

fn types_map(typs: Vec<Type>, f: &mut dyn FnMut(Type) -> Type) -> Vec<Type> {
    typs.into_iter().map(|t| type_map(t, f)).collect()
}

/// Recursion scheme to turn one type into another.
fn type_map(typ: Type, f: &mut dyn FnMut(Type) -> Type) -> Type {
    /// Really this is only here to be cute, it's used a grand total of twice.
    /// There's probably some horrible combinator chain we could use to make
    /// it generic over any iterator, if we want to make life even harder for
    /// ourself.
    fn types_map_btree<K>(
        typs: BTreeMap<K, Type>,
        f: &mut dyn FnMut(Type) -> Type,
    ) -> BTreeMap<K, Type>
    where
        K: Ord,
    {
        typs.into_iter()
            .map(|(key, ty)| (key, type_map(ty, f)))
            .collect()
    }
    let res = match typ {
        Type::Struct(fields, generics) => {
            let fields = types_map_btree(fields, f);
            Type::Struct(fields, generics)
        }
        Type::Sum(fields, generics) => {
            let new_fields = types_map_btree(fields, f);
            Type::Sum(new_fields, generics)
        }
        Type::Array(ty, len) => Type::Array(Box::new(type_map(*ty, f)), len),
        Type::Func(args, rettype) => {
            let new_args = types_map(args, f);
            let new_rettype = type_map(*rettype, f);
            Type::Func(new_args, Box::new(new_rettype))
        }
        // Not super sure whether this is necessary, but can't hurt.
        Type::Named(nm, tys) => Type::Named(nm, types_map(tys, f)),
        Type::Prim(_) => typ,
        Type::Enum(_) => typ,
        Type::Generic(_) => typ,
    };
    f(res)
}

/// Produce a new signature by transforming the types
fn signature_map(sig: hir::Signature, f: &mut dyn FnMut(Type) -> Type) -> hir::Signature {
    let new_params = sig
        .params
        .into_iter()
        .map(|(sym, ty)| (sym, type_map(ty, f)))
        .collect();
    hir::Signature {
        params: new_params,
        rettype: type_map(sig.rettype, f),
    }
}

pub fn generate_type_name(typ: &Type) -> String {
    match typ {
        Type::Enum(fields) => {
            let fieldnames: Vec<_> = fields
                .iter()
                .map(|(nm, vl)| format!("F{}_{}", nm, vl))
                .collect();
            let fieldstr = fieldnames.join("_");
            format!("__Enum{}", fieldstr)
        }
        Type::Func(params, rettype) => {
            let paramnames: Vec<String> = params.iter().map(generate_type_name).collect();
            let paramstr = paramnames.join("_");
            let retname = generate_type_name(rettype);
            format!("__Func__{}__{}", paramstr, retname)
        }
        Type::Struct(body, _) => {
            let fieldnames: Vec<_> = body
                .iter()
                .map(|(nm, ty)| format!("{}_{}", nm, generate_type_name(ty)))
                .collect();
            let fieldstr = fieldnames.join("_");
            format!("__Struct__{}", fieldstr)
        }
        Type::Sum(body, _) => {
            let fieldnames: Vec<_> = body
                .iter()
                .map(|(nm, ty)| format!("{}_{}", nm, generate_type_name(ty)))
                .collect();
            let fieldstr = fieldnames.join("_");
            format!("__Sum__{}", fieldstr)
        }
        Type::Generic(name) => {
            format!("G{}", name)
        }
        Type::Named(name, fields) => {
            let field_names: Vec<_> = fields.iter().map(generate_type_name).collect();
            let field_str = field_names.join("_");
            format!("__Named{}__{}", name, field_str)
        }
        Type::Prim(p) => p.get_name().into_owned(),
        Type::Array(t, len) => {
            format!("__Arr{}__{}", generate_type_name(t), len)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_expr_map() {
        fn swap_binop_args(expr: ExprNode) -> ExprNode {
            trace!("Swapping {:?}", expr);
            expr.map(&mut |e| match e {
                E::BinOp { op, lhs, rhs } => E::BinOp {
                    op,
                    lhs: rhs,
                    rhs: lhs,
                },
                other => other,
            })
        }
        // Make sure this works for trivial exppressions
        let inp = ExprNode::new(E::BinOp {
            op: hir::BOp::Add,
            lhs: ExprNode::int(3),
            rhs: ExprNode::int(4),
        });
        let desired = ExprNode::new(E::BinOp {
            op: hir::BOp::Add,
            lhs: ExprNode::int(4),
            rhs: ExprNode::int(3),
        });
        let out = expr_map(inp, &mut swap_binop_args);
        assert_eq!(out, desired);

        // Make sure it recurses properly for deeper trees.
        let inp2 = ExprNode::new(E::BinOp {
            op: hir::BOp::Add,
            lhs: ExprNode::new(E::BinOp {
                op: hir::BOp::Sub,
                lhs: ExprNode::int(100),
                rhs: ExprNode::int(200),
            }),
            rhs: ExprNode::int(4),
        });
        let desired2 = ExprNode::new(E::BinOp {
            op: hir::BOp::Add,
            lhs: ExprNode::int(4),
            rhs: ExprNode::new(E::BinOp {
                op: hir::BOp::Sub,
                lhs: ExprNode::int(200),
                rhs: ExprNode::int(100),
            }),
        });
        let out2 = expr_map(inp2, &mut swap_binop_args);
        assert_eq!(out2, desired2);
    }
}
