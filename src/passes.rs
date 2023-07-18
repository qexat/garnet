//! Transformation/optimization passes that function on the IR.
//! May happen before or after typechecking, either way.
//! For now each is a function from `IR -> IR`, with some recursion
//! scheme helper functions.  There's two types of passes, one that
//! happens before type checking and one that happens after.
//!
//! It could probably be more efficient, especially with
//! IR nodes flattened into a vec, but that doesn't matter for now.
//! We just allocate our way to happiness.
//!
//! TODO: A pass that might be useful would be "alpha renaming".
//! Essentially you walk through your entire compilation unit and
//! rename everything to a globally unique name.  This means that
//! scope vanishes, for example, and anything potentially
//! ambiguous becomes unambiguous.  It can come after typechecking tho.
//!
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
mod constinfer;
mod double_typeck;
mod generic_infer;
mod handle_imports;
mod lambda_lift;
//mod monomorphization;
mod struct_to_tuple;
//mod type_erasure;
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
    let passes: &[Pass] = &[
        handle_imports::handle_imports,
        lambda_lift::lambda_lift,
        generic_infer::generic_infer,
    ];
    passes.iter().fold(ir, |prev_ir, f| f(prev_ir))
}

pub fn run_typechecked_passes(ir: Ir, tck: &mut typeck::Tck) -> Ir {
    // let passes: &[TckPass] = &[nameify, enum_to_int];
    //let passes: &[TckPass] = &[nameify, struct_to_tuple];
    let passes: &[TckPass] = &[
        double_typeck::double_typeck,
        constinfer::constinfer,
        struct_to_tuple::struct_to_tuple,
        //monomorphization::monomorphize,
        //type_erasure::type_erasure,
    ];
    let res = passes.iter().fold(ir, |prev_ir, f| f(prev_ir, tck));
    res
}

/// Handy do-nothing combinator
fn id<T>(thing: T) -> T {
    thing
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
/// This can take a function to call to transform the current node
/// before descending into sub-expressions
/// sub-nodes, and one to call after.  Use expr_map_pre()
/// and expr_map_post() to just do a pre-traversal or a post-traversal
/// specifically.
fn expr_map(
    expr: ExprNode,
    f_pre: &mut dyn FnMut(ExprNode) -> ExprNode,
    f_post: &mut dyn FnMut(ExprNode) -> ExprNode,
) -> ExprNode {
    let thing = f_pre(expr);
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
            body: exprs_map(body, f_pre, f_post),
        },
        E::StructCtor { body } => {
            let new_body = body
                .into_iter()
                .map(|(sym, vl)| (sym, expr_map(vl, f_pre, f_post)))
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
            body: expr_map(body, f_pre, f_post),
        },
        E::SumCtor {
            name,
            variant,
            body,
        } => E::SumCtor {
            name,
            variant,
            body: expr_map(body, f_pre, f_post),
        },
        E::ArrayCtor { body } => E::ArrayCtor {
            body: exprs_map(body, f_pre, f_post),
        },
        E::TypeUnwrap { expr } => E::TypeUnwrap {
            expr: expr_map(expr, f_pre, f_post),
        },
        E::TupleRef { expr, elt } => E::TupleRef {
            expr: expr_map(expr, f_pre, f_post),
            elt,
        },
        E::StructRef { expr, elt } => E::StructRef {
            expr: expr_map(expr, f_pre, f_post),
            elt,
        },
        E::ArrayRef { expr, idx } => E::ArrayRef {
            expr: expr_map(expr, f_pre, f_post),
            idx,
        },
        E::Assign { lhs, rhs } => E::Assign {
            lhs: expr_map(lhs, f_pre, f_post), // TODO: Think real hard about lvalues
            rhs: expr_map(rhs, f_pre, f_post),
        },
        E::BinOp { op, lhs, rhs } => E::BinOp {
            op,
            lhs: expr_map(lhs, f_pre, f_post),
            rhs: expr_map(rhs, f_pre, f_post),
        },
        E::UniOp { op, rhs } => E::UniOp {
            op,
            rhs: expr_map(rhs, f_pre, f_post),
        },
        E::Block { body } => E::Block {
            body: exprs_map(body, f_pre, f_post),
        },
        E::Let {
            varname,
            typename,
            init,
            mutable,
        } => E::Let {
            varname,
            typename,
            init: expr_map(init, f_pre, f_post),
            mutable,
        },
        E::If { cases } => {
            let new_cases = cases
                .into_iter()
                .map(|(test, case)| {
                    let new_test = expr_map(test, f_pre, f_post);
                    let new_cases = exprs_map(case, f_pre, f_post);
                    (new_test, new_cases)
                })
                .collect();
            E::If { cases: new_cases }
        }
        E::Loop { body } => E::Loop {
            body: exprs_map(body, f_pre, f_post),
        },
        E::Return { retval } => E::Return {
            retval: expr_map(retval, f_pre, f_post),
        },
        E::Funcall {
            func,
            params,
            type_params,
        } => {
            let new_func = expr_map(func, f_pre, f_post);
            let new_params = exprs_map(params, f_pre, f_post);
            E::Funcall {
                func: new_func,
                params: new_params,
                type_params,
            }
        }
        E::Lambda { signature, body } => E::Lambda {
            signature,
            body: exprs_map(body, f_pre, f_post),
        },
        E::Typecast { e, to } => E::Typecast {
            e: expr_map(e, f_pre, f_post),
            to,
        },
        E::Ref { expr } => E::Ref {
            expr: expr_map(expr, f_pre, f_post),
        },
        E::Deref { expr } => E::Deref {
            expr: expr_map(expr, f_pre, f_post),
        },
    };
    let post_thing = thing.map(exprfn);
    f_post(post_thing)
}

pub fn expr_map_pre(expr: ExprNode, f: &mut dyn FnMut(ExprNode) -> ExprNode) -> ExprNode {
    expr_map(expr, f, &mut id)
}

pub fn expr_map_post(expr: ExprNode, f: &mut dyn FnMut(ExprNode) -> ExprNode) -> ExprNode {
    expr_map(expr, &mut id, f)
}

/// Map functions over a list of exprs.
fn exprs_map(
    exprs: Vec<ExprNode>,
    f_pre: &mut dyn FnMut(ExprNode) -> ExprNode,
    f_post: &mut dyn FnMut(ExprNode) -> ExprNode,
) -> Vec<ExprNode> {
    exprs
        .into_iter()
        .map(|e| expr_map(e, f_pre, f_post))
        .collect()
}

fn exprs_map_pre(exprs: Vec<ExprNode>, f: &mut dyn FnMut(ExprNode) -> ExprNode) -> Vec<ExprNode> {
    exprs_map(exprs, f, &mut id)
}

fn _exprs_map_post(exprs: Vec<ExprNode>, f: &mut dyn FnMut(ExprNode) -> ExprNode) -> Vec<ExprNode> {
    exprs_map(exprs, &mut id, f)
}

fn decl_map_pre(
    decl: D,
    fe: &mut dyn FnMut(ExprNode) -> ExprNode,
    ft: &mut dyn FnMut(Type) -> Type,
) -> D {
    decl_map(decl, fe, &mut id, ft)
}

fn decl_map_post(
    decl: D,
    fe: &mut dyn FnMut(ExprNode) -> ExprNode,
    ft: &mut dyn FnMut(Type) -> Type,
) -> D {
    decl_map(decl, &mut id, fe, ft)
}

/// Takes a decl and applies the given expr and type transformers to any
/// expression in it.
///
/// Technically not very necessary, since decls can't contain other decls,
/// but happens often enough that it's worth having.
fn decl_map(
    decl: D,
    fe_pre: &mut dyn FnMut(ExprNode) -> ExprNode,
    fe_post: &mut dyn FnMut(ExprNode) -> ExprNode,
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
            body: exprs_map(body, fe_pre, fe_post),
        },
        D::Const { name, typ, init } => D::Const {
            name,
            typ: type_map(typ, ft),
            init: expr_map(init, fe_pre, fe_post),
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
        D::Import { .. } => decl,
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
        Type::Func(args, rettype, typeparams) => {
            let new_args = types_map(args, f);
            let new_rettype = type_map(*rettype, f);
            let new_typeparams = types_map(typeparams, f);
            Type::Func(new_args, Box::new(new_rettype), new_typeparams)
        }
        // Not super sure whether this is necessary, but can't hurt.
        Type::Named(nm, tys) => Type::Named(nm, types_map(tys, f)),
        Type::Prim(_) => typ,
        Type::Never => typ,
        Type::Enum(_) => typ,
        Type::Generic(_) => typ,
        Type::Uniq(t) => {
            let new_t = type_map(*t, f);
            Type::Uniq(Box::new(new_t))
        }
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
        typeparams: types_map(sig.typeparams, f),
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
        Type::Func(params, rettype, typeparams) => {
            let paramnames: Vec<String> = params.iter().map(generate_type_name).collect();
            let paramstr = paramnames.join("_");
            let retname = generate_type_name(rettype);
            let tparamnames: Vec<String> = typeparams.iter().map(generate_type_name).collect();
            let tparamstr = tparamnames.join("_");
            format!("__Func__{}__{}_{}", paramstr, retname, tparamstr)
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
            format!("__G{}", name)
        }
        Type::Named(name, fields) => {
            let field_names: Vec<_> = fields.iter().map(generate_type_name).collect();
            let field_str = field_names.join("_");
            format!("__Named{}__{}", name, field_str)
        }
        Type::Prim(p) => p.get_name().into_owned(),
        Type::Never => format!("!"),
        Type::Array(t, len) => {
            format!("__Arr{}__{}", generate_type_name(t), len)
        }
        Type::Uniq(t) => {
            format!("__Uniq__{}", generate_type_name(t))
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
        let out = expr_map_pre(inp, &mut swap_binop_args);
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
        let out2 = expr_map_pre(inp2, &mut swap_binop_args);
        assert_eq!(out2, desired2);
    }

    /// Test whether our pre-traversal expr map works properly.
    #[test]
    fn test_expr_pretraverse() {
        let inp = ExprNode::new(E::Block {
            body: vec![ExprNode::new(E::Var {
                name: Sym::new("foo"),
            })],
        });
        // Make a transformer that renames vars, and check in the Block
        // body whether the inner var has been transformed.
        let f = &mut |e: ExprNode| {
            let helper = &mut |e| match e {
                E::Var { .. } => E::Var {
                    name: Sym::new("bar!"),
                },
                E::Block { body } => {
                    // Has the name within the body been transformed yet?
                    {
                        let inner = &*body[0].e;
                        assert_eq!(
                            inner,
                            &E::Var {
                                name: Sym::new("foo")
                            }
                        );
                    }
                    E::Block { body }
                }
                _other => _other,
            };
            e.map(helper)
        };
        let outp = expr_map_pre(inp, f);
        // Make sure that the actual transformation has done what we expected.
        let expected = ExprNode::new(E::Block {
            body: vec![ExprNode::new(E::Var {
                name: Sym::new("bar!"),
            })],
        });
        assert_eq!(outp, expected);
    }

    /// Similar to above with the case in helper() reversed
    #[test]
    fn test_expr_posttraverse() {
        let inp = ExprNode::new(E::Block {
            body: vec![ExprNode::new(E::Var {
                name: Sym::new("foo"),
            })],
        });
        let f = &mut |e: ExprNode| {
            let helper = &mut |e| match e {
                E::Var { .. } => E::Var {
                    name: Sym::new("bar!"),
                },
                E::Block { body } => {
                    // Has the name within the body been transformed yet?
                    {
                        let inner = &*body[0].e;
                        assert_eq!(
                            inner,
                            &E::Var {
                                name: Sym::new("bar!")
                            }
                        );
                    }
                    E::Block { body }
                }
                _other => _other,
            };
            e.map(helper)
        };
        let outp = expr_map_post(inp, f);
        // Make sure that the actual transformation has done what we expected.
        let expected = ExprNode::new(E::Block {
            body: vec![ExprNode::new(E::Var {
                name: Sym::new("bar!"),
            })],
        });
        assert_eq!(outp, expected);
    }
}
