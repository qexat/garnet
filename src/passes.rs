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
//! TODO also: monomorphization

use crate::hir::{Decl as D, Expr as E, ExprNode, Ir};
use crate::*;

type Pass = fn(Ir) -> Ir;

/// Tck has to be mutable because we may change the return type
/// of expr nodes.
type TckPass = fn(Ir, &mut typeck::Tck) -> Ir;

pub fn run_passes(ir: Ir) -> Ir {
    let passes: &[Pass] = &[lambda_lift];
    passes.iter().fold(ir, |prev_ir, f| f(prev_ir))
}

pub fn run_typechecked_passes(ir: Ir, tck: &mut typeck::Tck) -> Ir {
    // let passes: &[TckPass] = &[nameify, enum_to_int];
    //let passes: &[TckPass] = &[nameify, struct_to_tuple];
    let passes: &[TckPass] = &[struct_to_tuple, monomorphize];
    let res = passes.iter().fold(ir, |prev_ir, f| f(prev_ir, tck));
    println!();
    println!("{}", res);
    res
}

fn exprs_map(exprs: Vec<ExprNode>, f: &mut dyn FnMut(ExprNode) -> ExprNode) -> Vec<ExprNode> {
    exprs.into_iter().map(|e| expr_map(e, f)).collect()
}

/// To handle multiple expressions this will turn more into a fold()
/// than a map(), it will take an accumulator that gets threaded through
/// everything...  That gets *very weird* though and I manage to pour
/// my coffee on my cat this morning so let's give that a miss for now.
/// As it is, this can only transform subtrees into other subtrees.
///
/// We don't need a version of this for decls, 'cause decls
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
    /*
    let x = expr.map(exprfn);
    f(x)
    */
    thing.map(exprfn)
}

/*
/// A pure version of the expr_map() that takes and returns state explicitly.
///     Seems to be strictly more of a pain in the ass than smuggling a closure
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

/*
/// Takes a decl and applies the given expr transformer to any expression in it.
/// ...Not suuuuuuuure it is worth it since we often(?) want to know or change
/// something about the decl as well.
fn decl_map_expr(decl: D, f: &mut dyn FnMut(Expr) -> Expr) -> Decl {
    match decl {
        D::Function { name, signature, body } => {
            D::Function
        },
        D::Const { name, typename, init } => {

        },
        D::TypeDef { name, params, typedecl } => {

        }
    }
}
*/

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

//////  Lambda lifting  /////

/// Lambda lift a single expr, creating a new toplevel function if necessary.
fn lambda_lift_expr(expr: ExprNode, output_funcs: &mut Vec<D>) -> ExprNode {
    let result = &mut |e| match e {
        E::Lambda { signature, body } => {
            // This is actually the only important bit.
            // TODO: Make a more informative name, maybe including the file and line number or
            // such.
            let lambda_name = INT.gensym("lambda");
            let function_decl = D::Function {
                name: lambda_name,
                signature,
                body: exprs_map(body, &mut |e| lambda_lift_expr(e, output_funcs)),
            };
            output_funcs.push(function_decl);
            E::Var { name: lambda_name }
        }
        x => x,
    };
    expr.map(result)
}

/// A transformation pass that removes lambda expressions and turns
/// them into function decl's.
/// TODO: Doesn't handle actual closures yet though.  That should be a
/// separate pass that happens first?
///
/// TODO: Output is a HIR tree that does not have any lambda expr's in it, which
/// I would like to make un-representable, but don't see a good way
/// to do yet.  Add a tag type of some kind to the Ir?
/// Eeeeeeeeh we might just have to check on it later one way or another,
/// either when doing codegen or when transforming our HIR to a lower-level IR.
fn lambda_lift(ir: Ir) -> Ir {
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
                let new_body = exprs_map(body, &mut |e| lambda_lift_expr(e, &mut new_functions));
                D::Function {
                    name,
                    signature,
                    body: new_body,
                }
            }
            D::Const {
                name,
                typename,
                init,
            } => {
                let new_body = expr_map(init, &mut |e| lambda_lift_expr(e, &mut new_functions));
                D::Const {
                    name,
                    typename,
                    init: new_body,
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

//////  Monomorphization //////

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
fn monomorphize(ir: Ir, _tck: &mut typeck::Tck) -> Ir {
    let mut functioncalls: BTreeSet<(Sym, Vec<Type>)> = Default::default();
    fn mangle_generic_name(s: Sym, tys: &[Type]) -> Sym {
        todo!()
    }
    ir
}

//////  Turn all anonymous types into named ones ////

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
            // let generics = nam
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
                typename,
                init,
            } => {
                // gramble gramble can't have two closures borrowing known_types at the same time
                let new_type = type_map(typename, &mut |t| nameify_type(t, known_types));
                let new_init = expr_map(init, &mut |e| nameify_expr(e, tck, known_types));
                D::Const {
                    name,
                    typename: new_type,
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
        new_decls.push(res);
    }
    new_decls.extend(known_types.into_iter().map(|(_k, v)| v.clone()));
    Ir { decls: new_decls }
}

//////  Struct to tuple transformation //////

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
    println!("Tuplizing expr {:?}", expr);
    let expr_typeid = tck.get_expr_type(&expr);
    let struct_type = tck.reconstruct(expr_typeid).expect("Should never happen?");
    let new_contents = match &*expr.e {
        E::StructCtor { body } => match &struct_type {
            // TODO: Generics?
            Type::Struct(type_body, _generics) => {
                let mut ordered_body: Vec<_> = body
                    .into_iter()
                    .map(|(ky, vl)| (offset_of_field(&type_body, *ky), vl))
                    .collect();
                ordered_body.sort_by(|a, b| a.0.cmp(&b.0));
                let new_body = ordered_body
                    .into_iter()
                    .map(|(_i, expr)| expr.clone())
                    .collect();

                E::TupleCtor { body: new_body }
            }
            _other => unreachable!("Should never happen?"),
        },
        E::StructRef {
            expr: inner_expr,
            elt,
        } => {
            let inner_typeid = tck.get_expr_type(&inner_expr);
            let struct_type = tck.reconstruct(inner_typeid).expect("Should never happen?");
            match struct_type {
                Type::Struct(type_body, _generics) => {
                    let offset = offset_of_field(&type_body, *elt);
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
        _other => *expr.e.clone(),
    };
    // Change the type of the struct literal expr node to the new tuple
    // We need to do this to any expression because its type may be a struct.
    // And we need to do this after the above because we've changed the
    let new_t = tuplize_type(struct_type.clone());
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
fn struct_to_tuple(ir: Ir, tck: &mut typeck::Tck) -> Ir {
    let mut new_decls = vec![];
    let tuplize_expr = &mut |e| tuplize_expr(e, tck);

    for decl in ir.decls.into_iter() {
        let res = match decl {
            D::Const {
                name,
                typename,
                init,
            } => {
                let new_type = type_map(typename, &mut tuplize_type);
                let new_init = expr_map(init, tuplize_expr);
                D::Const {
                    name,
                    typename: new_type,
                    init: new_init,
                }
            }
            D::Function {
                name,
                signature,
                body,
            } => D::Function {
                name,
                signature: signature_map(signature, &mut tuplize_type),
                body: exprs_map(body, tuplize_expr),
            },
            D::TypeDef {
                name,
                params,
                typedecl,
            } => D::TypeDef {
                name,
                params,
                typedecl: type_map(typedecl, &mut tuplize_type),
            },
        };
        new_decls.push(res);
    }
    Ir { decls: new_decls }
}

//////  Enum to integer transformation //////

/// Takes an arbitrary type and recursively turns
/// structs into tuples.
fn intify_type(typ: Type) -> Type {
    match typ {
        Type::Enum(_fields) => Type::i32(),
        other => other,
    }
}

/// Take an expression and turn any enum literals into integers.
fn intify_expr(expr: ExprNode, tck: &mut typeck::Tck) -> ExprNode {
    let expr_typeid = tck.get_expr_type(&expr);
    let expr_type = tck.reconstruct(expr_typeid).expect("Should never happen?");
    let new_contents = match &*expr.e {
        E::EnumCtor {
            name: _,
            variant: _,
            value,
        } => E::Lit {
            val: hir::Literal::SizedInteger {
                vl: *value as i128,
                bytes: 4,
            },
            // match &expr_type {
            //     Type::Enum(body) => {
            //         let res = body
            //             .iter()
            //             .find(|(sym, _vl)| sym == valname)
            //             .expect("Enum didn't have the field we selected, should never happen");
            //         // Right now enums are always repr(i32)
            //     }
            //     _other => unreachable!("Should never happen?  {:?}", _other),
            //_other => *expr.e.clone(),
        },
        _other => *expr.e.clone(),
    };

    let new_t = intify_type(expr_type);
    tck.replace_expr_type(&expr, &new_t);
    expr.map(&mut |_| new_contents.clone())
}

/// Takes any enum types or values and replaces them with integers.
fn enum_to_int(ir: Ir, tck: &mut typeck::Tck) -> Ir {
    let mut new_decls = vec![];
    let intify_expr = &mut |e| intify_expr(e, tck);

    for decl in ir.decls.into_iter() {
        let res = match decl {
            D::Const {
                name,
                typename,
                init,
            } => {
                let new_type = type_map(typename, &mut intify_type);
                let new_init = expr_map(init, intify_expr);
                D::Const {
                    name,
                    typename: new_type,
                    init: new_init,
                }
            }
            D::Function {
                name,
                signature,
                body,
            } => D::Function {
                name,
                signature: signature_map(signature, &mut intify_type),
                body: exprs_map(body, intify_expr),
            },
            D::TypeDef {
                name,
                params,
                typedecl,
            } => D::TypeDef {
                name,
                params,
                typedecl: type_map(typedecl, &mut intify_type),
            },
        };
        new_decls.push(res);
    }
    Ir { decls: new_decls }
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

#[cfg(test)]
mod tests {
    use super::*;
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

    #[test]
    fn test_expr_map() {
        fn swap_binop_args(expr: ExprNode) -> ExprNode {
            println!("Swapping {:?}", expr);
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
