//! Closure conversion.
//!
//! Takes function expressions and rewrites
//! them to make any captured variables (and type
//! parameters!) explicit.
//!
//! From https://matt.might.net/articles/closure-conversion/
//! the key points are to rewrite a lambda term into
//! a closure-creation term that returns both a closure
//! and an environment?  Then rewrite function calls calling
//! the variable to include the environment.
//!
//! Right now we have no borrowing or anything, so all values
//! are *copied* into their environment.
//!
//! So for a first stab at this, let us merely worry about type
//! parameters.

use std::collections::BTreeSet;

use crate::hir::{Decl, Expr, Ir};
use crate::passes::*;
use crate::typeck::Tck;

// This might be the time to split the Symtbl into the
// scope bit and the renamed table.
use crate::symtbl::{self, Symtbl};

/// A dumb and simple scope for type params.
/// Will eventually get folded into Symtbl but I want to
/// be able to figure out what shape it needs to be first without
/// needing to alter anything that touches Symtbl.
#[derive(Debug, Default)]
struct ScopeThing {
    type_params: Vec<BTreeSet<Sym>>,
}

struct ScopeGuard<'a> {
    scope: &'a mut ScopeThing,
}

impl<'a> Drop for ScopeGuard<'a> {
    fn drop(&mut self) {
        self.scope.pop_scope()
    }
}

impl ScopeThing {
    fn push_scope(&mut self) -> ScopeGuard<'_> {
        self.type_params.push(Default::default());
        ScopeGuard { scope: self }
    }

    fn pop_scope(&mut self) {
        self.type_params
            .pop()
            .expect("Scope stack underflow aieeee");
    }

    fn add_type_params(&mut self, params: &[Sym]) {
        let top = self.type_params.last_mut().expect("Scope stack underflow");
        top.extend(params.iter().cloned());
    }

    fn is_type_param(&self, ty: Sym) -> bool {
        let msg = format!("Walking down scope {:?}", self);
        dbg!(msg);
        for scope in self.type_params.iter().rev() {
            let msg = format!("is {} a type param? {}", ty, scope.contains(&ty));
            dbg!(&msg);
            if scope.contains(&ty) {
                return true;
            }
        }
        return false;
    }
}

fn cc_type(scope: &mut ScopeThing, ty: Type) -> Type {
    match &ty {
        Type::Struct(fields, params) => ty,
        Type::Sum(fields, generics) => ty,
        Type::Array(ty, len) => *ty.clone(),
        Type::Func(args, rettype, typeparams) => ty,
        // Not super sure whether this is necessary, but can't hurt.
        Type::Named(nm, tys) => ty,
        Type::Prim(_) => ty,
        Type::Never => ty,
        Type::Enum(_) => ty,
        Type::Uniq(t) => ty,
    }
}

/// Is a more limited form of recursion scheme what we actually want?
/// This only recurses one level deep, if you want to recurse deeper
/// than that you have to call it explicitly.
/// But that also allows you more control over pre/post/around traversal
/// order.
///
/// Basically the idea is that you can handle the expr types that you
/// want to, and just call do_to_children(...) on the rest.
///
/// You can imagine this as being equivalent to a get_children()
/// function that returns a list of the immediate subexpr's of an expr,
/// and then calls the given function on each item in it.  The advantage
/// of this form being that you don't need to allocate a vec.
/// The result seems to be some kind of weird mutant hybrid of imperative
/// and functional which also seems to be just what I need...
fn do_to_children(expr: &ExprNode, f: &mut dyn FnMut(&ExprNode)) {
    use Expr::*;
    match &*expr.e {
        BinOp { lhs, rhs, .. } => {
            f(lhs);
            f(rhs);
        }
        UniOp { rhs, .. } => f(rhs),
        TupleCtor { body }
        | ArrayCtor { body }
        | Block { body }
        | Loop { body }
        | Lambda { body, .. } => {
            for expr in body {
                f(expr);
            }
        }
        TupleRef { expr, .. }
        | StructRef { expr, .. }
        | Ref { expr }
        | TypeUnwrap { expr }
        | Deref { expr } => f(expr),
        Funcall {
            func,
            params,
            type_params: _,
        } => {
            f(func);
            for vl in params {
                f(vl);
            }
        }
        Let { init, .. } => f(init),
        If { cases } => {
            for (test, exprs) in cases {
                f(test);
                for expr in exprs {
                    f(expr);
                }
            }
        }
        TypeCtor { body, .. } => f(body),
        SumCtor { body, .. } => todo!(),
        Typecast { .. } => todo!(),
        Return { retval } => f(retval),
        StructCtor { body } => {
            for (_nm, expr) in body {
                f(expr);
            }
        }
        ArrayRef { expr, idx } => {
            f(expr);
            f(idx);
        }
        Assign { lhs, rhs } => {
            f(lhs);
            f(rhs);
        }
        // No sub-expressions to these terminals
        Lit { .. } | Var { .. } | EnumCtor { .. } | Break => (),
        // other => expr_iter(expr, &mut |e| {
        //     let res = get_free_type_params(e, scope);
        //     accm.extend(res);
        // }),
    }
}

/// Returns a list of type parameters that are not defined in the given scope.
fn get_free_type_params(expr: &ExprNode, scope: &mut ScopeThing, symtbl: &Symtbl) -> BTreeSet<Sym> {
    let mut accm = BTreeSet::new();
    /// Take a type and collect any named types that aren't in the symbol table
    /// into an array
    ///
    /// Eventually we can probably refactor this into using type_iter()
    /// but handling type params is weird enough that for now I want to keep
    /// it separate.
    fn collect_free_types(
        t: &Type,
        scope: &mut ScopeThing,
        symtbl: &Symtbl,
        accm: &mut BTreeSet<Sym>,
    ) {
        use Type::*;
        match t {
            Named(nm, params) => {
                if !symtbl.type_exists(*nm) && !scope.is_type_param(*nm) {
                    accm.insert(nm.clone());
                }
                for ty in params {
                    collect_free_types(ty, scope, symtbl, accm);
                }
            }
            Func(params, rettype, typeparams) => {
                for ty in params {
                    collect_free_types(ty, scope, symtbl, accm);
                }
                collect_free_types(rettype, scope, symtbl, accm);
                for ty in typeparams {
                    collect_free_types(ty, scope, symtbl, accm);
                }
            }
            Struct(body, typeparams) | Sum(body, typeparams) => {
                for (_nm, ty) in body {
                    collect_free_types(ty, scope, symtbl, accm);
                }
                for ty in typeparams {
                    collect_free_types(ty, scope, symtbl, accm);
                }
            }
            Array(ty, _) | Uniq(ty) => {
                collect_free_types(ty, scope, symtbl, accm);
            }
            // No type parameters for these types
            Prim(_) | Never | Enum(_) => (),
        }
    }
    use hir::Expr::*;
    match &*expr.e {
        Funcall {
            func,
            params,
            type_params,
        } => {
            type_params
                .iter()
                .for_each(&mut |t| collect_free_types(t, scope, symtbl, &mut accm));
            do_to_children(func, &mut |e| {
                accm.extend(get_free_type_params(e, scope, symtbl))
            });
            for param in params {
                do_to_children(param, &mut |e| {
                    accm.extend(get_free_type_params(e, scope, symtbl))
                });
            }
        }
        Let {
            typename: Some(ty),
            init,
            ..
        } => {
            collect_free_types(ty, scope, symtbl, &mut accm);
            do_to_children(init, &mut |e| {
                accm.extend(get_free_type_params(e, scope, symtbl))
            });
        }
        Lambda { signature, body } => {
            // These are type params *defined by* the funcall
            let guard = scope.push_scope();
            guard.scope.add_type_params(&signature.typeparams);
            let ty = signature.to_type();
            collect_free_types(&ty, guard.scope, symtbl, &mut accm);

            // This recuses into the lambda's body *before* we
            // release the scope guard.
            for expr in body {
                do_to_children(expr, &mut |e| {
                    accm.extend(get_free_type_params(e, guard.scope, symtbl))
                });
            }
        }
        TypeCtor {
            type_params, body, ..
        } => {
            // Like a funcall,
            // These are type params *passed into* the constructor
            /*
            let f = &mut |t| collect_free_types(t, scope, &mut accm);
            type_params.iter().for_each(f);
            do_to_children(body, &mut |e| accm.extend(get_free_type_params(e, scope)));
            */

            type_params
                .iter()
                .for_each(&mut |t| collect_free_types(t, scope, symtbl, &mut accm));
            do_to_children(body, &mut |e| {
                accm.extend(get_free_type_params(e, scope, symtbl))
            });
        }
        Typecast { .. } => todo!(),
        // Nothing else binds new type parameters, so we just walk down
        // them looking for expressions that do.
        _other => do_to_children(expr, &mut |e| {
            accm.extend(get_free_type_params(e, scope, symtbl))
        }),
    }
    accm
}

fn cc_expr(scope: &mut ScopeThing, symtbl: &Symtbl, tck: &mut Tck, expr: ExprNode) -> ExprNode {
    let f = &mut |e: Expr| {
        // BUGGO:
        // Not sure whether get_free_type_params() etc should take
        // Expr or ExprNode here, so for now we do this ugly and maybe
        // broken thing
        let hackhackhack = ExprNode::new(e.clone());
        match e {
            Expr::Lambda { signature, body } => {
                // Ok, so what we need to do is look at the body of the closure
                // for types that are not declared in its type params,
                // and if we see any of them in the
                // signature we will need to add type params to this
                // lambda.
                let free_type_params = get_free_type_params(&hackhackhack, scope, symtbl);
                if free_type_params.is_empty() {
                    // no change necessary
                    Expr::Lambda {
                        signature: signature.clone(),
                        body: exprs_map_pre(body, &mut |e| cc_expr(scope, symtbl, tck, e)),
                    }
                } else {
                    // Rewrite lambda into an expression that contains the type params.
                    let mut new_sig = signature.clone();
                    new_sig.typeparams.extend(free_type_params.into_iter());
                    Expr::Lambda {
                        signature: new_sig.clone(),
                        body: exprs_map_pre(body, &mut |e| cc_expr(scope, symtbl, tck, e)),
                    }
                }
            }
            // Here we look at the function invoked and see if we need to modify the
            // funcall to give it more params.
            /*
                Expr::Funcall {
                    func,
                    params,
                    type_params,
                } => {
                    todo!()
                }
            */
            x => x,
        }
    };
    expr.map(f)
}

fn cc_exprs(
    scope: &mut ScopeThing,
    symtbl: &Symtbl,
    tck: &mut Tck,
    exprs: &[ExprNode],
) -> Vec<ExprNode> {
    exprs
        .into_iter()
        .map(|e| cc_expr(scope, symtbl, tck, e.clone()))
        .collect()
}

fn cc_decl(tck: &mut Tck, symtbl: &Symtbl, decl: Decl) -> Decl {
    let scope = &mut ScopeThing::default();
    scope.push_scope();
    match decl {
        D::Function {
            name,
            signature,
            body,
        } => D::Function {
            name,
            signature,
            body: exprs_map_pre(body, &mut |e| cc_expr(scope, symtbl, tck, e)),
        },
        D::Const { name, typ, init } => D::Const {
            name,
            typ,
            init: expr_map_pre(init, &mut |e| cc_expr(scope, symtbl, tck, e)),
        },
        D::TypeDef {
            name,
            params,
            typedecl,
        } => D::TypeDef {
            name,
            params,
            typedecl,
        },
        D::Import { .. } => decl,
    }
}

/// I am somewhat dissatisfied that we have to do symbol table
/// scope-wrangling stuff for this, and then do it again for
/// the symtbl alpha-renaming, but oh well.  They're different
/// things, this adds information and the renaming just
/// transforms it into something more convenient.
pub fn closure_convert(ir: Ir, symtbl: &symtbl::Symtbl, tck: &mut typeck::Tck) -> Ir {
    let new_decls: Vec<Decl> = ir
        .decls
        .into_iter()
        .map(|decl| cc_decl(tck, symtbl, decl))
        .collect();
    Ir {
        decls: new_decls,
        ..ir
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_free_type_param() {
        let test_cases = vec![
            ("3", vec![]),
            ("let foo = bar", vec![]),
            ("let foo I32 = bar", vec![]),
            ("let foo T = bar", vec![Sym::new("T")]),
            (
                "let foo T(Thing) = bar",
                vec![Sym::new("T"), Sym::new("Thing")],
            ),
            ("fn(x I32) I32 = x end", vec![]),
            ("fn(x T) T = x end", vec![Sym::new("T")]),
            ("fn(|T| x T) T = x end", vec![]),
            (
                // We can't have leading newlines in these strings, alas.
                r#"fn(|A| x A) A = 
    let f = fn(x A) A = x end
    f(x)
end"#,
                vec![],
            ),
            (
                r#"fn(|A| x A) A = 
    let f = fn(x A) A = x end
    -- Shadow the type param A
    let g = fn(|A| x A) A = x end
    f(x)
end"#,
                vec![],
            ),
            (
                r#"fn(|A| x A) A = 
    let f = fn(x A) A = x end
    -- Use the unbound type param B
    let g = fn(x B) B = x end
    f(x)
end"#,
                vec![Sym::new("B")],
            ),
        ];
        let scope = &mut ScopeThing::default();
        let symtbl = &symtbl::Symtbl::default();
        for (src, expected) in test_cases {
            let ir = compile_to_hir_expr(src);
            let frees = get_free_type_params(&ir, scope, symtbl);
            use std::iter::FromIterator;
            let expected: BTreeSet<Sym> = BTreeSet::from_iter(expected);
            assert_eq!(frees, expected);
        }
    }

    /*
        we need to run typechecking and stuff to make this work,
        and whole-program tests are best handled as whole-program tests.
    Alas.
        #[test]
        fn test_cc_exprs() {
            let test_cases = vec![
                (r#"3; fn(x A) A = x end"#, r#"3; fn(|A| x A) A = x end"#),
                (
                    r#"fn(x A) A =
        let f = fn(x A) A = x end
        -- Use the unbound type param B
        let g = fn(x B) B = x end
        f(x)
    end"#,
                    r#"fn(|A, B| x A) A =
        let f = fn(|A| x A) A = x end
        -- Use the unbound type param B
        let g = fn(|B| x B) B = x end
        f(x)
    end"#,
                ),
            ];
            let scope = &mut ScopeThing::default();
            for (src, expected) in test_cases {
                let hir = compile_to_hir_exprs(src);
                let (hir, mut symtbl) = symtbl::resolve_symbols(hir);
                let tck = &mut typeck::typecheck(&hir, &mut symtbl).unwrap();
                let res = cc_exprs(scope, &ir);
                dbg!(&res);
                let expected_ir = compile_to_hir_exprs(expected);
                assert_eq!(
                    res,
                    expected_ir,
                    "exprs do not match.  Got:\n{}Expected:\n{}\n",
                    Expr::Block { body: res.clone() },
                    Expr::Block {
                        body: expected_ir.clone()
                    },
                );
            }
        }
        */
}
