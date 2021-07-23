//! Intermediate representation, basically what you get after immediate parsing and where
//! all the typechecking and other kinds of checking happen.  Slightly simplified over the AST --
//! basically the result of a lowering/desugaring pass.  This might give us a nice place to do
//! other simple optimization-like things like constant folding, dead code detection, etc.
//!
//! Currently it's not really lowered by much!  If's and maybe loops or something.
//! It's mostly a layer of indirection for further stuff to happen to, so we can change
//! the parser without changing the typecheckin and such.

use crate::ast::{self};
pub use crate::ast::{BOp, IfCase, Literal, Signature, UOp};
use crate::*;

/// An expression with a type annotation.
/// Currently will be () for something that hasn't been
/// typechecked, or a `TypeSym` with the appropriate type
/// after type checking has been done.
#[derive(Debug, Clone, PartialEq)]
pub struct TypedExpr<T> {
    /// type
    pub t: T,
    /// expression
    pub e: Expr<T>,
}

impl<T> TypedExpr<T> {
    /// Takes a function that transforms a typedexpr and applies it to every single node
    /// in the expr tree.
    ///
    /// BUGGO: The function should really only transform the node it's called on,
    /// not recurse to alter its children...????????????
    ///
    /// Really?  If so then it's super easy to do, we just change the `t` field in the typedexpr.
    /// The whole point of this is to recurse.  But it can't recurse down the whole expression tree
    /// because there is probably stuff in there that needs something other than a simple pure
    /// function to decide on its type.
    ///
    /// This is really only used in a few odd places, now that I actually look at it...
    /// Investigate more.
    pub(crate) fn map_type<T2>(&self, f: &impl Fn(&T) -> T2) -> TypedExpr<T2> {
        use Expr::*;
        let map_vec = |e: &Vec<TypedExpr<T>>| e.iter().map(|te| te.map_type(f)).collect::<Vec<_>>();
        let new_e = match &self.e {
            Lit { val } => Lit { val: val.clone() },
            Var { name } => Var { name: *name },
            UniOp { op, rhs } => UniOp {
                op: *op,
                rhs: Box::new(rhs.map_type(f)),
            },
            BinOp { op, lhs, rhs } => BinOp {
                op: *op,
                lhs: Box::new(lhs.map_type(f)),
                rhs: Box::new(rhs.map_type(f)),
            },
            Block { body } => Block {
                body: map_vec(body),
            },
            Let {
                varname,
                typename,
                init,
                mutable,
            } => Let {
                varname: *varname,
                typename: *typename,
                init: Box::new(init.map_type(f)),
                mutable: *mutable,
            },
            If { cases } => {
                let new_cases = cases
                    .iter()
                    .map(|(c, bod)| (c.map_type(f), map_vec(bod)))
                    .collect();
                If { cases: new_cases }
            }
            Loop { body } => Loop {
                body: map_vec(body),
            },
            Lambda { signature, body } => Lambda {
                signature: signature.clone(),
                body: map_vec(body),
            },
            Funcall { func, params } => Funcall {
                func: Box::new(func.map_type(f)),
                params: map_vec(params),
            },
            Break => Break,
            Return { retval } => Return {
                retval: Box::new(retval.map_type(f)),
            },
            TupleCtor { body } => TupleCtor {
                body: map_vec(body),
            },
            StructCtor { name, body } => {
                let body = body.iter().map(|(nm, vl)| (*nm, vl.map_type(f))).collect();
                StructCtor { name: *name, body }
            }
            TupleRef { expr, elt } => TupleRef {
                expr: Box::new(expr.map_type(f)),
                elt: *elt,
            },
            StructRef { expr, elt } => StructRef {
                expr: Box::new(expr.map_type(f)),
                elt: *elt,
            },
            Assign { lhs, rhs } => Assign {
                lhs: Box::new(lhs.map_type(f)),
                rhs: Box::new(rhs.map_type(f)),
            },
            Deref { expr } => Deref {
                expr: Box::new(expr.map_type(f)),
            },
            Ref { expr } => Ref {
                expr: Box::new(expr.map_type(f)),
            },
        };
        TypedExpr {
            e: new_e,
            t: f(&self.t),
        }
    }
}

/// An expression.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr<T> {
    Lit {
        val: Literal,
    },
    Var {
        name: VarSym,
    },
    BinOp {
        op: BOp,
        lhs: Box<TypedExpr<T>>,
        rhs: Box<TypedExpr<T>>,
    },
    UniOp {
        op: UOp,
        rhs: Box<TypedExpr<T>>,
    },
    Block {
        body: Vec<TypedExpr<T>>,
    },
    Let {
        varname: VarSym,
        typename: Option<TypeSym>,
        init: Box<TypedExpr<T>>,
        mutable: bool,
    },
    If {
        cases: Vec<(TypedExpr<T>, Vec<TypedExpr<T>>)>,
    },
    Loop {
        body: Vec<TypedExpr<T>>,
    },
    Lambda {
        signature: Signature,
        body: Vec<TypedExpr<T>>,
    },
    Funcall {
        func: Box<TypedExpr<T>>,
        params: Vec<TypedExpr<T>>,
    },
    Break,
    Return {
        retval: Box<TypedExpr<T>>,
    },
    TupleCtor {
        body: Vec<TypedExpr<T>>,
    },
    StructCtor {
        name: VarSym,
        body: Vec<(VarSym, TypedExpr<T>)>,
    },
    TupleRef {
        expr: Box<TypedExpr<T>>,
        elt: usize,
    },
    StructRef {
        expr: Box<TypedExpr<T>>,
        elt: VarSym,
    },
    Assign {
        lhs: Box<TypedExpr<T>>,
        rhs: Box<TypedExpr<T>>,
    },
    Deref {
        expr: Box<TypedExpr<T>>,
    },
    Ref {
        expr: Box<TypedExpr<T>>,
    },
}

impl<T> Expr<T> {
    /// Shortcut function for making literal bools
    pub const fn bool(b: bool) -> Self {
        Self::Lit {
            val: Literal::Bool(b),
        }
    }

    /// Shortcut function for making literal integers
    pub const fn int(i: i128) -> Self {
        Self::Lit {
            val: Literal::Integer(i),
        }
    }

    /// Shortcut function for making literal unit
    pub const fn unit() -> Self {
        Self::TupleCtor { body: vec![] }
    }
}

/// A top-level declaration in the source file.
/// Like TypedExpr, contains a type annotation.
#[derive(Debug, Clone, PartialEq)]
pub enum Decl<T> {
    Function {
        name: VarSym,
        signature: Signature,
        body: Vec<TypedExpr<T>>,
    },
    Const {
        name: VarSym,
        typename: TypeSym,
        init: TypedExpr<T>,
    },
    TypeDef {
        name: VarSym,
        typedecl: TypeSym,
    },
    StructDef {
        name: VarSym,
        fields: Vec<(VarSym, TypeSym)>,
    },
    /// Our first compiler intrinsic!  \o/
    ///
    /// This is a function that is generated along with
    /// each typedef, which takes the args of that typedef
    /// and produces that type.  We have no way to write such
    /// a function by hand, so here we are.  In all other ways
    /// though we treat it exactly like a function though.
    Constructor {
        name: VarSym,
        signature: Signature,
    },
}

/// A compilable chunk of IR.
///
/// Currently, basically a compilation unit.
///
/// The T is a type annotation of some kind, but we really don't care what.
#[derive(Debug, Clone, Default)]
pub struct Ir<T> {
    pub decls: Vec<Decl<T>>,
}

/// Transforms AST into IR
///
/// The function `f` is a function that should generate whatever value we need
/// for our type info attached to the HIR node.  To start with it's a unit, 'cause
/// we have no type information.
pub fn lower<T>(f: &mut dyn FnMut(&hir::Expr<T>) -> T, ast: &ast::Ast) -> Ir<T> {
    lower_decls(f, &ast.decls)
}

fn lower_lit(lit: &ast::Literal) -> Literal {
    lit.clone()
}

fn lower_bop(bop: &ast::BOp) -> BOp {
    *bop
}

fn lower_uop(uop: &ast::UOp) -> UOp {
    *uop
}

fn lower_signature(sig: &ast::Signature) -> Signature {
    sig.clone()
}

/// This is the biggie currently
fn lower_expr<T>(f: &mut dyn FnMut(&hir::Expr<T>) -> T, expr: &ast::Expr) -> TypedExpr<T> {
    use ast::Expr as E;
    use Expr::*;
    let new_exp = match expr {
        E::Lit { val } => Lit {
            val: lower_lit(val),
        },
        E::Var { name } => Var { name: *name },
        E::BinOp { op, lhs, rhs } => {
            let nop = lower_bop(op);
            let nlhs = lower_expr(f, lhs);
            let nrhs = lower_expr(f, rhs);
            BinOp {
                op: nop,
                lhs: Box::new(nlhs),
                rhs: Box::new(nrhs),
            }
        }
        E::UniOp { op, rhs } => {
            let nop = lower_uop(op);
            let nrhs = lower_expr(f, rhs);
            UniOp {
                op: nop,
                rhs: Box::new(nrhs),
            }
        }
        E::Block { body } => {
            let nbody = body.iter().map(|e| lower_expr(f, e)).collect();
            Block { body: nbody }
        }
        E::Let {
            varname,
            typename,
            init,
            mutable,
        } => {
            let ninit = Box::new(lower_expr(f, init));
            Let {
                varname: *varname,
                typename: *typename,
                init: ninit,
                mutable: *mutable,
            }
        }
        E::If { cases, falseblock } => {
            assert!(!cases.is_empty(), "Should never happen");
            let mut cases: Vec<_> = cases
                .iter()
                .map(|case| (lower_expr(f, &*case.condition), lower_exprs(f, &case.body)))
                .collect();
            // Add the "else" case, which we can just make `if true then...`
            // No idea if this is a good idea, but it makes life easier right
            // this instant, so.  Hasn't bit me yet, so it's not a *bad* idea.
            let else_case = E::Lit {
                val: ast::Literal::Bool(true),
            };
            let nelse_case = lower_expr(f, &else_case);
            // Empty false block becomes a false block that returns unit
            let false_exprs = if falseblock.len() == 0 {
                lower_exprs(f, &vec![ast::Expr::TupleCtor { body: vec![] }])
            } else {
                lower_exprs(f, falseblock)
            };
            cases.push((nelse_case, false_exprs));
            If { cases }
        }
        E::Loop { body } => {
            let nbody = lower_exprs(f, body);
            Loop { body: nbody }
        }
        E::Lambda { signature, body } => {
            let nsig = lower_signature(signature);
            let nbody = lower_exprs(f, body);
            Lambda {
                signature: nsig,
                body: nbody,
            }
        }
        E::Funcall { func, params } => {
            let nfunc = Box::new(lower_expr(f, func));
            let nparams = lower_exprs(f, params);
            Funcall {
                func: nfunc,
                params: nparams,
            }
        }
        E::Break => Break,
        E::Return { retval: e } => Return {
            retval: Box::new(lower_expr(f, e)),
        },
        E::TupleCtor { body } => Expr::TupleCtor {
            body: lower_exprs(f, body),
        },
        E::StructCtor { name, body } => {
            let lowered_body = body
                .iter()
                .map(|(nm, expr)| (*nm, lower_expr(f, expr)))
                .collect();
            Expr::StructCtor {
                name: *name,
                body: lowered_body,
            }
        }
        E::TupleRef { expr, elt } => Expr::TupleRef {
            expr: Box::new(lower_expr(f, expr)),
            elt: *elt,
        },
        E::StructRef { expr, elt } => Expr::StructRef {
            expr: Box::new(lower_expr(f, expr)),
            elt: *elt,
        },
        E::Deref { expr } => Expr::Deref {
            expr: Box::new(lower_expr(f, expr)),
        },
        E::Ref { expr } => Expr::Ref {
            expr: Box::new(lower_expr(f, expr)),
        },
        E::Assign { lhs, rhs } => Expr::Assign {
            lhs: Box::new(lower_expr(f, lhs)),
            rhs: Box::new(lower_expr(f, rhs)),
        },
    };
    let t = f(&new_exp);
    TypedExpr { t, e: new_exp }
}

/// handy shortcut to lower Vec<ast::Expr>
fn lower_exprs<T>(f: &mut dyn FnMut(&hir::Expr<T>) -> T, exprs: &[ast::Expr]) -> Vec<TypedExpr<T>> {
    exprs.iter().map(|e| lower_expr(f, e)).collect()
}

/// Lower an AST decl to IR.
///
/// Is there a more elegant way of doing this than passing an accumulator?
/// Returning a vec is lame.  Return an iterator?  Sounds like work.
fn lower_decl<T>(accm: &mut Vec<Decl<T>>, f: &mut dyn FnMut(&hir::Expr<T>) -> T, decl: &ast::Decl) {
    use ast::Decl as D;
    match decl {
        D::Function {
            name,
            signature,
            body,
            ..
        } => accm.push(Decl::Function {
            name: *name,
            signature: lower_signature(signature),
            body: lower_exprs(f, body),
        }),
        D::Const {
            name,
            typename,
            init,
            ..
        } => accm.push(Decl::Const {
            name: *name,
            typename: *typename,
            init: lower_expr(f, init),
        }),
        // this needs to generate the typedef AND the type constructor
        // declaration.  Plus the type deconstructor.
        D::TypeDef { name, typedecl, .. } => {
            accm.push(Decl::TypeDef {
                name: *name,
                typedecl: *typedecl,
            });
            let paramname = INT.intern("input");
            let rtype = INT.intern_type(&TypeDef::Named(*name));
            accm.push(Decl::Constructor {
                name: *name,
                signature: hir::Signature {
                    params: vec![(paramname, *typedecl)],
                    rettype: rtype,
                },
            });
        }
        D::StructDef { name, fields, .. } => accm.push(Decl::StructDef {
            name: *name,
            fields: fields.clone(),
        }),
    }
}

fn lower_decls<T>(f: &mut dyn FnMut(&hir::Expr<T>) -> T, decls: &[ast::Decl]) -> Ir<T> {
    let mut accm = Vec::with_capacity(decls.len() * 2);
    for d in decls.iter() {
        lower_decl(&mut accm, f, d)
    }
    Ir { decls: accm }
}

/// Shortcut to take an Expr and wrap it
/// in a TypedExpr with a unit type.
///
/// TODO: Better name?
pub(crate) fn plz(e: Expr<()>) -> Box<TypedExpr<()>> {
    Box::new(TypedExpr { t: (), e })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Expr as A;
    use crate::hir::Expr as I;
    use crate::testutil::*;

    /*
    /// Does `return;` turn into `return ();`?
    /// Not doing that to make parsing simpler
    #[test]
    fn test_return_none() {
        let input = A::Return { retval: None };
        let output = *plz(I::Return {
            retval: plz(I::unit()),
        });
        let res = lower_expr(&input);
        assert_eq!(&res, &output);
    }
    */

    /// Does `return ();` also turn into `return ();`?
    #[test]
    fn test_return_unit() {
        let input = A::Return {
            retval: Box::new(A::unit()),
        };
        let output = *plz(I::Return {
            retval: plz(I::unit()),
        });
        let res = lower_expr(&mut rly, &input);
        assert_eq!(&res, &output);
    }

    /// Do we turn chained if's properly into nested ones?
    /// Single if.
    #[test]
    fn test_if_lowering_single() {
        let input = A::If {
            cases: vec![ast::IfCase {
                condition: Box::new(A::bool(false)),
                body: vec![A::int(1)],
            }],
            falseblock: vec![A::int(2)],
        };
        let output = *plz(I::If {
            cases: vec![
                (*plz(I::bool(false)), vec![*plz(I::int(1))]),
                (*plz(I::bool(true)), vec![*plz(I::int(2))]),
            ],
        });
        let res = lower_expr(&mut rly, &input);
        assert_eq!(&res, &output);
    }

    /// Do we turn chained if's properly into nested ones?
    /// Chained if/else's
    #[test]
    fn test_if_lowering_chained() {
        let input = A::If {
            cases: vec![
                ast::IfCase {
                    condition: Box::new(A::bool(false)),
                    body: vec![A::int(1)],
                },
                ast::IfCase {
                    condition: Box::new(A::bool(true)),
                    body: vec![A::int(2)],
                },
            ],
            falseblock: vec![A::int(3)],
        };
        let output = *plz(I::If {
            cases: vec![
                (*plz(I::bool(false)), vec![*plz(I::int(1))]),
                (*plz(I::bool(true)), vec![*plz(I::int(2))]),
                (*plz(I::bool(true)), vec![*plz(I::int(3))]),
            ],
        });
        let res = lower_expr(&mut rly, &input);
        assert_eq!(&res, &output);
    }

    /// Do we panic if we get an impossible if with no cases?
    #[test]
    #[should_panic]
    fn test_if_nothing() {
        let input = A::If {
            cases: vec![],
            falseblock: vec![],
        };
        let _ = lower_expr(&mut rly, &input);
    }
}
