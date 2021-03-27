//! Intermediate representation, basically what you get after immediate parsing and where
//! all the typechecking and other kinds of checking happen.  Slightly simplified over the AST --
//! basically the result of a lowering/desugaring pass.  This might give us a nice place to do
//! other simple optimization-like things like constant folding, dead code detection, etc.
//!
//! So... do we want it to be an expression tree like AST is, or SSA form, or a control-
//! flow graph, or what?  Ponder this, since different things are easy to do on different
//! representations. ...well, the FIRST thing we need to do is type checking and related
//! junk anyway, so, this should just be a slightly-lowered syntax tree.
//!
//! Currently it's not really lowered by much!  If's and maybe loops or something.
//! It's mostly a layer of indirection for further stuff to happen to.

use crate::ast::{self};
pub use crate::ast::{BOp, IfCase, Literal, Signature, UOp};
use crate::*;

/// An expression with an optional type annotation.
/// Currently will be () for something that hasn't been
/// typechecked, or a TypeSym with the appropriate type
/// after type checking has been done.
#[derive(Debug, Clone, PartialEq)]
pub struct TypedExpr<T> {
    /// type
    pub t: T,
    /// expression
    pub e: Expr<T>,
}

impl<T> TypedExpr<T> {
    /// TODO: This is less useful than it should be,
    /// I was kinda imagining it as a general purpose
    /// transformer/visitor but it can only correctly transform
    /// leaf nodes.  Hnyrn.
    pub(crate) fn map<T2>(self, new_t: T2) -> TypedExpr<T2>
    where
        T2: Copy,
    {
        TypedExpr {
            t: new_t,
            e: self.e.map(new_t),
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
        typename: TypeSym,
        init: Box<TypedExpr<T>>,
        mutable: bool,
    },
    If {
        cases: Vec<(TypedExpr<T>, Vec<TypedExpr<T>>)>,
        falseblock: Vec<TypedExpr<T>>,
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
    TupleRef {
        expr: Box<TypedExpr<T>>,
        elt: usize,
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
    pub fn unit() -> Self {
        Self::TupleCtor { body: vec![] }
    }

    /// Again, general purpose visitor that's only
    /// actually used in a couple places...
    pub(crate) fn map<T2>(self, new_t: T2) -> Expr<T2>
    where
        T2: Copy,
    {
        use Expr as E;
        let map_vec =
            |body: Vec<TypedExpr<T>>| body.into_iter().map(|e| TypedExpr::map(e, new_t)).collect();
        match self {
            E::Lit { val } => E::Lit { val },
            E::Var { name } => E::Var { name },
            E::BinOp { op, lhs, rhs } => E::BinOp {
                op,
                lhs: Box::new(lhs.map(new_t)),
                rhs: Box::new(rhs.map(new_t)),
            },
            E::UniOp { op, rhs } => E::UniOp {
                op,
                rhs: Box::new(rhs.map(new_t)),
            },
            E::Block { body } => E::Block {
                body: map_vec(body),
            },
            E::Let {
                varname,
                typename,
                init,
                mutable,
            } => E::Let {
                varname,
                typename,
                init: Box::new(init.map(new_t)),
                mutable,
            },
            E::If { cases, falseblock } => {
                let new_cases = cases
                    .into_iter()
                    .map(|(c, bod)| (c.map(new_t), map_vec(bod)))
                    .collect();
                E::If {
                    cases: new_cases,
                    falseblock: map_vec(falseblock),
                }
            }
            E::Loop { body } => E::Loop {
                body: map_vec(body),
            },
            E::Lambda { signature, body } => E::Lambda {
                signature,
                body: map_vec(body),
            },
            E::Funcall { func, params } => E::Funcall {
                func: Box::new(func.map(new_t)),
                params: map_vec(params),
            },
            E::Break => E::Break,
            E::Return { retval } => E::Return {
                retval: Box::new(retval.map(new_t)),
            },
            E::TupleCtor { body } => E::TupleCtor {
                body: map_vec(body),
            },
            E::TupleRef { expr, elt } => E::TupleRef {
                expr: Box::new(expr.map(new_t)),
                elt,
            },
            E::Assign { lhs, rhs } => E::Assign {
                lhs: Box::new(lhs.map(new_t)),
                rhs: Box::new(rhs.map(new_t)),
            },
            E::Deref { expr } => E::Deref {
                expr: Box::new(expr.map(new_t)),
            },
            E::Ref { expr } => E::Ref {
                expr: Box::new(expr.map(new_t)),
            },
        }
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
/// for our type info attached to the HIR node.  For real code it generally generates
/// a new `InfTypeSym` saying "we don't know anything about this type yet", but
/// for test code it's also useful to make it return unit, and after inference
/// I think we're going to just reify our types by turning them all into concrete
/// `TypeSym`'s...  zomg.
pub fn lower<T>(f: &mut dyn FnMut() -> T, ast: &ast::Ast) -> Ir<T> {
    lower_decls(f, &ast.decls)
}

fn lower_lit(lit: &ast::Literal) -> Literal {
    lit.clone()
}

fn lower_bop(bop: &ast::BOp) -> BOp {
    bop.clone()
}

fn lower_uop(uop: &ast::UOp) -> UOp {
    uop.clone()
}

fn lower_signature(sig: &ast::Signature) -> Signature {
    sig.clone()
}

/// This is the biggie currently
fn lower_expr<T>(f: &mut dyn FnMut() -> T, expr: &ast::Expr) -> TypedExpr<T> {
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
                typename: typename.clone(),
                init: ninit,
                mutable: *mutable,
            }
        }
        E::If { cases, falseblock } => {
            assert!(cases.len() > 0, "Should never happen");
            let cases = cases
                .iter()
                .map(|case| (lower_expr(f, &*case.condition), lower_exprs(f, &case.body)))
                .collect();
            let falseblock = lower_exprs(f, falseblock);
            If { cases, falseblock }
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
        E::TupleRef { expr, elt } => Expr::TupleRef {
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
    let t = f();
    TypedExpr { t, e: new_exp }
}

/// handy shortcut to lower Vec<ast::Expr>
fn lower_exprs<T>(f: &mut dyn FnMut() -> T, exprs: &[ast::Expr]) -> Vec<TypedExpr<T>> {
    exprs.iter().map(|e| lower_expr(f, e)).collect()
}

/// Lower an AST decl to IR.
fn lower_decl<T>(f: &mut dyn FnMut() -> T, decl: &ast::Decl) -> Decl<T> {
    use ast::Decl as D;
    match decl {
        D::Function {
            name,
            signature,
            body,
            ..
        } => Decl::Function {
            name: *name,
            signature: lower_signature(signature),
            body: lower_exprs(f, body),
        },
        D::Const {
            name,
            typename,
            init,
            ..
        } => Decl::Const {
            name: *name,
            typename: *typename,
            init: lower_expr(f, init),
        },
    }
}

fn lower_decls<T>(f: &mut dyn FnMut() -> T, decls: &[ast::Decl]) -> Ir<T> {
    Ir {
        decls: decls.iter().map(|d| lower_decl(f, d)).collect(),
    }
}

/// Shortcut to take an Expr and wrap it
/// in a TypedExpr with a unit type.
///
/// TODO: Better name?
pub(crate) fn plz(e: Expr<()>) -> Box<TypedExpr<()>> {
    Box::new(TypedExpr { t: (), e: e })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Expr as A;
    use crate::hir::Expr as I;

    /// High on the "I can't believe this is working" scale.
    fn rly() {
        ()
    }

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
            cases: vec![(*plz(I::bool(false)), vec![*plz(I::int(1))])],
            falseblock: vec![*plz(I::int(2))],
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
            ],
            falseblock: vec![*plz(I::int(3))],
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
