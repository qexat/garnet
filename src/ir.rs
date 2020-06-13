//! Intermediate representation, basically what you get after all the typechecking and other kinds
//! of checking happen.  Slightly simplified over the AST -- basically the result of a
//! lowering/desugaring pass.  This might give us a nice place to do other simple
//! optimization-like things like constant folding, dead code detection, etc.
//!
//! An IR is always assumed to be a valid program, since it has passed all the checking stuff.
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
use crate::{TypeSym, VarSym};

/// Any expression.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Lit {
        val: Literal,
    },
    Var {
        name: VarSym,
    },
    BinOp {
        op: BOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    UniOp {
        op: UOp,
        rhs: Box<Expr>,
    },
    Block {
        body: Vec<Expr>,
    },
    Let {
        varname: VarSym,
        typename: TypeSym,
        init: Box<Expr>,
    },
    If {
        cases: Vec<(Expr, Vec<Expr>)>,
        falseblock: Vec<Expr>,
    },
    Loop {
        body: Vec<Expr>,
    },
    Lambda {
        signature: Signature,
        body: Vec<Expr>,
    },
    Funcall {
        func: Box<Expr>,
        params: Vec<Expr>,
    },
    Break,
    Return {
        retval: Box<Expr>,
    },
}

impl Expr {
    /// Shortcut function for making literal bools
    pub const fn bool(b: bool) -> Expr {
        Expr::Lit {
            val: Literal::Bool(b),
        }
    }

    /// Shortcut function for making literal integers
    pub const fn int(i: i64) -> Expr {
        Expr::Lit {
            val: Literal::Integer(i),
        }
    }

    /// Shortcut function for making literal unit
    pub const fn unit() -> Expr {
        Expr::Lit { val: Literal::Unit }
    }
}

/// A top-level declaration in the source file.
#[derive(Debug, Clone, PartialEq)]
pub enum Decl {
    Function {
        name: VarSym,
        signature: Signature,
        body: Vec<Expr>,
    },
    Const {
        name: VarSym,
        typename: TypeSym,
        init: Expr,
    },
}

/// A compilable chunk of AST.
///
/// Currently, basically a compilation unit.
#[derive(Debug, Clone, Default)]
pub struct Ir {
    pub decls: Vec<Decl>,
}

/// Transforms AST into IR, doing simplifying transformations on the way.
pub fn lower(ast: &ast::Ast) -> Ir {
    Ir {
        decls: lower_decls(&ast.decls),
    }
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
fn lower_expr(expr: &ast::Expr) -> Expr {
    use ast::Expr as E;
    use Expr::*;
    match expr {
        E::Lit { val } => Lit {
            val: lower_lit(val),
        },
        E::Var { name } => Var { name: *name },
        E::BinOp { op, lhs, rhs } => {
            let nop = lower_bop(op);
            let nlhs = lower_expr(lhs);
            let nrhs = lower_expr(rhs);
            BinOp {
                op: nop,
                lhs: Box::new(nlhs),
                rhs: Box::new(nrhs),
            }
        }
        E::UniOp { op, rhs } => {
            let nop = lower_uop(op);
            let nrhs = lower_expr(rhs);
            UniOp {
                op: nop,
                rhs: Box::new(nrhs),
            }
        }
        E::Block { body } => {
            let nbody = body.iter().map(lower_expr).collect();
            Block { body: nbody }
        }
        E::Let {
            varname,
            typename,
            init,
        } => {
            let ninit = Box::new(lower_expr(init));
            Let {
                varname: *varname,
                typename: typename.clone(),
                init: ninit,
            }
        }
        E::If { cases, falseblock } => {
            assert!(cases.len() > 0, "Should never happen");
            let cases = cases
                .iter()
                .map(|case| (lower_expr(&*case.condition), lower_exprs(&case.body)))
                .collect();
            let falseblock = lower_exprs(falseblock);
            If { cases, falseblock }
        }
        E::Loop { body } => {
            let nbody = lower_exprs(body);
            Loop { body: nbody }
        }
        E::Lambda { signature, body } => {
            let nsig = lower_signature(signature);
            let nbody = lower_exprs(body);
            Lambda {
                signature: nsig,
                body: nbody,
            }
        }
        E::Funcall { func, params } => {
            let nfunc = Box::new(lower_expr(func));
            let nparams = lower_exprs(params);
            Funcall {
                func: nfunc,
                params: nparams,
            }
        }
        E::Break => Break,
        E::Return { retval: None } => Return {
            // Return unit
            retval: Box::new(Expr::unit()),
        },
        E::Return { retval: Some(e) } => Return {
            retval: Box::new(lower_expr(e)),
        },
    }
}

/// handy shortcut to lower Vec<ast::Expr>
fn lower_exprs(exprs: &[ast::Expr]) -> Vec<Expr> {
    exprs.iter().map(lower_expr).collect()
}

fn lower_decl(decl: &ast::Decl) -> Decl {
    use ast::Decl as D;
    match decl {
        D::Function {
            name,
            signature,
            body,
        } => Decl::Function {
            name: *name,
            signature: lower_signature(signature),
            body: lower_exprs(body),
        },
        D::Const {
            name,
            typename,
            init,
        } => Decl::Const {
            name: *name,
            typename: *typename,
            init: lower_expr(init),
        },
    }
}

fn lower_decls(decls: &[ast::Decl]) -> Vec<Decl> {
    decls.iter().map(lower_decl).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Expr as A;
    use crate::ir::Expr as I;

    /// Does `return;` turn into `return ();`?
    #[test]
    fn test_return_none() {
        let input = A::Return { retval: None };
        let output = I::Return {
            retval: Box::new(I::unit()),
        };
        let res = lower_expr(&input);
        assert_eq!(&res, &output);
    }

    /// Does `return ();` also turn into `return ();`?
    #[test]
    fn test_return_unit() {
        let input = A::Return {
            retval: Some(Box::new(A::unit())),
        };
        let output = I::Return {
            retval: Box::new(I::unit()),
        };
        let res = lower_expr(&input);
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
        let output = I::If {
            cases: vec![(I::bool(false), vec![I::int(1)])],
            falseblock: vec![I::int(2)],
        };
        let res = lower_expr(&input);
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
        let output = I::If {
            cases: vec![
                (I::bool(false), vec![I::int(1)]),
                (I::bool(true), vec![I::int(2)]),
            ],
            falseblock: vec![I::int(3)],
        };
        let res = lower_expr(&input);
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
        let _ = lower_expr(&input);
    }
}
