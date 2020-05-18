//! Intermediate representation, basically what you get after all the typechecking and other kinds
//! of checking happen.  Slightly simplified over the AST -- basically the result of a
//! lowering/desugaring pass.  This might give us a nice place to do other simple
//! lowering/optimization-like things like constant folding, dead code detection, etc.
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

use crate::intern::Sym;

use crate::ast::{self, BOp, Literal, Signature, Type, UOp};

#[derive(Debug, Clone)]
pub struct Symbol(pub Sym);

impl From<&ast::Symbol> for Symbol {
    fn from(s: &ast::Symbol) -> Self {
        Symbol(s.0)
    }
}

/// Any expression.
#[derive(Debug, Clone)]
pub enum Expr {
    Lit {
        val: Literal,
    },
    Var {
        name: Symbol,
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
        varname: Symbol,
        typename: Type,
        init: Box<Expr>,
    },
    If {
        condition: Box<Expr>,
        trueblock: Vec<Expr>,
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

/// A top-level declaration in the source file.
#[derive(Debug, Clone)]
pub enum Decl {
    Function {
        name: Symbol,
        signature: Signature,
        body: Vec<Expr>,
    },
}

/// A compilable chunk of AST.
///
/// Currently, basically a compilation unit.
#[derive(Debug, Clone, Default)]
pub struct Ir {
    decls: Vec<Decl>,
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

fn lower_symbol(sym: &ast::Symbol) -> Symbol {
    Symbol::from(sym)
}

fn lower_bop(bop: &ast::BOp) -> BOp {
    bop.clone()
}

fn lower_uop(uop: &ast::UOp) -> UOp {
    uop.clone()
}

fn lower_type(ty: &ast::Type) -> Type {
    ty.clone()
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
        E::Var { name } => Var {
            name: lower_symbol(name),
        },
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
            let nvarname = lower_symbol(varname);
            let ntypename = lower_type(typename);
            let ninit = Box::new(lower_expr(init));
            Let {
                varname: nvarname,
                typename: ntypename,
                init: ninit,
            }
        }
        E::If { cases, falseblock } => {
            // Expand cases out into nested if ... else if ... else if ... else
            fn unheck_if(ifcases: &[ast::IfCase], elsecase: &[ast::Expr]) -> Expr {
                match ifcases {
                    [] => panic!("If statement with no condition; should never happen"),
                    [single] => {
                        let nelsecase = lower_exprs(elsecase);
                        If {
                            condition: Box::new(lower_expr(&*single.condition)),
                            trueblock: lower_exprs(single.body.as_slice()),
                            falseblock: nelsecase,
                        }
                    }
                    [itm, rst @ ..] => {
                        let res = unheck_if(rst, elsecase);
                        If {
                            condition: Box::new(lower_expr(&*itm.condition)),
                            trueblock: lower_exprs(itm.body.as_slice()),
                            falseblock: vec![res],
                        }
                    }
                }
            }
            unheck_if(cases.as_slice(), falseblock.as_slice())
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
            retval: Box::new(Expr::Lit { val: Literal::Unit }),
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
            name: lower_symbol(name),
            signature: lower_signature(signature),
            body: lower_exprs(body),
        },
    }
}

fn lower_decls(decls: &[ast::Decl]) -> Vec<Decl> {
    decls.iter().map(lower_decl).collect()
}

#[cfg(test)]
mod tests {}
