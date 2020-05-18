//! Abstract syntax tree.
//!
//! Should be a *pretty exact* representation of the source code,
//! including things like parentheses and comments.  That way we can
//! eventually use the same structure for a code formatter and not
//! have it nuke anything.

use crate::intern::Sym;

#[derive(Debug, Clone)]
pub enum Literal {
    Integer(i64),
    Bool(bool),
    Unit,
}

#[derive(Debug, Clone)]
pub struct Symbol(pub Sym);

/// Binary operation
#[derive(Debug, Clone)]
pub enum BOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
}

/// Unary operation
#[derive(Debug, Clone)]
pub enum UOp {
    Neg,
    Deref,
}

#[derive(Debug, Clone)]
pub struct IfCase {
    pub condition: Box<Expr>,
    pub body: Vec<Expr>,
}

#[derive(Debug, Clone)]
pub struct Type {
    pub name: Symbol,
}

/// A function type signature
#[derive(Debug, Clone)]
pub struct Signature {
    pub params: Vec<(Symbol, Type)>,
    pub rettype: Type,
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
        cases: Vec<IfCase>,
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
        retval: Option<Box<Expr>>,
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
pub struct Ast {
    pub decls: Vec<Decl>,
}
