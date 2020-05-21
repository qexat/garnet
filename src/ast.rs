//! Abstract syntax tree.
//!
//! Should be a *pretty exact* representation of the source code,
//! including things like parentheses and comments.  That way we can
//! eventually use the same structure for a code formatter and not
//! have it nuke anything.

use crate::{TypeDef, TypeSym, VarSym};

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Integer(i64),
    Bool(bool),
    Unit,
}

/// Binary operation
#[derive(Debug, Clone, PartialEq)]
pub enum BOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
}

impl BOp {
    /// Returns the type that the bin op operates on.
    /// Currently, only numbers.
    pub fn type_of(&self, cx: &mut crate::Cx) -> TypeDef {
        TypeDef::SInt(4)
    }
}

/// Unary operation
#[derive(Debug, Clone, PartialEq)]
pub enum UOp {
    Neg,
}

impl UOp {
    /// Returns the type that the unary op operates on.
    /// Currently, only numbers.
    pub fn type_of(&self, cx: &mut crate::Cx) -> TypeDef {
        TypeDef::SInt(4)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct IfCase {
    pub condition: Box<Expr>,
    pub body: Vec<Expr>,
}

/// A function type signature
#[derive(Debug, Clone, PartialEq)]
pub struct Signature {
    pub params: Vec<(VarSym, TypeDef)>,
    pub rettype: TypeDef,
}

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
        typename: TypeDef,
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
        typedef: TypeDef,
        init: Expr,
    },
}

/// A compilable chunk of AST.
///
/// Currently, basically a compilation unit.
#[derive(Debug, Clone, Default)]
pub struct Ast {
    pub decls: Vec<Decl>,
}
