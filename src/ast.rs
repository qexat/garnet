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
}

/// Binary operation
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum BOp {
    // Math
    Add,
    Sub,
    Mul,
    Div,
    Mod,

    // Comparison
    Eq,
    Neq,
    Gt,
    Lt,
    Gte,
    Lte,

    //Logic
    And,
    Or,
    Xor,
}

impl BOp {
    /// Returns the type that the bin op operates on.
    pub fn input_type(&self, cx: &mut crate::Cx) -> TypeSym {
        use BOp::*;
        match self {
            And | Or | Xor => cx.bool(),
            _ => cx.i32(),
        }
    }

    /// What the resultant type of the binop is
    pub fn output_type(&self, cx: &mut crate::Cx) -> TypeSym {
        use BOp::*;
        match self {
            Add | Sub | Mul | Div | Mod => cx.i32(),
            Eq | Neq | Gt | Lt | Gte | Lte => cx.bool(),
            And | Or | Xor => cx.bool(),
        }
    }
}

/// Unary operation
#[derive(Debug, Clone, PartialEq)]
pub enum UOp {
    Neg,
    Not,
}

impl UOp {
    /// Returns the type that the unary op operates on.
    /// Currently, only numbers.
    pub fn input_type(&self, cx: &mut crate::Cx) -> TypeSym {
        use UOp::*;
        match self {
            Neg => cx.i32(),
            Not => cx.bool(),
        }
    }

    /// What the resultant type of the uop is
    pub fn output_type(&self, cx: &mut crate::Cx) -> TypeSym {
        use UOp::*;
        match self {
            Neg => cx.i32(),
            Not => cx.bool(),
        }
    }
}

/// An arm in an `if ... elseif ... elseif ...` chain.
/// Includes the initial `if`.
#[derive(Debug, Clone, PartialEq)]
pub struct IfCase {
    pub condition: Box<Expr>,
    pub body: Vec<Expr>,
}

/// A function type signature
#[derive(Debug, Clone, PartialEq)]
pub struct Signature {
    pub params: Vec<(VarSym, TypeSym)>,
    pub rettype: TypeSym,
}

impl Signature {
    /// Returns a lambda typedef representing the signature
    pub(crate) fn to_type(&self, cx: &crate::Cx) -> TypeSym {
        let args = self.params.iter().map(|(_v, t)| *t).collect();
        let t = TypeDef::Lambda(args, self.rettype);
        cx.intern_type(&t)
    }
}

/// Any expression.
/// So, basically anything not a top-level type decl.
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
    /// Tuple constructor
    TupleCtor {
        body: Vec<Expr>,
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
    pub fn unit() -> Expr {
        Expr::TupleCtor { body: vec![] }
    }

    /// Shortcuts for making vars
    pub fn var(cx: &crate::Cx, name: &str) -> Expr {
        Expr::Var {
            name: cx.intern(name),
        }
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
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Ast {
    pub decls: Vec<Decl>,
}
