//! Abstract syntax tree.
//!
//! Should be a *pretty exact* representation of the source code,
//! including things like parentheses and comments.  That way we can
//! eventually use the same structure for a code formatter and not
//! have it nuke anything.
//!
//! Though code formatters have different constraints and priorities, if they have line wrapping
//! and stuff at least.  So, it might not be a particularly great code formatter.

use crate::types::*;
use crate::*;

/// Literal value
#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    /// An integer of some kind
    Integer(i128),
    /// An integer with a known size and signedness
    SizedInteger {
        /// Literal value
        vl: i128,
        /// The size of the integer, in bytes
        bytes: u8,
        /// is_signed
        signed: bool,
    },
    /// A bool literal
    Bool(bool),
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Literal::Integer(i) => write!(f, "{}", i),
            Literal::SizedInteger { vl, bytes, signed } => {
                let size = bytes * 8;
                if *signed {
                    write!(f, "{}_I{}", vl, size)
                } else {
                    write!(f, "{}_U{}", vl, size)
                }
            }
            Literal::Bool(b) => write!(f, "{}", b),
        }
    }
}

/// Binary operation
#[allow(missing_docs)]
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

/// Unary operation
#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UOp {
    Neg,
    Not,
    Ref,
    Deref,
}

/// An arm in an `if ... elseif ... elseif ...` chain.
/// Includes the initial `if`.
#[derive(Debug, Clone, PartialEq)]
pub struct IfCase {
    /// the expr in `if expr then body ...`
    pub condition: ExprNode,
    /// The body of the if expr
    pub body: Vec<ExprNode>,
}

/// Any expression.
/// So, basically anything not a top-level decl.
#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Lit {
        val: Literal,
    },
    Var {
        name: Sym,
    },
    BinOp {
        op: BOp,
        lhs: ExprNode,
        rhs: ExprNode,
    },
    UniOp {
        op: UOp,
        rhs: ExprNode,
    },
    Block {
        body: Vec<ExprNode>,
    },
    Let {
        varname: Sym,
        typename: Option<Type>,
        init: ExprNode,
        mutable: bool,
    },
    If {
        cases: Vec<IfCase>,
        falseblock: Vec<ExprNode>,
    },
    Loop {
        body: Vec<ExprNode>,
    },
    While {
        cond: ExprNode,
        body: Vec<ExprNode>,
    },
    Lambda {
        signature: Signature,
        body: Vec<ExprNode>,
    },
    Funcall {
        func: ExprNode,
        params: Vec<ExprNode>,
        typeparams: Vec<Type>,
    },
    Break,
    Return {
        retval: ExprNode,
    },
    /// Tuple constructor
    TupleCtor {
        body: Vec<ExprNode>,
    },
    StructCtor {
        body: BTreeMap<Sym, ExprNode>,
    },
    ArrayCtor {
        body: Vec<ExprNode>,
    },
    /// Array element reference
    ArrayRef {
        expr: ExprNode,
        idx: ExprNode,
    },
    /// Tuple element reference
    TupleRef {
        expr: ExprNode,
        elt: usize,
    },
    /// Struct element reference
    StructRef {
        expr: ExprNode,
        elt: Sym,
    },
    /// Separate from a BinOp because its typechecking rules are different.
    Assign {
        lhs: ExprNode,
        rhs: ExprNode,
    },
    TypeUnwrap {
        expr: ExprNode,
    },
    Ref {
        expr: ExprNode,
    },
    Deref {
        expr: ExprNode,
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
    pub const fn int(i: i128) -> Expr {
        Expr::Lit {
            val: Literal::Integer(i),
        }
    }

    /// Shortcut function for making literal integers of a known size
    pub const fn sized_int(i: i128, bytes: u8, signed: bool) -> Expr {
        Expr::Lit {
            val: Literal::SizedInteger {
                vl: i,
                bytes,
                signed,
            },
        }
    }

    /// Shortcut function for making literal unit
    pub fn unit() -> Expr {
        Expr::TupleCtor { body: vec![] }
    }

    /// Shortcuts for making vars
    pub fn var(name: &str) -> Expr {
        Expr::Var {
            name: Sym::new(name),
        }
    }
}

/// An expression node with source info
#[derive(Debug, Clone, PartialEq)]
pub struct ExprNode {
    /// expression
    pub e: Box<Expr>,
}

impl ExprNode {
    pub fn new(e: Expr) -> Self {
        Self { e: Box::new(e) }
    }

    /// Shortcut function for making literal bools
    pub fn bool(b: bool) -> Self {
        Self::new(Expr::Lit {
            val: Literal::Bool(b),
        })
    }

    /// Shortcut function for making literal integers
    pub fn int(i: i128) -> Self {
        Self::new(Expr::Lit {
            val: Literal::Integer(i),
        })
    }

    /// Shortcut function for making literal integers of a known size
    pub fn sized_int(i: i128, bytes: u8, signed: bool) -> Self {
        Self::new(Expr::Lit {
            val: Literal::SizedInteger {
                vl: i,
                bytes,
                signed,
            },
        })
    }

    /// Shortcut function for making literal unit
    pub fn unit() -> Self {
        Self::new(Expr::TupleCtor { body: vec![] })
    }

    /// Shortcuts for making vars
    pub fn var(name: &str) -> Self {
        Self::new(Expr::Var {
            name: Sym::new(name),
        })
    }
}

/// A top-level declaration in the source file.
#[derive(Debug, Clone, PartialEq)]
pub enum Decl {
    Function {
        name: Sym,
        signature: Signature,
        body: Vec<ExprNode>,
        doc_comment: Vec<String>,
    },
    Const {
        name: Sym,
        typename: Type,
        init: ExprNode,
        doc_comment: Vec<String>,
    },
    TypeDef {
        name: Sym,
        params: Vec<Sym>,
        typedecl: Type,
        doc_comment: Vec<String>,
    },
    Import {
        name: Sym,
        rename: Option<Sym>,
    },
}

/// A compilable chunk of AST.
///
/// Currently, basically a compilation unit.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Ast {
    pub decls: Vec<Decl>,
    pub filename: String,
    pub modulename: String,
    pub module_docstring: String,
}
