//! Abstract syntax tree.
//!
//! Should be a *pretty exact* representation of the source code,
//! including things like parentheses and comments.  That way we can
//! eventually use the same structure for a code formatter and not
//! have it nuke anything.
//!
//! Though code formatters have different constraints and priorities, if they have line wrapping
//! and stuff at least.  So, it might not be a particularly great code formatter.

use crate::*;

/// Literal value
#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    /// An integer of some kind
    Integer(i128),
    /// An integer with a known size
    SizedInteger {
        /// Literal value
        vl: i128,
        /// The size of the integer, in bytes
        bytes: u8,
    },
    /// A bool literal
    Bool(bool),
    /// Enum literal
    EnumLit(Sym, Sym),
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
    pub condition: Box<Expr>,
    /// The body of the if expr
    pub body: Vec<Expr>,
}

/// A function type signature
#[derive(Debug, Clone, PartialEq)]
pub struct Signature {
    /// Parameters
    pub params: Vec<(Sym, Type)>,
    /// Return type
    pub rettype: Type,
}

impl Signature {
    /// Returns a lambda typedef representing the signature
    pub fn to_type(&self) -> Type {
        let paramtypes = self.params.iter().map(|(_nm, ty)| ty.clone()).collect();
        Type::Func(paramtypes, Box::new(self.rettype.clone()))
    }

    /// Get all the generic params out of this function sig
    pub fn generic_type_names(&self) -> Vec<Sym> {
        self.to_type().collect_generic_names()
    }
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
        varname: Sym,
        typename: Type,
        init: Box<Expr>,
        mutable: bool,
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
        generic_types: Vec<Type>,
    },
    Break,
    Return {
        retval: Box<Expr>,
    },
    /// Tuple constructor
    TupleCtor {
        body: Vec<Expr>,
    },
    StructCtor {
        types: BTreeMap<Sym, Type>,
        body: Vec<(Sym, Expr)>,
    },
    /// Tuple element reference
    TupleRef {
        expr: Box<Expr>,
        elt: usize,
    },
    /// Struct element reference
    StructRef {
        expr: Box<Expr>,
        elt: Sym,
    },
    /// Separate from a BinOp because its typechecking rules are different.
    Assign {
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    Ref {
        expr: Box<Expr>,
    },
    Deref {
        expr: Box<Expr>,
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
    pub const fn sized_int(i: i128, bytes: u8) -> Expr {
        Expr::Lit {
            val: Literal::SizedInteger { vl: i, bytes },
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

/// A top-level declaration in the source file.
#[derive(Debug, Clone, PartialEq)]
pub enum Decl {
    Function {
        name: Sym,
        signature: Signature,
        body: Vec<Expr>,
        doc_comment: Vec<String>,
    },
    Const {
        name: Sym,
        typename: Type,
        init: Expr,
        doc_comment: Vec<String>,
    },
    TypeDef {
        name: Sym,
        typedecl: Type,
        doc_comment: Vec<String>,
    },
}

/// A compilable chunk of AST.
///
/// Currently, basically a compilation unit.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Ast {
    pub decls: Vec<Decl>,
}
