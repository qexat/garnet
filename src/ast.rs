//! Abstract syntax tree.
//!
//! Should be a *pretty exact* representation of the source code,
//! including things like parentheses and comments.  That way we can
//! eventually use the same structure for a code formatter and not
//! have it nuke anything.
//!
//! Though code formatters have different constraints and priorities, if they have line wrapping
//! and stuff at least.  So, it might not be a particularly great code formatter.

use std::fmt;

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
    /// Type parameters
    pub typeparams: Vec<Sym>,
}

impl Signature {
    /// Returns a lambda typedef representing the signature
    pub fn to_type(&self) -> Type {
        let paramtypes = self.params.iter().map(|(_nm, ty)| ty.clone()).collect();
        let typeparams = self.typeparams.iter().map(|nm| Type::named0(*nm)).collect();
        Type::Func(paramtypes, Box::new(self.rettype.clone()), typeparams)
    }

    /// Get all the generic params out of this function sig
    pub fn type_params(&self) -> Vec<Sym> {
        self.to_type().get_toplevel_type_params()
    }

    /// Returns a string containing just the params and rettype bits of the sig
    pub fn to_name(&self) -> String {
        let names: Vec<_> = self
            .params
            .iter()
            .map(|(s, t)| format!("{} {}", &*s.val(), t.get_name()))
            .collect();
        let args = names.join(", ");

        let typenames: Vec<_> = self
            .typeparams
            .iter()
            .map(|t| (t.val().as_str()).to_string())
            .collect();
        let typeargs = typenames.join(", ");
        format!("(|{}| {}) {}", typeargs, args, self.rettype.get_name())
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
        typename: Option<Type>,
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
    While {
        cond: Box<Expr>,
        body: Vec<Expr>,
    },
    Lambda {
        signature: Signature,
        body: Vec<Expr>,
    },
    Funcall {
        func: Box<Expr>,
        params: Vec<Expr>,
        typeparams: Vec<Type>,
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
        body: BTreeMap<Sym, Expr>,
    },
    ArrayCtor {
        body: Vec<Expr>,
    },
    /// Array element reference
    ArrayRef {
        expr: Box<Expr>,
        idx: Box<Expr>,
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
    TypeUnwrap {
        expr: Box<Expr>,
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
