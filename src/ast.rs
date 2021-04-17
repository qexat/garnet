//! Abstract syntax tree.
//!
//! Should be a *pretty exact* representation of the source code,
//! including things like parentheses and comments.  That way we can
//! eventually use the same structure for a code formatter and not
//! have it nuke anything.
//!
//! Though code formatters have different constraints, if they have line
//! wrapping and stuff at least.  So, it might not be a particularly
//! great code formatter.

use crate::*;

/// Literal value
#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    /// An integer of some kind
    Integer(i128),
    /// An integer with a known size
    SizedInteger {
        vl: i128,
        bytes: u8,
    },
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
    pub fn input_type(&self) -> TypeSym {
        use BOp::*;
        match self {
            And | Or | Xor => INT.bool(),
            _ => INT.iunknown(),
        }
    }

    /// What the resultant type of the binop is.
    ///
    /// Needs to know what the type of the expression given to it is,
    /// but also assumes that the LHS and RHS have the same input type.
    /// Ensuring that is left as an exercise to the user.
    pub fn output_type(&self, input_type: TypeSym) -> TypeSym {
        use BOp::*;
        match self {
            Add | Sub | Mul | Div | Mod => {
                let def = INT.fetch_type(input_type);
                if def.is_integer() {
                    return input_type;
                } else {
                    unimplemented!("hmmmm, typechecking should probably never allow this");
                }
            }
            Eq | Neq | Gt | Lt | Gte | Lte => INT.bool(),
            And | Or | Xor => INT.bool(),
        }
    }
}

/// Unary operation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UOp {
    Neg,
    Not,
    Ref,
    Deref,
}

impl UOp {
    /// Returns the type that the unary op operates on.
    /// Currently, only numbers.
    pub fn input_type(&self) -> TypeSym {
        use UOp::*;
        match self {
            Neg => INT.iunknown(),
            Not => INT.bool(),
            Ref => todo!(),
            Deref => todo!(),
        }
    }

    /// What the resultant type of the uop is
    pub fn output_type(&self, input_type: TypeSym) -> TypeSym {
        use UOp::*;
        match self {
            Neg => {
                let def = INT.fetch_type(input_type);
                if def.is_integer() {
                    return input_type;
                } else {
                    unimplemented!("hmmmm, typechecking should probably never allow this");
                }
            }
            Not => INT.bool(),
            Ref => todo!(),
            Deref => todo!(),
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
    pub(crate) fn to_type(&self) -> TypeSym {
        let args = self.params.iter().map(|(_v, t)| *t).collect();
        let t = TypeDef::Lambda(args, self.rettype);
        INT.intern_type(&t)
    }
}

/// Any expression.
/// So, basically anything not a top-level decl.
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
        typename: Option<TypeSym>,
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
    },
    Break,
    Return {
        retval: Box<Expr>,
    },
    /// Tuple constructor
    TupleCtor {
        body: Vec<Expr>,
    },
    /// Tuple element reference
    TupleRef {
        expr: Box<Expr>,
        elt: usize,
    },
    Ref {
        expr: Box<Expr>,
    },
    Deref {
        expr: Box<Expr>,
    },
    /// Separate from a BinOp because its typechecking rules are different.
    Assign {
        lhs: Box<Expr>,
        rhs: Box<Expr>,
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
            name: INT.intern(name),
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
        doc_comment: Vec<String>,
    },
    Const {
        name: VarSym,
        typename: TypeSym,
        init: Expr,
        doc_comment: Vec<String>,
    },
    TypeDef {
        name: VarSym,
        typedecl: TypeSym,
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
