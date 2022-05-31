//! Typechecking and other semantic checking.
//! Operates on the HIR.
//!
//! This is mostly based on
//! https://ahnfelt.medium.com/type-inference-by-example-793d83f98382
//!
//! We go through and generate a type variable for each expression,
//! then unify to solve them, probably in a more-or-less-bidirectional
//! fashion.

use std::collections::HashMap;

use crate::hir::{self, Expr};
use crate::{TypeDef, TypeSym, VarSym, INT};

#[derive(Debug, Clone)]
pub enum TypeError {
    UnknownVar(VarSym),
    /*
    UnknownType(VarSym),
    InvalidReturn,
    Return {
        fname: VarSym,
        got: TypeSym,
        expected: TypeSym,
    },
    BopType {
        bop: hir::BOp,
        got1: TypeSym,
        got2: TypeSym,
        expected: TypeSym,
    },
    UopType {
        op: hir::UOp,
        got: TypeSym,
        expected: TypeSym,
    },
    LetType {
        name: VarSym,
        got: TypeSym,
        expected: TypeSym,
    },
    IfType {
        expected: TypeSym,
        got: TypeSym,
    },
    Cond {
        got: TypeSym,
    },
    Param {
        got: TypeSym,
        expected: TypeSym,
    },
    Call {
        got: TypeSym,
    },
    TupleRef {
        got: TypeSym,
    },
    StructRef {
        fieldname: VarSym,
        got: TypeSym,
    },
    StructField {
        expected: Vec<VarSym>,
        got: Vec<VarSym>,
    },
    EnumVariant {
        expected: Vec<VarSym>,
        got: VarSym,
    },
    TypeMismatch {
        expr_name: Cow<'static, str>,
        got: TypeSym,
        expected: TypeSym,
    },
    Mutability {
        expr_name: Cow<'static, str>,
    },
    */
}

impl std::error::Error for TypeError {}

impl std::fmt::Display for TypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.format())
    }
}

impl TypeError {
    pub fn format(&self) -> String {
        match self {
            TypeError::UnknownVar(sym) => format!("Unknown var: {}", INT.fetch(*sym)),
            /*
            TypeError::UnknownType(sym) => format!("Unknown type: {}", INT.fetch(*sym)),
            TypeError::InvalidReturn => {
                "return expression happened somewhere that isn't in a function!".to_string()
            }
            TypeError::Return {
                fname,
                got,
                expected,
            } => format!(
                "Function {} returns {} but should return {}",
                INT.fetch(*fname),
                INT.fetch_type(*got).get_name(),
                INT.fetch_type(*expected).get_name(),
            ),
            TypeError::BopType {
                bop,
                got1,
                got2,
                expected,
            } => format!(
                "Invalid types for BOp {:?}: expected {}, got {} + {}",
                bop,
                INT.fetch_type(*expected).get_name(),
                INT.fetch_type(*got1).get_name(),
                INT.fetch_type(*got2).get_name()
            ),
            TypeError::UopType { op, got, expected } => format!(
                "Invalid types for UOp {:?}: expected {}, got {}",
                op,
                INT.fetch_type(*expected).get_name(),
                INT.fetch_type(*got).get_name()
            ),
            TypeError::LetType {
                name,
                got,
                expected,
            } => format!(
                "initializer for variable {}: expected {} ({:?}), got {} ({:?})",
                INT.fetch(*name),
                INT.fetch_type(*expected).get_name(),
                *expected,
                INT.fetch_type(*got).get_name(),
                *got,
            ),
            TypeError::IfType { expected, got } => format!(
                "If block return type is {}, but we thought it should be something like {}",
                INT.fetch_type(*expected).get_name(),
                INT.fetch_type(*got).get_name(),
            ),
            TypeError::Cond { got } => format!(
                "If expr condition is {}, not bool",
                INT.fetch_type(*got).get_name(),
            ),
            TypeError::Param { got, expected } => format!(
                "Function wanted type {} in param but got type {}",
                INT.fetch_type(*expected).get_name(),
                INT.fetch_type(*got).get_name()
            ),
            TypeError::Call { got } => format!(
                "Tried to call function but it is not a function, it is a {}",
                INT.fetch_type(*got).get_name()
            ),
            TypeError::TupleRef { got } => format!(
                "Tried to reference tuple but didn't get a tuple, got {}",
                INT.fetch_type(*got).get_name()
            ),
            TypeError::StructRef { fieldname, got } => format!(
                "Tried to reference field {} of struct, but struct is {}",
                INT.fetch(*fieldname),
                INT.fetch_type(*got).get_name(),
            ),
            TypeError::StructField { expected, got } => format!(
                "Invalid field in struct constructor: expected {:?}, but got {:?}",
                expected, got
            ),
            TypeError::EnumVariant { expected, got } => {
                let expected_names: Vec<String> = expected
                    .into_iter()
                    .map(|nm| (&*INT.fetch(*nm)).clone())
                    .collect();
                format!(
                    "Unknown enum variant '{}', valid ones are {:?}",
                    INT.fetch(*got),
                    expected_names,
                )
            }
            TypeError::TypeMismatch {
                expr_name,
                expected,
                got,
            } => format!(
                "Type mismatch in '{}' expresssion, expected {} but got {}",
                expr_name,
                INT.fetch_type(*expected).get_name(),
                INT.fetch_type(*got).get_name()
            ),
            TypeError::Mutability { expr_name } => {
                format!("Mutability mismatch in '{}' expresssion", expr_name)
            }
            */
        }
    }
}

/// A variable binding
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarBinding {
    pub name: VarSym,
    pub typevar: TypeVar,
    pub mutable: bool,
}

/// A generated type variable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeVar(usize);

/// Top level type checking context struct.
#[derive(Clone, Debug)]
pub struct Tck {
    /// A mapping containing the return types of all expressions.
    /// The return type may not be known at any particular time,
    /// or may have partial type data known about it, which is fine.
    /// This table gets updated with real types as we figure them out.
    exprtypes: HashMap<hir::Eid, TypeSym>,
    /// Stores a map of the variables and types in scope at every
    /// expression in our program.  A liiiiiittle brute force, I
    /// admit, but pretty simple.  Maybe someday use the `im` crate?
    scopes: HashMap<hir::Eid, HashMap<VarSym, VarBinding>>,
}

impl Default for Tck {
    /// Create a new Tck with whatever default types etc we need.
    fn default() -> Self {
        Tck {
            exprtypes: Default::default(),
            scopes: Default::default(),
        }
    }
}

impl Tck {
    /// Panics if type does not exist.
    pub fn get_type(&self, eid: hir::Eid) -> TypeSym {
        *self
            .exprtypes
            .get(&eid)
            .expect("Tried to get type for Eid that doesn't exist!")
    }
}

pub fn typecheck(ir: hir::Ir) -> Result<Tck, TypeError> {
    let mut tck = Tck::default();
    /*
    ir.decls.iter().for_each(|d| predeclare_decl(&mut tck, d));
    let checked_decls = ir
        .decls
        .into_iter()
        .map(|decl| typecheck_decl(&mut tck, decl))
        .collect::<Result<Vec<hir::Decl>, TypeError>>()?;
    */
    Ok(tck)
}
