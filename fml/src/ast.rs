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
    Integer(i32),
}

/// A function type signature
#[derive(Debug, Clone, PartialEq)]
pub struct Signature {
    /// Parameters
    pub params: Vec<(String, TypeInfo)>,
    /// Return type
    pub rettype: TypeInfo,
}

#[derive(Debug, Copy, Clone, PartialEq, Hash)]
pub struct AstId(usize);

/// An AST node wrapper that contains information
/// common to all AST nodes.
#[derive(Debug, Clone, PartialEq)]
pub struct ExprNode {
    pub node: Box<Expr>,
    pub id: AstId,
}

use once_cell;
static AST_ID: once_cell::sync::Lazy<usize> = once_cell::sync::Lazy::new(|| 0);

impl ExprNode {
    pub fn new(expr: Expr) -> Self {
        let new_id = AstId(0);
        Self {
            node: Box::new(expr),
            id: new_id,
        }
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
        name: String,
    },
    Let {
        varname: String,
        typename: TypeInfo,
        init: ExprNode,
    },
    Lambda {
        signature: Signature,
        body: Vec<ExprNode>,
    },
    Funcall {
        func: ExprNode,
        params: Vec<ExprNode>,
    },
}

impl Expr {
    /// Shortcut function for making literal bools
    /// Shortcut function for making literal integers
    pub const fn int(i: i32) -> Expr {
        Expr::Lit {
            val: Literal::Integer(i),
        }
    }
}

/// A top-level declaration in the source file.
#[derive(Debug, Clone, PartialEq)]
pub enum Decl {
    Function {
        name: String,
        signature: Signature,
        body: Vec<ExprNode>,
    },
}

/// A compilable chunk of AST.
///
/// Currently, basically a compilation unit.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Ast {
    pub decls: Vec<Decl>,
}