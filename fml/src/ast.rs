//! Abstract syntax tree.
//!
//! Should be a *pretty exact* representation of the source code,
//! including things like parentheses and comments.  That way we can
//! eventually use the same structure for a code formatter and not
//! have it nuke anything.
//!
//! Though code formatters have different constraints and priorities, if they have line wrapping
//! and stuff at least.  So, it might not be a particularly great code formatter.

use std::collections::HashMap;
use std::sync::Mutex;

use crate::*;

/// Literal value
#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    /// An integer of some kind
    Integer(i32),
    /// A boolean
    Bool(bool),
    /// This is kinda weird 'cause we can't parse it,
    /// but we can lower our enums to it.
    /// First string is the enum name, second is the value.
    EnumLit(String, String),
}

/// A function type signature
#[derive(Debug, Clone, PartialEq)]
pub struct Signature<T = Type> {
    /// Parameters
    pub params: Vec<(String, T)>,
    /// Return type
    pub rettype: T,
}

impl Signature {
    /// Turn the function signature into a Lambda
    pub fn as_type(&self) -> Type {
        let paramtypes = self.params.iter().map(|(_nm, ty)| ty.clone()).collect();
        Type::Func(paramtypes, Box::new(self.rettype.clone()))
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct AstId(usize);

impl AstId {
    /// Creates a new globally unique AstId
    fn new() -> AstId {
        let mut val = AST_ID.lock().unwrap();
        let new_val = *val + 1;
        let new_id = AstId(*val);
        *val = new_val;
        new_id
    }
}

/// An AST node wrapper that contains information
/// common to all AST nodes.
#[derive(Debug, Clone, PartialEq)]
pub struct ExprNode {
    pub node: Box<Expr>,
    pub id: AstId,
}

use once_cell::sync::Lazy;
static AST_ID: Lazy<Mutex<usize>> = once_cell::sync::Lazy::new(|| Mutex::new(0));

impl ExprNode {
    pub fn new(expr: Expr) -> Self {
        let new_id = AstId::new();
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
        typename: Option<Type>,
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
    TupleCtor {
        body: Vec<ExprNode>,
    },
    StructCtor {
        body: HashMap<String, ExprNode>,
    },
    TypeCtor {
        name: String,
        type_params: Vec<Type>,
        body: ExprNode,
    },
    // Opposite of TypeCtor
    TypeUnwrap {
        e: ExprNode,
    },
    StructRef {
        e: ExprNode,
        name: String,
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
    TypeDef {
        name: String,
        params: Vec<String>,
        ty: Type,
    },
    ConstDef {
        name: String,
        // ty: Type,
        init: ExprNode,
    },
}

/// A compilable chunk of AST.
///
/// Currently, basically a compilation unit.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Ast {
    pub decls: Vec<Decl>,
}

/// To implement enum values we just juggle enum decl's
/// to create a new const def too.
impl Ast {
    pub fn lower(self) -> Self {
        let mut out = vec![];
        for decl in self.decls.into_iter() {
            use Decl::*;
            match &decl {
                Function { .. } => out.push(decl),
                ConstDef { .. } => out.push(decl),
                TypeDef {
                    name,
                    ty,
                    params: _params, // Can never be anything for an enum I think
                } => {
                    out.push(decl.clone());
                    // For `type Foo = enum Foo, Bar, Baz end`
                    // synthesize
                    // const Foo = $Foo {
                    //     .Foo =
                    // }
                    match ty {
                        Type::Enum(ts) => {
                            let struct_body: HashMap<_, _> = ts
                                .iter()
                                .map(|s| {
                                    (
                                        s.clone(),
                                        ExprNode::new(Expr::Lit {
                                            val: Literal::EnumLit(name.clone(), s.clone()),
                                        }),
                                    )
                                })
                                .collect();
                            let init_val = ExprNode::new(Expr::StructCtor { body: struct_body });
                            let new_constdef = ConstDef {
                                name: name.clone(),
                                init: init_val,
                            };
                            out.push(new_constdef);
                        }
                        _other => (),
                    }
                }
            }
        }
        Ast { decls: out }
    }
}
