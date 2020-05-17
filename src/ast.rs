//! Abstract syntax tree.

pub enum Literal {
    Integer(i64),
    Bool(bool),
}

pub struct Symbol(String);

/// Binary operation
pub enum BOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
}

/// Unary operation
pub enum UOp {
    Neg,
    Deref,
}

pub struct IfCase {
    pub condition: Box<Expr>,
    pub body: Vec<Expr>,
}

pub enum Type {
    Name(Symbol),
}

/// A function type signature
pub struct Signature {
    pub params: Vec<(Symbol, Type)>,
    pub rettype: Type,
}

pub enum Expr {
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
}
