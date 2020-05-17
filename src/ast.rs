//! Abstract syntax tree.

#[derive(Debug, Clone)]
pub enum Literal {
    Integer(i64),
    Bool(bool),
}

#[derive(Debug, Clone)]
pub struct Symbol(pub String);

/// Binary operation
#[derive(Debug, Clone)]
pub enum BOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
}

/// Unary operation
#[derive(Debug, Clone)]
pub enum UOp {
    Neg,
    Deref,
}

#[derive(Debug, Clone)]
pub struct IfCase {
    pub condition: Box<Expr>,
    pub body: Vec<Expr>,
}

#[derive(Debug, Clone)]
pub enum Type {
    Name(Symbol),
}

/// A function type signature
#[derive(Debug, Clone)]
pub struct Signature {
    pub params: Vec<(Symbol, Type)>,
    pub rettype: Type,
}

/// Any expression.
#[derive(Debug, Clone)]
pub enum Expr {
    Lit {
        val: Literal,
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
    Funcall {
        func: Box<Expr>,
        params: Vec<Expr>,
    },
}

/// A top-level declaration in the source file.
#[derive(Debug, Clone)]
pub enum Decl {
    Function {
        name: Symbol,
        signature: Signature,
        body: Vec<Expr>,
    },
}
