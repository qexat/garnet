//! Low-level intermediate representation.
//!
//! Basically, I feel like my barely-lower-than-AST IR
//! does not have enough information about memory-type stuffs.
//! We could make a fairly pessimal and/or complicated stab
//! at compiling fancier things anyway, but it seems like
//! the hard way of doing it.
//!
//! So I want to be able to decide things like:
//!  * Whether a variable is in memory or is somewhere
//!    not addressable (register, WASM stack, wasm local, etc)
//!  * How many variables my function has, what its
//!    total stack size is, where it's returning
//!    its value
//!
//! So I'm going to try to make an SSA IR and see how it feels.

use std::collections::HashMap;

use crate::ir;
use crate::{Cx, TypeSym, VarSym};

/// The type of a var/virtual register, just a number
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Var(usize);

/// Basic block
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct BB(usize);

/// A function
pub struct Func {
    name: VarSym,
    params: Vec<(Var, TypeSym)>,
    returns: TypeSym,
    body: HashMap<BB, Block>,
}

/// The various ways we can end a basic block
/// Apparently a switch instruction also makes using
/// this far easier, and compiling things somewhat
/// harder.  Something to think about.
pub enum Branch {
    /// Unconditional jump
    Jump(Var, BB),
    /// Conditional branch
    Branch(Var, BB, BB),
    Return(Option<Var>),
    Unreachable,
}

/// The actual type of a basic block
pub struct Block {
    id: BB,
    body: Vec<Instr>,
    terminator: Branch,
}

/// an IR instruction
pub enum Instr {
    Assign(Var, Op),
}

/// Operations/rvalues
pub enum Op {
    // Values
    ValI32(i32),

    // Operators
    AddI32(Var, Var),
    SubI32(Var, Var),
    MulI32(Var, Var),
    DivI32(Var, Var),
    NegI32(Var),
    NotI32(Var),

    // Memory
    AddrOf(Var),
    LoadI32(Var),
    StoreI32(Var),

    // Control flow
    // TODO: phi can take any number of args
    Phi(BB, BB),
    Call(Var, VarSym, Vec<Var>),
}

pub struct FuncBuilder {
    func: Func,
    next_var: usize,
    next_bb: usize,
}

impl FuncBuilder {
    pub fn new(func_name: VarSym, params: &[TypeSym], rettype: TypeSym) -> Self {
        let mut res = Self {
            func: Func {
                name: func_name,
                params: vec![],
                returns: rettype,
                body: HashMap::new(),
            },
            next_var: 0,
            next_bb: 0,
        };
        let new_params = params.iter().map(|t| (res.next_var(), *t)).collect();
        res.func.params = new_params;
        res
    }

    fn next_var(&mut self) -> Var {
        let s = Var(self.next_var);
        self.next_var += 1;
        s
    }

    fn next_block(&mut self) -> Block {
        let id = BB(self.next_bb);
        self.next_bb += 1;
        let block = Block {
            id,
            body: vec![],
            terminator: Branch::Unreachable,
        };
        block
    }

    fn build(self) -> Func {
        self.func
    }

    /// Build a new instruction
    /// apparently currently the only option is Assign?
    ///
    fn assign(&mut self, bb: &mut Block, op: Op) -> Var {
        let v = self.next_var();
        let instr = Instr::Assign(v, op);
        bb.body.push(instr);
        v
    }
}

pub fn lower_decl(cx: &Cx, decl: &ir::Decl<TypeSym>) {
    match decl {
        ir::Decl::Function {
            name,
            signature,
            body,
        } => {
            // TODO: Map param names somehow...
            let params: Vec<_> = signature.params.iter().map(|(name, ty)| *ty).collect();
            let mut fb = FuncBuilder::new(*name, params.as_ref(), signature.rettype);
            todo!()
        }
        _ => todo!(),
    }
}

fn lower_expr(cx: &Cx, fb: &mut FuncBuilder, bb: &mut Block, expr: &ir::TypedExpr<TypeSym>) {
    use ir::Expr as E;
    use ir::*;
    match &expr.e {
        E::Lit { val } => match val {
            Literal::Integer(i) => {
                assert!(*i < (std::i32::MAX as i64));
                let op = Op::ValI32(*i as i32);
                fb.assign(bb, op);
            }
            // Turn bools into integers.
            Literal::Bool(b) => {
                let op = Op::ValI32(if *b { 1 } else { 0 });
                fb.assign(bb, op);
            }
        },

        _ => todo!(),
    }
}
