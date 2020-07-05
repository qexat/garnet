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

use crate::{TypeSym, VarSym};

/// The type of a var/virtual register, just a number
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Var(usize);

/// Basic block
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct BB(usize);

/// A function
pub struct Func {
    name: VarSym,
    params: Vec<TypeSym>,
    returns: TypeSym,
    body: HashMap<BB, Vec<Instr>>,
}

/// an IR instruction
pub enum Instr {
    Assign(Var, Op),
    /// Conditional branch
    Branch(Var, BB, BB),
    /// Unconditional jump
    Jump(Var, BB),
    Return(Var),
}

/// Operations/rvalues
pub enum Op {
    // Values
    ValI32(i32),
    ValBool(bool),

    // Operators
    AddI32(Var, Var),
    SubI32(Var, Var),
    MulI32(Var, Var),
    DivI32(Var, Var),
    NegI32(Var),
    NotBool(Var),

    // Memory
    AddrOf(Var),
    LoadI32(Var),
    StoreI32(Var),

    // Control flow
    Phi(BB, BB),
    Call(Var, VarSym, Vec<Var>),
}
