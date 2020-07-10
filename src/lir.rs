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
    /// What the ending instruction of the basic block is
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
    ValUnit,

    // Operators
    AddI32(Var, Var),
    SubI32(Var, Var),
    MulI32(Var, Var),
    DivI32(Var, Var),
    ModI32(Var, Var),
    NotI32(Var),

    // Memory
    AddrOf(Var),
    LoadI32(Var),
    StoreI32(Var),

    // Control flow
    Phi(Vec<BB>),
    Call(VarSym, Vec<Var>),
    CallIndirect(Var, Vec<Var>),
}

pub struct FuncBuilder {
    func: Func,
    next_var: usize,
    next_bb: usize,
}

impl FuncBuilder {
    pub fn new(func_name: VarSym, params: &[(VarSym, TypeSym)], rettype: TypeSym) -> Self {
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
        let new_params = params.iter().map(|(_v, t)| (res.next_var(), *t)).collect();
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
            let mut fb = FuncBuilder::new(*name, signature.params.as_ref(), signature.rettype);
            todo!()
        }
        _ => todo!(),
    }
}
fn lower_exprs(
    cx: &Cx,
    fb: &mut FuncBuilder,
    bb: &mut Block,
    exprs: &[ir::TypedExpr<TypeSym>],
) -> Option<Var> {
    let mut last = None;
    for e in exprs {
        last = Some(lower_expr(cx, fb, bb, e));
    }
    last
}

fn lower_expr(cx: &Cx, fb: &mut FuncBuilder, bb: &mut Block, expr: &ir::TypedExpr<TypeSym>) -> Var {
    use ir::Expr as E;
    use ir::*;
    match &expr.e {
        E::Lit { val } => {
            let v = match val {
                Literal::Integer(i) => {
                    assert!(*i < (std::i32::MAX as i64));
                    *i
                }
                // Turn bools into integers.
                Literal::Bool(b) => {
                    if *b {
                        1
                    } else {
                        0
                    }
                }
            };
            let op = Op::ValI32(v as i32);
            fb.assign(bb, op)
        }

        E::Var { .. } => todo!(),
        E::BinOp { op, lhs, rhs } => {
            let v1 = lower_expr(cx, fb, bb, &*lhs);
            let v2 = lower_expr(cx, fb, bb, &*rhs);
            let o = match op {
                ir::BOp::Add => Op::AddI32,
                ir::BOp::Sub => Op::SubI32,
                ir::BOp::Mul => Op::MulI32,
                // TODO: Check for div0?
                ir::BOp::Div => Op::DivI32,
                // TODO: Check for div0?
                ir::BOp::Mod => Op::ModI32,
                _ => todo!(),
            }(v1, v2);
            fb.assign(bb, o)
        }
        E::UniOp { op, rhs } => match op {
            // Turn -x into 0 - x
            ir::UOp::Neg => {
                let v1 = fb.assign(bb, Op::ValI32(0));
                let v2 = lower_expr(cx, fb, bb, &*rhs);
                let op = Op::SubI32(v1, v2);
                fb.assign(bb, op)
            }
            ir::UOp::Not => {
                let v = lower_expr(cx, fb, bb, &*rhs);
                let op = Op::NotI32(v);
                fb.assign(bb, op)
            }
        },
        E::Block { body } => {
            // If the body is empty, just make an unreachable value
            // that is used nowhere
            // If it is used as something other than unit then
            // this (hopefully) doesn't typecheck.
            lower_exprs(cx, fb, bb, &*body).unwrap_or_else(|| fb.assign(bb, Op::ValUnit))
        }
        E::Let {
            varname,
            typename,
            init,
            mutable,
        } => todo!(),
        E::If { cases, falseblock } => {
            /* NOT AWAKE ENOUGH FOR THIS
                let mut bbs = vec![];
                let mut next_bb = fb.next_block();
                for (case, body) in cases {
                    let case_res = lower_expr(cx, fb, bb, case);
                    let mut new_bb = fb.next_block();
                    bbs.push(new_bb.id);

                    let res = lower_exprs(cx, fb, bb, &*body).unwrap_or_else(|| fb.assign(bb, Op::ValUnit))
                    new_bb.terminator = Branch::Jump(res, next_bb);
                }
                ....
                fb.assign(bb, Op::Phi(bbs));
            */
            todo!()
        }
        E::Loop { body } => todo!(),
        E::Lambda { .. } => {
            panic!("Unlifted lambda, should never happen?");
        }
        E::Funcall { func, params } => {
            // First, evaluate the args.

            let param_vars = params.iter().map(|e| lower_expr(cx, fb, bb, e)).collect();
            match &func.e {
                E::Var { name } => {
                    let var = todo!();
                    fb.assign(bb, Op::Call(var, param_vars))
                }
                _expr => {
                    // We got some other expr, compile it and do an indirect call 'cause it heckin'
                    // better be a function
                    let v = lower_expr(cx, fb, bb, &func);
                    fb.assign(bb, Op::CallIndirect(v, param_vars))
                }
            }
        }
        E::Break => todo!(),
        E::Return { retval } => todo!(),
        E::TupleCtor { body } => todo!(),
        E::TupleRef { expr, elt } => todo!(),
        E::Assign { lhs, rhs } => todo!(),
    }
}
