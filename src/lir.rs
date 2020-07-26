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
//!
//! Okay, so apparently the easy way is to start with everything
//! compound in memory: structs, tuples, etc and everything
//! manipulating them through pointers.  Then you have a pass
//! to go through and scoot them into registers as the opportunity
//! provides.

use std::collections::HashMap;

use crate::ir;
use crate::{Cx, TypeSym, VarSym};

/// Shortcut
type TExpr = ir::TypedExpr<TypeSym>;

/// A var/virtual register, just a number
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Var(usize);

/// Basic block ID.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct BB(usize);

/// A single struct containing all the bits necessary for a LIR module.
#[derive(Debug, Default)]
pub struct Lir {
    pub funcs: Vec<Func>,
}

#[derive(Debug, Clone)]
pub struct VarBinding {
    pub name: VarSym,
    pub typename: TypeSym,
    pub mutable: bool,
    pub var: Var,
}

/// A function
#[derive(Debug)]
pub struct Func {
    pub name: VarSym,
    pub signature: ir::Signature,
    pub params: Vec<VarBinding>,
    pub returns: TypeSym,
    pub body: HashMap<BB, Block>,
    /// The first basic block to execute.
    /// Could implicitly be 0, but making it
    /// explicit here might make things easier
    /// to rearrange down the line.
    pub entry: BB,

    /// Total list of local variable declarations
    pub locals: Vec<VarBinding>,

    /// Layout of stack frame, sorted by address
    /// Each item is symbol, size-in-bytes
    /// TODO: ...goddammit pointer size changes with platform.
    pub frame_layout: Vec<(VarSym, usize)>,
}

/// The various ways we can end a basic block
/// Apparently a switch instruction also makes using
/// this far easier, and compiling things somewhat
/// harder.  Something to think about.
#[derive(Debug)]
pub enum Branch {
    /// Unconditional jump
    Jump(Var, BB),
    /// Conditional branch
    Branch(Var, BB, BB),
    /// Return, optionally with some value
    Return(Option<Var>),
    /// Never returns -- dead code, infinite loop, etc.
    Unreachable,
}

/// The actual type of a basic block
#[derive(Debug)]
pub struct Block {
    pub id: BB,
    pub body: Vec<Instr>,
    /// What the ending instruction of the basic block is
    pub terminator: Branch,
}

impl Block {
    /// Shortcut for retrieving the last value stored in the block.
    /// Returns None if the block is empty.
    fn last_value(&self) -> Option<Var> {
        match self.body.last()? {
            Instr::Assign(var, _, _) => Some(*var),
        }
    }
}

/// an IR instruction
#[derive(Debug)]
pub enum Instr {
    Assign(Var, TypeSym, Op),
}

/// Operations/rvalues
#[derive(Debug)]
pub enum Op {
    // Values
    ValI32(i32),
    ValUnit,

    // Operators
    BinOpI32(ir::BOp, Var, Var),
    UniOpI32(ir::UOp, Var),

    // Memory
    GetLocal(VarSym),
    SetLocal(VarSym, Var),
    AddrOf(Var),
    LoadI32(Var),
    /// Loads address + offset
    LoadOffsetI32(Var, Var),
    /// Address, value
    StoreI32(Var, Var),
    /// Store address + offset
    StoreOffsetI32(Var, Var, Var),

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
    /// Creates a new func.  The func already has a BB in it that is labelled as
    /// the entry point
    pub fn new(func_name: VarSym, params: &[(VarSym, TypeSym)], rettype: TypeSym) -> Self {
        let mut res = Self {
            func: Func {
                name: func_name,
                signature: ir::Signature {
                    params: Vec::from(params),
                    rettype: rettype,
                },
                params: vec![],
                returns: rettype,
                body: HashMap::new(),
                entry: BB(0),
                locals: vec![],

                frame_layout: vec![],
            },
            next_var: 0,
            next_bb: 0,
        };
        let new_params = params
            .iter()
            .map(|(v, t)| VarBinding {
                name: *v,
                typename: *t,
                mutable: false,
                var: res.next_var(),
            })
            .collect();
        let first_block = res.next_block();
        res.func.params = new_params;
        res.func.entry = first_block.id;

        res
    }

    fn next_var(&mut self) -> Var {
        let s = Var(self.next_var);
        self.next_var += 1;
        s
    }

    /// Create a new empty Block with a valid ID.
    /// Must be added to the function builder after it's full.
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

    /// Set the entry point of the function being built to the given block.
    fn set_entry(&mut self, block: &Block) {
        self.func.entry = block.id;
    }

    /// Add a block
    fn add_block(&mut self, block: Block) -> BB {
        let id = block.id;
        self.func.body.insert(id, block);
        id
    }

    /*
    /// Gets a mutable reference to a given block.
    fn get_block(&mut self, bb: BB) -> &mut Block {
        self.func.body.get_mut(&bb).expect("Should never fail")
    }
    */

    fn build(self) -> Func {
        self.func
    }

    fn add_var(&mut self, name: VarSym, ty: TypeSym, v: Var, mutable: bool) {
        self.func.locals.push(VarBinding {
            name,
            typename: ty,
            mutable,
            var: v,
        });
    }

    fn get_var(&self, name: VarSym) -> &VarBinding {
        self.func
            .locals
            .iter()
            .rev()
            .find(|v| v.name == name)
            .unwrap_or_else(|| {
                self.func
                    .params
                    .iter()
                    .find(|v| v.name == name)
                    .expect("Unknown var in LIR, should never happen")
            })
    }

    /// Build a new instruction
    /// apparently currently the only option is Assign?
    ///
    fn assign(&mut self, bb: &mut Block, typ: TypeSym, op: Op) -> Var {
        let v = self.next_var();
        let instr = Instr::Assign(v, typ, op);
        bb.body.push(instr);
        v
    }
}

pub fn lower_ir(cx: &Cx, ir: &ir::Ir<TypeSym>) -> Lir {
    let mut lir = Lir::default();
    for decl in ir.decls.iter() {
        lower_decl(cx, &mut lir, &decl);
    }
    lir
}

fn lower_decl(cx: &Cx, lir: &mut Lir, decl: &ir::Decl<TypeSym>) {
    match decl {
        ir::Decl::Function {
            name,
            signature,
            body,
        } => {
            // TODO HERE: Add arguments to symbol table!
            let mut fb = FuncBuilder::new(*name, signature.params.as_ref(), signature.rettype);
            let mut first_block = fb.next_block();
            let last_var = lower_exprs(cx, &mut fb, &mut first_block, body);
            first_block.terminator = Branch::Return(last_var);
            fb.set_entry(&first_block);
            fb.add_block(first_block);
            lir.funcs.push(fb.build());
        }
        _ => todo!(),
    }
}
fn lower_exprs(cx: &Cx, fb: &mut FuncBuilder, bb: &mut Block, exprs: &[TExpr]) -> Option<Var> {
    let mut last = None;
    for e in exprs {
        last = Some(lower_expr(cx, fb, bb, e));
    }
    last
}

fn lower_expr(cx: &Cx, fb: &mut FuncBuilder, bb: &mut Block, expr: &TExpr) -> Var {
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
            fb.assign(bb, expr.t, op)
        }

        E::Var { name } => {
            let op = Op::GetLocal(*name);
            fb.assign(bb, expr.t, op)
        }
        E::BinOp { op, lhs, rhs } => {
            let v1 = lower_expr(cx, fb, bb, &*lhs);
            let v2 = lower_expr(cx, fb, bb, &*rhs);
            let o = Op::BinOpI32(*op, v1, v2);
            fb.assign(bb, expr.t, o)
        }
        E::UniOp { op, rhs } => match op {
            // Turn -x into 0 - x
            ir::UOp::Neg => {
                let v1 = fb.assign(bb, cx.i32(), Op::ValI32(0));
                let v2 = lower_expr(cx, fb, bb, &*rhs);
                let op = Op::BinOpI32(ir::BOp::Sub, v1, v2);
                fb.assign(bb, expr.t, op)
            }
            ir::UOp::Not => {
                let v = lower_expr(cx, fb, bb, &*rhs);
                let op = Op::UniOpI32(ir::UOp::Not, v);
                fb.assign(bb, expr.t, op)
            }
            _ => todo!(),
        },
        E::Block { body } => {
            // If the body is empty, just make an unreachable value
            // that is used nowhere
            // If it is used as something other than unit then
            // this (hopefully) doesn't typecheck.
            lower_exprs(cx, fb, bb, &*body).unwrap_or_else(|| fb.assign(bb, cx.unit(), Op::ValUnit))
        }
        E::Let {
            varname,
            typename,
            init,
            mutable,
        } => {
            let v = lower_expr(cx, fb, bb, &*init);
            fb.add_var(*varname, *typename, v, *mutable);
            fb.assign(bb, cx.unit(), Op::ValUnit)
        }
        E::If { cases, falseblock } => {
            // For a single if expr, we make this structure:
            // start:
            //   %1 = cond
            //   branch %1 ifpart elsepart
            // ifpart:
            //   %2 = if_body
            //   jump end
            // elsepart:
            //   %3 = else_body
            //   jump end
            // end:
            //   %4 = phi ifpart elsepart
            //
            // But our if expr's have multiple cases, so say we have:
            // if cond1 then if_body1
            // elseif cond2 then if_body2
            // elseif cond3 then if_body3
            // else else_body
            // end
            //
            // That then turns in to:
            //
            // check_cond1:
            //   %1 = cond1
            //   branch %1 ifpart1 check_cond2
            // ifpart1:
            //   %2 = if_body1
            //   jump end
            // check_cond2:
            //   %3 = cond2
            //   branch %3 ifpart2 check_cond3
            // ifpart2:
            //   %4 = if_body2
            //   jump end
            // check_cond3:
            //   %5 = cond3
            //   branch %5 ifpart3 elsepart
            // ifpart3:
            //   %6 = if_body3
            //   jump end
            // elsepart:
            //   %6 = else_body
            //   jump end
            // end:
            //   %7 = phi ifpart1 ifpart2 ifpart3 elsepart
            //
            // That might get simpler if we write it as:
            //
            // check_cond1:
            //   %1 = cond1
            //   branch %1 ifpart1 check_cond2
            // check_cond2:
            //   %3 = cond2
            //   branch %3 ifpart2 check_cond3
            // check_cond3:
            //   %5 = cond3
            //   branch %5 ifpart3 elsepart
            //
            // ifpart1:
            //   %2 = if_body1
            //   jump end
            // ifpart2:
            //   %4 = if_body2
            //   jump end
            // ifpart3:
            //   %6 = if_body3
            //   jump end
            //
            // elsepart:
            //   %6 = else_body
            //   jump end
            // end:
            //   %7 = phi ifpart1 ifpart2 ifpart3 elsepart

            // I feel like this actually gets lots simpler using a recursive solution.
            // Returns the ID of the first check_cond basic block.
            //
            // TODO: Refactor, there's some redundant code here.
            fn recursive_build_bbs(
                cx: &Cx,
                fb: &mut FuncBuilder,
                end_bb: BB,
                cases: &[(TExpr, Vec<TExpr>)],
                elsebody: &[TExpr],
                accm: &mut Vec<BB>,
            ) -> BB {
                assert!(cases.len() != 0);
                let (cond, ifbody) = &cases[0];
                if cases.len() == 1 {
                    // We are on the last if statement, wire it up to the else statement.
                    let cond_bb = &mut fb.next_block();
                    let body_bb = &mut fb.next_block();
                    let else_body_bb = &mut fb.next_block();

                    let cond_result = lower_expr(cx, fb, cond_bb, &cond);
                    let body_result = lower_exprs(cx, fb, body_bb, &ifbody)
                        .unwrap_or_else(|| fb.assign(body_bb, cx.unit(), Op::ValUnit));
                    let else_result = lower_exprs(cx, fb, else_body_bb, elsebody)
                        .unwrap_or_else(|| fb.assign(else_body_bb, cx.unit(), Op::ValUnit));

                    // Wire all the BB's together
                    cond_bb.terminator = Branch::Branch(cond_result, body_bb.id, else_body_bb.id);
                    body_bb.terminator = Branch::Jump(body_result, end_bb);
                    else_body_bb.terminator = Branch::Jump(else_result, end_bb);
                    // Accumulate the BB's that jump to "end" because we're going to have to put
                    // them in the phi
                    accm.push(body_bb.id);
                    accm.push(else_body_bb.id);
                    return cond_bb.id;
                } else {
                    // Build the next case...
                    let next_bb = recursive_build_bbs(cx, fb, end_bb, &cases[1..], elsebody, accm);
                    // Build our cond and body
                    let cond_bb = &mut fb.next_block();
                    let body_bb = &mut fb.next_block();

                    let cond_result = lower_expr(cx, fb, cond_bb, &cond);
                    let body_result = lower_exprs(cx, fb, body_bb, &ifbody)
                        .unwrap_or_else(|| fb.assign(body_bb, cx.unit(), Op::ValUnit));

                    // Wire together BB's
                    cond_bb.terminator = Branch::Branch(cond_result, body_bb.id, next_bb);
                    body_bb.terminator = Branch::Jump(body_result, end_bb);
                    return cond_bb.id;
                }
            }
            // This is the block that all the if branches feed into.
            let end_bb = &mut fb.next_block();
            // Make blocks for all the if branches
            let mut accm = vec![];
            let first_cond_bb =
                recursive_build_bbs(cx, fb, end_bb.id, cases, falseblock, &mut accm);

            // Add a phi instruction combining all the branch results
            let last_result = fb.assign(end_bb, expr.t, Op::Phi(accm));
            // Connect the first BB to the start of the if block
            let last_value = bb
                .last_value()
                .unwrap_or_else(|| fb.assign(bb, cx.unit(), Op::ValUnit));
            bb.terminator = Branch::Jump(last_value, first_cond_bb);

            // TODO: How do we make this also say "here is the next BB to work with"?
            last_result
        }
        E::Loop { .. } => todo!(),
        E::Lambda { .. } => {
            panic!("Unlifted lambda, should never happen?");
        }
        E::Funcall { func, params } => {
            // First, evaluate the args.
            let param_vars = params.iter().map(|e| lower_expr(cx, fb, bb, e)).collect();
            // Then, issue the instruction depending on the type of call.
            match &func.e {
                E::Var { name } => fb.assign(bb, expr.t, Op::Call(*name, param_vars)),
                _expr => {
                    // We got some other expr, compile it and do an indirect call 'cause it heckin'
                    // better be a function
                    let v = lower_expr(cx, fb, bb, &func);
                    fb.assign(bb, expr.t, Op::CallIndirect(v, param_vars))
                }
            }
        }
        E::Break => todo!(),
        E::Return { retval } => todo!(),
        E::TupleCtor { body } => todo!(),
        E::TupleRef { expr, elt } => todo!(),
        E::Assign { lhs, rhs } => todo!(),
        E::Ref { .. } => todo!(),
        E::Deref { .. } => todo!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;

    fn compile_lir(src: &str) -> Lir {
        let cx = &mut Cx::new();
        let ast = {
            let mut parser = parser::Parser::new(cx, src);
            parser.parse()
        };
        let ir = ir::lower(&ast);
        let ir = passes::run_passes(cx, ir);
        let checked = typeck::typecheck(cx, ir)
            .unwrap_or_else(|e| panic!("Type check error: {}", e.format(cx)));
        lower_ir(cx, &checked)
    }

    #[test]
    fn lir_works_at_all() {
        let src = "fn foo(): I32 = 3 end";
        let lir = compile_lir(src);
        assert!(!lir.funcs.is_empty());
    }
}
