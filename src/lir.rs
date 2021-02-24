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

use std::collections::BTreeMap;

use crate::hir;
use crate::{Cx, TypeSym, VarSym};

/// Shortcut
type TExpr = hir::TypedExpr<TypeSym>;

/// A var/virtual register, just a number
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Var(usize);

/// Basic block ID.  Just a number.
/// The block structure itself is the `Block`.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct BB(usize);

/// A single struct containing all the bits necessary for a LIR module.
#[derive(Debug, Default)]
pub struct Lir {
    pub funcs: Vec<Func>,
}

/// A declaration for a bound variable
#[derive(Debug, Copy, Clone)]
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
    pub signature: hir::Signature,
    pub params: Vec<VarBinding>,
    /// Use a BTreeMap here so it stays ordered, which makes
    /// it easier to heckin' read.
    pub body: BTreeMap<BB, Block>,
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
    /// Unconditional jump.
    /// The var that goes into it is the "result of the block" that gets fed
    /// into any following Phi instructions.
    Jump(Var, BB),
    /// Conditional branch
    /// Tests the given var and jumps to the first BB if true, second if false.
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
    BinOpI32(hir::BOp, Var, Var),
    UniOpI32(hir::UOp, Var),

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
    current_bb: BB,

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
                signature: hir::Signature {
                    params: Vec::from(params),
                    rettype,
                },
                params: vec![],
                body: BTreeMap::new(),
                entry: BB(0),
                locals: vec![],

                frame_layout: vec![],
            },
            current_bb: BB(0),
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
        let first_block = res.next_block().id;
        res.func.params = new_params;
        res.func.entry = first_block;

        res
    }

    /// Create a new Var slot with an unused ID.
    fn next_var(&mut self) -> Var {
        let s = Var(self.next_var);
        self.next_var += 1;
        s
    }

    /// Create a new empty Block with a valid ID.
    /// Must be added to the function builder after it's full.
    ///
    /// ...this is kinda awful, actually, but I don't see a better
    /// way to be able to construct multiple blocks at once.
    fn next_block(&mut self) -> &mut Block {
        let id = BB(self.next_bb);
        let block = Block {
            id,
            body: vec![],
            terminator: Branch::Unreachable,
        };
        self.next_bb += 1;
        self.current_bb = id;
        let id = block.id;
        self.func.body.insert(id, block);
        self.get_current_block()
    }

    /// Set the entry point of the function being built to the given block.
    fn _set_entry(&mut self, block: BB) {
        self.func.entry = block;
    }

    /// Gets a mutable reference to a given block.
    fn get_block(&mut self, bb: BB) -> &mut Block {
        self.func.body.get_mut(&bb).expect("Should never fail")
    }

    fn get_current_block(&mut self) -> &mut Block {
        self.get_block(self.current_bb)
    }

    fn _set_current_block(&mut self, bb: BB) {
        self.current_bb = bb;
    }

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

    fn _get_var(&self, name: VarSym) -> &VarBinding {
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

    /// Build a new instruction in the current basic block.
    /// Apparently currently the only option is Assign.
    /// Later we might have things like alloca and whatever though.
    fn assign(&mut self, typ: TypeSym, op: Op) -> Var {
        let v = self.next_var();
        let instr = Instr::Assign(v, typ, op);
        self.get_current_block().body.push(instr);
        v
    }

    /// Build a new instruction in the given basic block.
    fn assign_block(&mut self, bb: BB, typ: TypeSym, op: Op) -> Var {
        let v = self.next_var();
        let instr = Instr::Assign(v, typ, op);
        self.get_block(bb).body.push(instr);
        v
    }
}

pub fn lower_hir(cx: &Cx, hir: &hir::Ir<TypeSym>) -> Lir {
    let mut lir = Lir::default();
    for decl in hir.decls.iter() {
        lower_decl(cx, &mut lir, &decl);
    }
    lir
}

fn lower_decl(cx: &Cx, lir: &mut Lir, decl: &hir::Decl<TypeSym>) {
    match decl {
        hir::Decl::Function {
            name,
            signature,
            body,
        } => {
            let mut fb = FuncBuilder::new(*name, signature.params.as_ref(), signature.rettype);
            // Add arguments to symbol bindings.
            for (pname, ptype) in signature.params.iter() {
                let next_var = fb.next_var();
                fb.add_var(*pname, *ptype, next_var, false);
            }
            // Construct the body
            let last_var = lower_exprs(cx, &mut fb, body);
            let last_block = fb.get_current_block();
            last_block.terminator = Branch::Return(last_var);
            lir.funcs.push(fb.build());
        }
        _ => todo!(),
    }
}
fn lower_exprs(cx: &Cx, fb: &mut FuncBuilder, exprs: &[TExpr]) -> Option<Var> {
    let mut last = None;
    for e in exprs {
        last = Some(lower_expr(cx, fb, e));
    }
    last
}

fn lower_expr(cx: &Cx, fb: &mut FuncBuilder, expr: &TExpr) -> Var {
    use hir::Expr as E;
    use hir::*;
    match &expr.e {
        E::Lit { val } => {
            let v = match val {
                Literal::Integer(i) => {
                    assert!(*i < (std::i32::MAX as i128));
                    *i
                }
                Literal::SizedInteger { vl, .. } => {
                    assert!(*vl < (std::i32::MAX as i128));
                    *vl
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
            fb.assign(expr.t, op)
        }

        E::Var { name } => {
            let op = Op::GetLocal(*name);
            fb.assign(expr.t, op)
        }
        E::BinOp { op, lhs, rhs } => {
            let v1 = lower_expr(cx, fb, &*lhs);
            let v2 = lower_expr(cx, fb, &*rhs);
            let o = Op::BinOpI32(*op, v1, v2);
            fb.assign(expr.t, o)
        }
        E::UniOp { op, rhs } => match op {
            // Turn -x into 0 - x
            hir::UOp::Neg => {
                let v1 = fb.assign(cx.i32(), Op::ValI32(0));
                let v2 = lower_expr(cx, fb, &*rhs);
                let op = Op::BinOpI32(hir::BOp::Sub, v1, v2);
                fb.assign(expr.t, op)
            }
            hir::UOp::Not => {
                let v = lower_expr(cx, fb, &*rhs);
                let op = Op::UniOpI32(hir::UOp::Not, v);
                fb.assign(expr.t, op)
            }
            _ => todo!(),
        },
        E::Block { body } => {
            // If the body is empty, just make an unreachable value
            // that is used nowhere
            // If it is used as something other than unit then
            // this (hopefully) doesn't typecheck.
            lower_exprs(cx, fb, &*body).unwrap_or_else(|| fb.assign(cx.unit(), Op::ValUnit))
        }
        E::Let {
            varname,
            typename,
            init,
            mutable,
        } => {
            let v = lower_expr(cx, fb, &*init);
            fb.add_var(*varname, *typename, v, *mutable);
            fb.assign(expr.t, Op::SetLocal(*varname, v))
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
            // Returns the ID of the fhirst check_cond basic block.
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
                    let cond_bb = fb.get_current_block().id;
                    let cond_result = lower_expr(cx, fb, &cond);

                    let body_bb = fb.next_block().id;
                    let body_result = lower_exprs(cx, fb, &ifbody)
                        .unwrap_or_else(|| fb.assign(cx.unit(), Op::ValUnit));

                    let else_body_bb = fb.next_block().id;
                    let else_result = lower_exprs(cx, fb, elsebody)
                        .unwrap_or_else(|| fb.assign(cx.unit(), Op::ValUnit));

                    // Wire all the BB's together
                    fb.get_block(cond_bb).terminator =
                        Branch::Branch(cond_result, body_bb, else_body_bb);
                    fb.get_block(body_bb).terminator = Branch::Jump(body_result, end_bb);
                    fb.get_block(else_body_bb).terminator = Branch::Jump(else_result, end_bb);
                    // Accumulate the BB's that jump to "end" because we're going to have to put
                    // them in the phi
                    accm.push(body_bb);
                    accm.push(else_body_bb);
                    cond_bb
                } else {
                    // Build the next case...
                    let next_bb = recursive_build_bbs(cx, fb, end_bb, &cases[1..], elsebody, accm);
                    // Build our cond and body
                    let cond_bb = fb.next_block().id;
                    let cond_result = lower_expr(cx, fb, &cond);

                    let body_bb = fb.next_block().id;
                    let body_result = lower_exprs(cx, fb, &ifbody)
                        .unwrap_or_else(|| fb.assign(cx.unit(), Op::ValUnit));

                    // Wire together BB's
                    fb.get_block(cond_bb).terminator =
                        Branch::Branch(cond_result, body_bb, next_bb);
                    fb.get_block(body_bb).terminator = Branch::Jump(body_result, end_bb);
                    accm.push(body_bb);
                    cond_bb
                }
            }
            // This is the block that all the if branches feed into.
            let start_bb = fb.get_current_block().id;
            let end_bb = fb.next_block().id;
            // Make blocks for all the if branches
            let mut accm = vec![];
            let first_cond_bb = recursive_build_bbs(cx, fb, end_bb, cases, falseblock, &mut accm);

            // Add a phi instruction combining all the branch results
            let last_result = fb.assign(expr.t, Op::Phi(accm));
            // Save the end block
            // Connect the first BB to the start of the if block
            if let Some(last_value) = fb.get_block(start_bb).last_value() {
                fb.get_block(start_bb).terminator = Branch::Jump(last_value, first_cond_bb);
            } else {
                let last_value = fb.assign_block(start_bb, cx.unit(), Op::ValUnit);
                fb.get_block(start_bb).terminator = Branch::Jump(last_value, first_cond_bb);
            }

            last_result
        }
        E::Loop { .. } => todo!(),
        E::Lambda { .. } => {
            panic!("Unlifted lambda, should never happen?");
        }
        E::Funcall { func, params } => {
            // First, evaluate the args.
            let param_vars = params.iter().map(|e| lower_expr(cx, fb, e)).collect();
            // Then, issue the instruction depending on the type of call.
            match &func.e {
                E::Var { name } => fb.assign(expr.t, Op::Call(*name, param_vars)),
                _expr => {
                    // We got some other expr, compile it and do an indirect call 'cause it heckin'
                    // better be a function
                    let v = lower_expr(cx, fb, &func);
                    fb.assign(expr.t, Op::CallIndirect(v, param_vars))
                }
            }
        }
        E::Break => todo!(),
        E::Return { retval: _ } => todo!(),
        E::TupleCtor { body: _ } => todo!(),
        E::TupleRef { expr: _, elt: _ } => todo!(),

        E::Assign { lhs, rhs } => match &lhs.e {
            hir::Expr::Var { name } => {
                // Well, this at least is easy
                let res = lower_expr(cx, fb, rhs);
                // Look up var name.
                // TODO: Globals
                let mut binding = None;
                for b in fb.func.locals.iter().rev() {
                    if b.name == *name {
                        if !b.mutable {
                            unreachable!("Attempted to mutate immutable variable");
                        }
                        binding = Some(*b);
                    }
                }
                let binding = binding.expect("Var not found, should not happen");
                // We just add a new binding shadowing the old one.
                fb.add_var(binding.name, binding.typename, res, false);
                fb.assign(expr.t, Op::SetLocal(*name, res))
            }
            hir::Expr::TupleRef { .. } => todo!("FDSA"),
            _ => unreachable!(),
        },
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
        let hir = hir::lower(&ast);
        let hir = passes::run_passes(cx, hir);
        let checked =
            typeck::typecheck(cx, hir).unwrap_or_else(|e| panic!("Type check error: {}", e));
        lower_hir(cx, &checked)
    }

    #[test]
    fn lir_works_at_all() {
        let src = "fn foo(): I32 = 3 end";
        let lir = compile_lir(src);
        assert!(!lir.funcs.is_empty());
    }
}
