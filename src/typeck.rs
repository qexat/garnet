//! Typechecking and other semantic checking.
//! Operates on the frontend IR.

use std::collections::HashMap;

use crate::intern::Sym;
use crate::ir;
use crate::Cx;

pub enum TypeError {
    UnknownType(String),
}

/// A variable binding
#[derive(Debug, Clone)]
pub struct VarBinding {
    name: Sym,
    typename: Sym,
}

/// Symbol table.  Stores the scope stack and variable bindings.
pub struct Symtbl {
    syms: Vec<HashMap<Sym, VarBinding>>,
}

impl Symtbl {
    pub fn new() -> Self {
        Self {
            syms: vec![HashMap::new()],
        }
    }

    fn push_scope(&mut self) {
        self.syms.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.syms
            .pop()
            .expect("Scope underflow; should not happen!");
    }

    /// Add a variable to the top level of the scope.
    /// Allows shadowing.
    fn add_var(&mut self, name: Sym, typename: Sym) {
        let tbl = self
            .syms
            .last_mut()
            .expect("Scope underflow while adding var; should not happen");
        let binding = VarBinding { name, typename };
        tbl.insert(name, binding);
    }
}

/// Does this type exist in the type table?
fn type_exists(cx: &Cx, t: Sym) -> bool {
    cx.types.contains_key(&t)
}

/// Does t1 equal t2?
///
/// Currently we have no covariance or contravariance, so this is pretty simple.
/// Currently it's just, if the names match, the types match.
fn type_matches(cx: &Cx, t1: Sym, t2: Sym) -> bool {
    t1 == t2
}

pub fn typecheck(cx: &mut Cx, ir: &ir::Ir) {}

/// Typechecks a single decl
pub fn typecheck_decl(cx: &mut Cx, symtbl: &mut Symtbl, decl: &ir::Decl) -> Result<(), TypeError> {
    match decl {
        ir::Decl::Function {
            name,
            signature,
            body,
        } => {
            // Push scope, typecheck and add params to symbol table
            symtbl.push_scope();
            // TODO: Add the function itself!
            for (pname, ptype) in signature.params.iter() {
                if !type_exists(cx, ptype.name.0) {
                    let msg = format!(
                        "Unknown type {} in function param {}",
                        cx.unintern(ptype.name.0),
                        cx.unintern(pname.0)
                    );
                    return Err(TypeError::UnknownType(msg));
                }
                symtbl.add_var(pname.0, ptype.name.0);
            }
            for expr in body {
                typecheck_expr(cx, symtbl, expr)?;
            }

            symtbl.pop_scope();
        }
    }
    Ok(())
}

/// Returns a symbol representing the type of this expression, if it's ok, or an error if not.
fn typecheck_expr(cx: &mut Cx, symtbl: &mut Symtbl, expr: &ir::Expr) -> Result<Sym, TypeError> {
    use ir::Expr::*;
    let unit = cx.intern("()");
    assert!(type_exists(cx, unit));
    let ok = Ok(unit);
    match expr {
        Lit { val } => typecheck_literal(cx, val),
        Var { name } => ok,
        BinOp { op, lhs, rhs } => ok,
        UniOp { op, rhs } => ok,
        Block { body } => ok,
        Let {
            varname,
            typename,
            init,
        } => ok,
        If {
            condition,
            trueblock,
            falseblock,
        } => ok,
        Loop { body } => ok,
        Lambda { signature, body } => ok,
        Funcall { func, params } => ok,
        Break => ok,
        Return { retval } => ok,
    }
}

fn typecheck_literal(cx: &mut Cx, lit: &ir::Literal) -> Result<Sym, TypeError> {
    let t_i32 = cx.intern("i32");
    let t_bool = cx.intern("bool");
    let t_unit = cx.intern("()");
    assert!(type_exists(cx, t_i32));
    assert!(type_exists(cx, t_bool));
    assert!(type_exists(cx, t_unit));
    match lit {
        ir::Literal::Integer(_) => Ok(t_i32),
        ir::Literal::Bool(_) => Ok(t_bool),
        ir::Literal::Unit => Ok(t_unit),
    }
}
