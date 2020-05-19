//! Typechecking and other semantic checking.
//! Operates on the frontend IR.

use std::collections::HashMap;

use crate::intern::Sym;
use crate::ir;
use crate::Cx;

pub enum TypeError {
    UnknownType(String),
    Misc(String),
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

fn typecheck_expr(cx: &mut Cx, symtbl: &mut Symtbl, expr: &ir::Expr) -> Result<(), TypeError> {
    use ir::Expr::*;
    match expr {
        Lit { val } => (),
        Var { name } => (),
        BinOp { op, lhs, rhs } => (),
        UniOp { op, rhs } => (),
        Block { body } => (),
        Let {
            varname,
            typename,
            init,
        } => (),
        If {
            condition,
            trueblock,
            falseblock,
        } => (),
        Loop { body } => (),
        Lambda { signature, body } => (),
        Funcall { func, params } => (),
        Break => (),
        Return { retval } => (),
    }
    Ok(())
}
