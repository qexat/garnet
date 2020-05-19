//! Typechecking and other semantic checking.
//! Operates on the frontend IR.

use std::collections::HashMap;

use crate::intern::Sym;
use crate::ir;
use crate::Cx;

#[derive(Debug, Clone)]
pub enum TypeError {
    UnknownType(String),
    UnknownVar(String),
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

    /// Get the type of the given variable, or an error
    fn get_var(&self, cx: &Cx, name: Sym) -> Result<Sym, TypeError> {
        for scope in self.syms.iter().rev() {
            if let Some(binding) = scope.get(&name) {
                return Ok(binding.typename);
            }
        }
        let msg = format!("Unknown var: {}", cx.unintern(name));
        Err(TypeError::UnknownVar(msg))
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
            // TODO: How to handle return statements, hm?
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
        Var { name } => symtbl.get_var(cx, name.0),
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Sanity check
    #[test]
    fn test_typecheck_lit() {
        use ir;
        let cx = &mut crate::Cx::new();
        let t_i32 = cx.intern("i32");
        let t_bool = cx.intern("bool");
        let t_unit = cx.intern("()");
        assert!(type_exists(cx, t_i32));
        assert!(type_exists(cx, t_bool));
        assert!(type_exists(cx, t_unit));

        assert_eq!(
            typecheck_literal(cx, &ir::Literal::Integer(9)).unwrap(),
            t_i32
        );
        assert_eq!(
            typecheck_literal(cx, &ir::Literal::Bool(false)).unwrap(),
            t_bool
        );
        assert_eq!(typecheck_literal(cx, &ir::Literal::Unit).unwrap(), t_unit);
    }
}
