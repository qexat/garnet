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
    TypeMismatch(String),
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
    ///
    /// TODO: cx is only for error message, can we fix?
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
fn type_matches(t1: Sym, t2: Sym) -> bool {
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
            // This is squirrelly; basically, we want to return unit
            // if the function has no body, otherwise return the
            // type of the last expression.
            //
            // Oh gods, what in the name of Eris do we do if there's
            // a return statement here?
            let mut last_expr_type = cx.intern("unit");
            for expr in body {
                last_expr_type = typecheck_expr(cx, symtbl, expr)?;
            }

            if !type_matches(signature.rettype.name.0, last_expr_type) {
                let msg = format!(
                    "Function {} returns {} but should return {}",
                    cx.unintern(name.0),
                    cx.unintern(last_expr_type),
                    cx.unintern(signature.rettype.name.0),
                );
                return Err(TypeError::TypeMismatch(msg));
            }

            symtbl.pop_scope();
        }
    }
    Ok(())
}

/// Returns a symbol representing the type of this expression, if it's ok, or an error if not.
fn typecheck_expr(cx: &mut Cx, symtbl: &mut Symtbl, expr: &ir::Expr) -> Result<Sym, TypeError> {
    use ir::Expr::*;
    let unitsym = cx.intern("()");
    assert!(type_exists(cx, unitsym));
    let unit = Ok(unitsym);
    let ok = Ok(unitsym);
    match expr {
        Lit { val } => typecheck_literal(cx, val),
        Var { name } => symtbl.get_var(cx, name.0),
        BinOp { op, lhs, rhs } => {
            let tlhs = typecheck_expr(cx, symtbl, lhs)?;
            let trhs = typecheck_expr(cx, symtbl, rhs)?;
            // Currently, our only valid binops are on numbers.
            let binop_type = op.type_of(cx);
            if type_matches(tlhs, trhs) && type_matches(binop_type, tlhs) {
                Ok(binop_type)
            } else {
                let msg = format!(
                    "Invalid types for BOp {:?}: expected {}, got {} + {}",
                    op,
                    cx.unintern(binop_type),
                    cx.unintern(tlhs),
                    cx.unintern(trhs)
                );
                Err(TypeError::TypeMismatch(msg))
            }
        }
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
        Break => unit,
        Return { .. } => panic!("oh gods oh gods oh gods oh gods whyyyyyy"),
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

    /// Test symbol table
    #[test]
    fn test_symtbl() {
        let cx = &mut crate::Cx::new();
        let t_foo = cx.intern("foo");
        let t_bar = cx.intern("bar");
        let t_i32 = cx.intern("i32");
        let t_bool = cx.intern("bool");
        let mut t = Symtbl::new();

        // Make sure we can get a value
        assert!(t.get_var(cx, t_foo).is_err());
        assert!(t.get_var(cx, t_bar).is_err());
        t.add_var(t_foo, t_i32);
        assert_eq!(t.get_var(cx, t_foo).unwrap(), t_i32);
        assert!(t.get_var(cx, t_bar).is_err());

        // Push scope, verify behavior is the same.
        t.push_scope();
        assert_eq!(t.get_var(cx, t_foo).unwrap(), t_i32);
        assert!(t.get_var(cx, t_bar).is_err());
        // Add var, make sure we can get it
        t.add_var(t_bar, t_bool);
        assert_eq!(t.get_var(cx, t_foo).unwrap(), t_i32);
        assert_eq!(t.get_var(cx, t_bar).unwrap(), t_bool);

        // Pop scope, make sure bar is gone.
        t.pop_scope();
        assert_eq!(t.get_var(cx, t_foo).unwrap(), t_i32);
        assert!(t.get_var(cx, t_bar).is_err());
    }

    /// Make sure an empty symtbl gives errors
    #[test]
    #[should_panic]
    fn test_symtbl_underflow() {
        let mut t = Symtbl::new();
        t.pop_scope();
        t.pop_scope();
    }

    /// Test literal type inference
    #[test]
    fn test_type_lit() {
        let cx = &mut crate::Cx::new();
        let t_i32 = cx.intern("i32");
        let t_bool = cx.intern("bool");
        let t_unit = cx.intern("()");
        assert!(type_exists(cx, t_i32));
        assert!(type_exists(cx, t_bool));
        assert!(type_exists(cx, t_unit));

        let l1 = ir::Literal::Integer(3);
        let l1t = typecheck_literal(cx, &l1).unwrap();
        let l2 = ir::Literal::Bool(true);
        let l2t = typecheck_literal(cx, &l2).unwrap();
        let l3 = ir::Literal::Unit;
        let l3t = typecheck_literal(cx, &l3).unwrap();
        assert!(!type_matches(l1t, l2t));
        assert!(!type_matches(l1t, l3t));
        assert!(!type_matches(l2t, l3t));

        assert!(type_matches(l1t, t_i32));
        assert!(type_matches(l2t, t_bool));
        assert!(type_matches(l3t, t_unit));
    }

    /// Test binop typechecks
    #[test]
    fn test_binop() {
        let cx = &mut crate::Cx::new();
        let t_i32 = cx.intern("i32");
        let t_bool = cx.intern("bool");
        let tbl = &mut Symtbl::new();

        use ir::*;
        // Basic addition
        {
            let ir = Expr::BinOp {
                op: BOp::Add,
                lhs: Box::new(Expr::Lit {
                    val: Literal::Integer(3),
                }),
                rhs: Box::new(Expr::Lit {
                    val: Literal::Integer(4),
                }),
            };
            assert!(type_matches(typecheck_expr(cx, tbl, &ir).unwrap(), t_i32));

            let bad_ir = Expr::BinOp {
                op: BOp::Add,
                lhs: Box::new(Expr::Lit {
                    val: Literal::Integer(3),
                }),
                rhs: Box::new(Expr::Lit {
                    val: Literal::Bool(false),
                }),
            };
            assert!(typecheck_expr(cx, tbl, &bad_ir).is_err());
        }
    }
}