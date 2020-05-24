//! Typechecking and other semantic checking.
//! Operates on the frontend IR.

use std::collections::HashMap;

use crate::ir::{self};
use crate::{Cx, TypeDef, TypeSym, VarSym};

#[derive(Debug, Clone)]
pub enum TypeError {
    UnknownType(String),
    UnknownVar(String),
    TypeMismatch(String),
    InferenceFailure(String),
}

/// A variable binding
#[derive(Debug, Clone)]
pub struct VarBinding {
    name: VarSym,
    typename: TypeSym,
}

/// Symbol table.  Stores the scope stack and variable bindings.
pub struct Symtbl {
    syms: Vec<HashMap<VarSym, VarBinding>>,
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
    fn add_var(&mut self, name: VarSym, typedef: &TypeSym) {
        let tbl = self
            .syms
            .last_mut()
            .expect("Scope underflow while adding var; should not happen");
        let binding = VarBinding {
            name,
            typename: typedef.clone(),
        };
        tbl.insert(name, binding);
    }

    /// Get the type of the given variable, or an error
    ///
    /// TODO: cx is only for error message, can we fix?
    fn get_var(&self, cx: &Cx, name: VarSym) -> Result<TypeSym, TypeError> {
        for scope in self.syms.iter().rev() {
            if let Some(binding) = scope.get(&name) {
                return Ok(binding.typename.clone());
            }
        }
        let msg = format!("Unknown var: {}", cx.unintern(name));
        Err(TypeError::UnknownVar(msg))
    }
}

impl Cx {
    /// Make the types of two terms equivalent, or produce an error if they're in conflict
    fn unify(&mut self, a: TypeSym, b: TypeSym) -> Result<(), TypeError> {
        let ta = self.unintern_type(a).clone();
        let tb = self.unintern_type(b).clone();
        use TypeDef::*;
        match (ta, tb) {
            // Follow references
            (Ref(a), _) => self.unify(a, b),
            (_, Ref(b)) => self.unify(a, b),

            // When we don't know about a type, assume they match and
            // make the one know nothing about refer to the one we may
            // know something about
            (Unknown, _) => Ok(()),
            (_, Unknown) => Ok(()),

            // Primitives are easy to unify
            (SInt(sa), SInt(sb)) if sa == sb => Ok(()),
            (SInt(sa), SInt(sb)) => Err(TypeError::TypeMismatch(format!(
                "Integer mismatch: {} != {}",
                sa, sb
            ))),
            (Bool, Bool) => Ok(()),

            // For complex types, we must unify their sub-types.
            (Tuple(ta), Tuple(tb)) => {
                for (a, b) in ta.iter().zip(tb) {
                    self.unify(*a, b)?;
                }
                Ok(())
            }
            (Lambda(pa, ra), Lambda(pb, rb)) => {
                for (a, b) in pa.iter().zip(pb) {
                    self.unify(*a, b)?;
                }
                self.unify(*ra, *rb)
            }

            // No attempt to match was successful, error.
            (a, b) => Err(TypeError::InferenceFailure(format!(
                "Could not unify types: {:?} and {:?}",
                a, b
            ))),
        }
    }

    /// Attempt to reconstruct a concrete type from a symbol.  This may
    /// fail if we don't have enough info to figure out what the type is.
    pub fn reconstruct(&self, sym: TypeSym) -> Result<TypeDef, TypeError> {
        let t = self.unintern_type(sym).clone();
        use TypeDef::*;
        match t {
            Unknown => Err(TypeError::InferenceFailure(format!(
                "No type for {:?}",
                sym
            ))),
            Ref(id) => self.reconstruct(id),
            SInt(i) => Ok(SInt(i)),
            Bool => Ok(Bool),
            Lambda(p, r) => {
                let mut ts = vec![];
                for ty in p {
                    ts.push(self.reconstruct(ty)?);
                }
                let rettype = self.reconstruct(*r)?;
                //Ok(Lambda(ts, rettype))

                Ok(Bool)
            }
            Tuple(t) => {
                let mut ts = vec![];
                for ty in t {
                    ts.push(self.reconstruct(ty)?);
                }
                //Ok(Tuple(ts))
                Ok(Bool)
            }
        }
    }
}

/// Does t1 equal t2?
///
/// Currently we have no covariance or contravariance, so this is pretty simple.
/// Currently it's just, if the structures match, the types match.
fn type_matches(t1: &TypeSym, t2: &TypeSym) -> bool {
    t1 == t2
}

pub fn typecheck(cx: &mut Cx, ir: &ir::Ir) -> Result<(), TypeError> {
    let symtbl = &mut Symtbl::new();
    for decl in ir.decls.iter() {
        typecheck_decl(cx, symtbl, decl)?;
    }
    Ok(())
}

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
                symtbl.add_var(*pname, ptype);
            }
            // This is squirrelly; basically, we want to return unit
            // if the function has no body, otherwise return the
            // type of the last expression.
            //
            // Oh gods, what in the name of Eris do we do if there's
            // a return statement here?
            let last_expr_type = typecheck_exprs(cx, symtbl, body)?;

            if !type_matches(&signature.rettype, &last_expr_type) {
                let msg = format!(
                    "Function {} returns {} but should return {}",
                    cx.unintern(*name),
                    cx.unintern_type(last_expr_type).get_name(),
                    cx.unintern_type(signature.rettype).get_name(),
                );
                return Err(TypeError::TypeMismatch(msg));
            }

            symtbl.pop_scope();
        }
        ir::Decl::Const {
            name,
            typename,
            init: _,
        } => {
            symtbl.add_var(*name, typename);
        }
    }
    Ok(())
}

/// Typecheck a vec of expr's and return the type of the last one.
/// If the vec is empty, return unit.
fn typecheck_exprs(
    cx: &mut Cx,
    symtbl: &mut Symtbl,
    exprs: &[ir::Expr],
) -> Result<TypeSym, TypeError> {
    let mut last_expr_type = cx.intern_type(&TypeDef::Tuple(vec![]));
    for expr in exprs {
        last_expr_type = typecheck_expr(cx, symtbl, expr)?;
    }
    Ok(last_expr_type)
}

/// Returns a symbol representing the type of this expression, if it's ok, or an error if not.
fn typecheck_expr(cx: &mut Cx, symtbl: &mut Symtbl, expr: &ir::Expr) -> Result<TypeSym, TypeError> {
    use ir::Expr::*;
    let unittype = cx.intern_type(&TypeDef::Tuple(vec![]));
    let booltype = cx.intern_type(&TypeDef::Bool);
    match expr {
        Lit { val } => typecheck_literal(cx, val),
        Var { name } => symtbl.get_var(cx, *name),
        BinOp { op, lhs, rhs } => {
            let tlhs = typecheck_expr(cx, symtbl, lhs)?;
            let trhs = typecheck_expr(cx, symtbl, rhs)?;
            // Currently, our only valid binops are on numbers.
            let binop_type = op.type_of(cx);
            if type_matches(&tlhs, &trhs) && type_matches(&binop_type, &tlhs) {
                Ok(binop_type)
            } else {
                let msg = format!(
                    "Invalid types for BOp {:?}: expected {}, got {} + {}",
                    op,
                    cx.unintern_type(binop_type).get_name(),
                    cx.unintern_type(tlhs).get_name(),
                    cx.unintern_type(trhs).get_name()
                );
                Err(TypeError::TypeMismatch(msg))
            }
        }
        UniOp { op, rhs } => {
            let trhs = typecheck_expr(cx, symtbl, rhs)?;
            // Currently, our only valid binops are on numbers.
            let uniop_type = op.type_of(cx);
            if type_matches(&uniop_type, &trhs) {
                Ok(uniop_type)
            } else {
                let msg = format!(
                    "Invalid types for UOp {:?}: expected {}, got {}",
                    op,
                    cx.unintern_type(uniop_type).get_name(),
                    cx.unintern_type(trhs).get_name()
                );
                Err(TypeError::TypeMismatch(msg))
            }
        }
        Block { body } => typecheck_exprs(cx, symtbl, body),
        Let {
            varname,
            typename,
            init,
        } => {
            let init_type = typecheck_expr(cx, symtbl, init)?;
            if type_matches(&init_type, &typename) {
                // Add var to symbol table, proceed
                symtbl.add_var(*varname, typename);
                Ok(unittype)
            } else {
                let msg = format!(
                    "initializer for variable {}: expected {}, got {}",
                    cx.unintern(*varname),
                    cx.unintern_type(*typename).get_name(),
                    cx.unintern_type(init_type).get_name()
                );
                Err(TypeError::TypeMismatch(msg))
            }
        }
        If {
            condition,
            trueblock,
            falseblock,
        } => {
            let cond_type = typecheck_expr(cx, symtbl, condition)?;
            if type_matches(&cond_type, &booltype) {
                // Proceed to typecheck arms
                let if_type = typecheck_exprs(cx, symtbl, trueblock)?;
                let else_type = typecheck_exprs(cx, symtbl, falseblock)?;
                if type_matches(&if_type, &else_type) {
                    Ok(if_type)
                } else {
                    let msg = format!(
                        "If block return type is {}, but else block returns {}",
                        cx.unintern_type(if_type).get_name(),
                        cx.unintern_type(else_type).get_name(),
                    );
                    Err(TypeError::TypeMismatch(msg))
                }
            } else {
                let msg = format!(
                    "If expr condition is {}, not bool",
                    cx.unintern_type(cond_type).get_name(),
                );
                Err(TypeError::TypeMismatch(msg))
            }
        }
        Loop { body } => typecheck_exprs(cx, symtbl, body),
        Lambda { signature, body } => {
            symtbl.push_scope();
            // add params to symbol table
            for (paramname, paramtype) in signature.params.iter() {
                symtbl.add_var(*paramname, paramtype);
            }
            let bodytype = typecheck_exprs(cx, symtbl, body)?;
            // TODO: validate/unify types
            if !type_matches(&bodytype, &signature.rettype) {
                let msg = format!(
                    "Function returns type {:?} but should be {:?}",
                    cx.unintern_type(bodytype),
                    cx.unintern_type(signature.rettype)
                );
                return Err(TypeError::TypeMismatch(msg));
            }
            symtbl.pop_scope();
            let paramtypes: Vec<TypeSym> = signature
                .params
                .iter()
                .map(|(_varsym, typesym)| *typesym)
                .collect();
            let lambdatype =
                cx.intern_type(&TypeDef::Lambda(paramtypes, Box::new(signature.rettype)));
            Ok(lambdatype)
        }
        Funcall { func, params } => {
            // First, get param types
            let given_param_types = params
                .iter()
                .map(|p| typecheck_expr(cx, symtbl, p))
                .collect::<Result<Vec<TypeSym>, _>>()?;
            // Then, look up function
            let f = typecheck_expr(cx, symtbl, func)?;
            let fdef = cx.unintern_type(f);
            match fdef {
                TypeDef::Lambda(paramtypes, rettype) => {
                    // Now, make sure all the function's params match what it wants
                    for (given, wanted) in given_param_types.iter().zip(paramtypes) {
                        if !type_matches(given, wanted) {
                            let msg = format!(
                                "Function wanted type {} in param but got type {}",
                                cx.unintern_type(*wanted).get_name(),
                                cx.unintern_type(*given).get_name()
                            );
                            return Err(TypeError::TypeMismatch(msg));
                        }
                    }
                    Ok(**rettype)
                }
                other => {
                    let msg = format!(
                        "Tried to call function but it is not a function, it is a {}",
                        other.get_name()
                    );
                    Err(TypeError::TypeMismatch(msg))
                }
            }
        }
        Break => Ok(unittype),
        Return { .. } => panic!("oh gods oh gods oh gods oh gods whyyyyyy"),
    }
}

fn typecheck_literal(cx: &mut Cx, lit: &ir::Literal) -> Result<TypeSym, TypeError> {
    let i32type = cx.intern_type(&TypeDef::SInt(4));
    let unittype = cx.intern_type(&TypeDef::Tuple(vec![]));
    let booltype = cx.intern_type(&TypeDef::Bool);
    match lit {
        ir::Literal::Integer(_) => Ok(i32type),
        ir::Literal::Bool(_) => Ok(booltype),
        ir::Literal::Unit => Ok(unittype),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;

    /// Sanity check
    #[test]
    fn test_typecheck_lit() {
        use ir;
        let cx = &mut crate::Cx::new();
        let t_i32 = cx.intern_type(&TypeDef::SInt(4));
        let t_bool = cx.intern_type(&TypeDef::Bool);
        let t_unit = cx.intern_type(&TypeDef::Tuple(vec![]));

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
        let t_i32 = cx.intern_type(&TypeDef::SInt(4));
        let t_bool = cx.intern_type(&TypeDef::Bool);
        let mut t = Symtbl::new();

        // Make sure we can get a value
        assert!(t.get_var(cx, t_foo).is_err());
        assert!(t.get_var(cx, t_bar).is_err());
        t.add_var(t_foo, &t_i32);
        assert_eq!(t.get_var(cx, t_foo).unwrap(), t_i32);
        assert!(t.get_var(cx, t_bar).is_err());

        // Push scope, verify behavior is the same.
        t.push_scope();
        assert_eq!(t.get_var(cx, t_foo).unwrap(), t_i32);
        assert!(t.get_var(cx, t_bar).is_err());
        // Add var, make sure we can get it
        t.add_var(t_bar, &t_bool);
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
        let t_i32 = cx.intern_type(&TypeDef::SInt(4));
        let t_bool = cx.intern_type(&TypeDef::Bool);
        let t_unit = cx.intern_type(&TypeDef::Tuple(vec![]));

        let l1 = ir::Literal::Integer(3);
        let l1t = typecheck_literal(cx, &l1).unwrap();
        let l2 = ir::Literal::Bool(true);
        let l2t = typecheck_literal(cx, &l2).unwrap();
        let l3 = ir::Literal::Unit;
        let l3t = typecheck_literal(cx, &l3).unwrap();
        assert!(!type_matches(&l1t, &l2t));
        assert!(!type_matches(&l1t, &l3t));
        assert!(!type_matches(&l2t, &l3t));

        assert!(type_matches(&l1t, &t_i32));
        assert!(type_matches(&l2t, &t_bool));
        assert!(type_matches(&l3t, &t_unit));
    }

    /// Test binop typechecks
    #[test]
    fn test_binop() {
        let cx = &mut crate::Cx::new();
        let t_i32 = cx.intern_type(&TypeDef::SInt(4));
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
            assert!(type_matches(&typecheck_expr(cx, tbl, &ir).unwrap(), &t_i32));

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

    /// Test binop typechecks
    #[test]
    fn test_uniop() {
        let cx = &mut crate::Cx::new();
        let t_i32 = cx.intern_type(&TypeDef::SInt(4));
        let tbl = &mut Symtbl::new();

        use ir::*;
        {
            let ir = Expr::UniOp {
                op: UOp::Neg,
                rhs: Box::new(Expr::Lit {
                    val: Literal::Integer(4),
                }),
            };
            assert!(type_matches(&typecheck_expr(cx, tbl, &ir).unwrap(), &t_i32));

            let bad_ir = Expr::UniOp {
                op: UOp::Neg,
                rhs: Box::new(Expr::Lit {
                    val: Literal::Bool(false),
                }),
            };
            assert!(typecheck_expr(cx, tbl, &bad_ir).is_err());
        }
    }

    /// TODO
    #[test]
    fn test_block() {}

    /// Do let expr's have the right return?
    #[test]
    fn test_let() {
        let cx = &mut crate::Cx::new();
        let tbl = &mut Symtbl::new();
        let t_i32 = cx.intern_type(&TypeDef::SInt(4));
        let t_unit = cx.intern_type(&TypeDef::Tuple(vec![]));
        let fooname = VarSym::new(cx, "foo");

        use ir::*;
        {
            let ir = Expr::Let {
                varname: fooname,
                typename: t_i32.clone(),
                init: Box::new(Expr::Lit {
                    val: Literal::Integer(42),
                }),
            };
            assert!(type_matches(
                &typecheck_expr(cx, tbl, &ir).unwrap(),
                &t_unit
            ));
            // Is the variable now bound in our symbol table?
            assert_eq!(tbl.get_var(cx, fooname).unwrap(), t_i32);
        }
    }

    /// TODO
    #[test]
    fn test_if() {}

    /// TODO
    #[test]
    fn test_loop() {}

    /// TODO
    #[test]
    fn test_break() {}

    /// TODO
    #[test]
    fn test_funcall() {
        let cx = &mut crate::Cx::new();
        let tbl = &mut Symtbl::new();
        let t_i32 = cx.intern_type(&TypeDef::SInt(4));
        let fname = VarSym::new(cx, "foo");
        let aname = VarSym::new(cx, "a");
        let bname = VarSym::new(cx, "b");
        let ftype = cx.intern_type(&TypeDef::Lambda(vec![t_i32, t_i32], Box::new(t_i32)));

        use ir::*;
        {
            let ir = vec![
                Expr::Let {
                    varname: fname,
                    typename: ftype,
                    init: Box::new(Expr::Lambda {
                        signature: Signature {
                            params: vec![(aname, t_i32), (bname, t_i32)],
                            rettype: t_i32,
                        },
                        body: vec![Expr::BinOp {
                            op: BOp::Add,
                            lhs: Box::new(Expr::Var { name: aname }),
                            rhs: Box::new(Expr::Var { name: bname }),
                        }],
                    }),
                },
                Expr::Funcall {
                    func: Box::new(Expr::Var { name: fname }),
                    params: vec![Expr::int(3), Expr::int(4)],
                },
            ];
            assert!(type_matches(
                &typecheck_exprs(cx, tbl, &ir).unwrap(),
                &t_i32
            ));
            // Is the variable now bound in our symbol table?
            assert_eq!(tbl.get_var(cx, fname).unwrap(), ftype);
        }
    }

    /// TODO
    #[test]
    fn test_return() {}
}
