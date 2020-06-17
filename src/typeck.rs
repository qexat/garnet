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
        let msg = format!("Unknown var: {}", cx.fetch(name));
        Err(TypeError::UnknownVar(msg))
    }
}

impl Cx {
    /*
    /// Make the types of two terms equivalent, or produce an error if they're in conflict
    /// TODO: Figure out how to use this
    pub fn unify(&mut self, a: TypeInfo, b: TypeInfo) -> Result<(), TypeError> {
        use TypeInfo::*;
        match (a, b) {
            // Follow references
            (Ref(a), _) => self.unify(a, b),
            (_, Ref(b)) => self.unify(a, b),

            // When we don't know about a type, assume they match and
            // make the one know nothing about refer to the one we may
            // know something about
            (Unknown, _) => Ok(()),
            (_, Unknown) => Ok(()),

            (Known(t1), Known(t2)) if t1 == t2 => Ok(()),
            (Known(t1), Known(t2)) => Err(TypeError::TypeMismatch(format!(
                "type mismatch: {} != {}",
                t1, t2
            ))),

            /*
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
            */
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
                let _rettype = self.reconstruct(*r)?;
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
    */
}

/// Does t1 equal t2?
///
/// Currently we have no covariance or contravariance, so this is pretty simple.
/// Currently it's just, if the structures match, the types match.
fn type_matches(t1: &TypeSym, t2: &TypeSym) -> bool {
    t1 == t2
}

pub fn typecheck(cx: &mut Cx, ir: ir::Ir<()>) -> Result<ir::Ir<TypeSym>, TypeError> {
    let symtbl = &mut Symtbl::new();
    let checked_decls = ir
        .decls
        .into_iter()
        .map(|decl| typecheck_decl(cx, symtbl, decl))
        .collect::<Result<Vec<ir::Decl<TypeSym>>, TypeError>>()?;
    Ok(ir::Ir {
        decls: checked_decls,
    })
}

/// Typechecks a single decl
pub fn typecheck_decl(
    cx: &mut Cx,
    symtbl: &mut Symtbl,
    decl: ir::Decl<()>,
) -> Result<ir::Decl<TypeSym>, TypeError> {
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
            let typechecked_exprs = typecheck_exprs(cx, symtbl, body)?;
            // TODO: This only works if there are no return statements.
            let last_expr_type = last_type_of(cx, &typechecked_exprs);

            if !type_matches(&signature.rettype, &last_expr_type) {
                let msg = format!(
                    "Function {} returns {} but should return {}",
                    cx.fetch(name),
                    cx.fetch_type(last_expr_type).get_name(),
                    cx.fetch_type(signature.rettype).get_name(),
                );
                return Err(TypeError::TypeMismatch(msg));
            }

            symtbl.pop_scope();
            Ok(ir::Decl::Function {
                name,
                signature,
                body: typechecked_exprs,
            })
        }
        ir::Decl::Const {
            name,
            typename,
            init,
        } => {
            symtbl.add_var(name, &typename);
            Ok(ir::Decl::Const {
                name,
                typename,
                init: typecheck_expr(cx, symtbl, init)?,
            })
        }
    }
}

/// Typecheck a vec of expr's and return the type of the last one.
/// If the vec is empty, return unit.
fn typecheck_exprs(
    cx: &mut Cx,
    symtbl: &mut Symtbl,
    exprs: Vec<ir::TypedExpr<()>>,
) -> Result<Vec<ir::TypedExpr<TypeSym>>, TypeError> {
    exprs
        .into_iter()
        .map(|e| typecheck_expr(cx, symtbl, e))
        .collect()
}

/// Takes a slice of typed expr's and retursn the type of the last one.
/// Returns unit if the slice is empty.
fn last_type_of(cx: &Cx, exprs: &[ir::TypedExpr<TypeSym>]) -> TypeSym {
    exprs
        .last()
        .map(|e| e.t)
        .unwrap_or_else(|| cx.intern_type(&TypeDef::Tuple(vec![])))
}

fn typecheck_expr(
    cx: &mut Cx,
    symtbl: &mut Symtbl,
    expr: ir::TypedExpr<()>,
) -> Result<ir::TypedExpr<TypeSym>, TypeError> {
    use ir::Expr::*;
    let unittype = cx.intern_type(&TypeDef::Tuple(vec![]));
    let booltype = cx.intern_type(&TypeDef::Bool);
    match expr.e {
        Lit { val } => {
            let t = typecheck_literal(cx, &val)?;
            Ok(ir::TypedExpr { e: Lit { val }, t })
        }

        Var { name } => {
            let t = symtbl.get_var(cx, name)?;
            Ok(expr.map(t))
        }
        BinOp { op, lhs, rhs } => {
            let lhs = Box::new(typecheck_expr(cx, symtbl, *lhs)?);
            let rhs = Box::new(typecheck_expr(cx, symtbl, *rhs)?);
            // Currently, our only valid binops are on numbers.
            let binop_type = op.type_of(cx);
            if type_matches(&lhs.t, &rhs.t) && type_matches(&binop_type, &lhs.t) {
                Ok(ir::TypedExpr {
                    e: BinOp { op, lhs, rhs },
                    t: binop_type,
                })
            } else {
                let msg = format!(
                    "Invalid types for BOp {:?}: expected {}, got {} + {}",
                    op,
                    cx.fetch_type(binop_type).get_name(),
                    cx.fetch_type(lhs.t).get_name(),
                    cx.fetch_type(rhs.t).get_name()
                );
                Err(TypeError::TypeMismatch(msg))
            }
        }
        UniOp { op, rhs } => {
            let rhs = Box::new(typecheck_expr(cx, symtbl, *rhs)?);
            // Currently, our only valid binops are on numbers.
            let uniop_type = op.type_of(cx);
            if type_matches(&uniop_type, &rhs.t) {
                Ok(ir::TypedExpr {
                    e: UniOp { op, rhs },
                    t: uniop_type,
                })
            } else {
                let msg = format!(
                    "Invalid types for UOp {:?}: expected {}, got {}",
                    op,
                    cx.fetch_type(uniop_type).get_name(),
                    cx.fetch_type(rhs.t).get_name()
                );
                Err(TypeError::TypeMismatch(msg))
            }
        }
        Block { body } => {
            let b = typecheck_exprs(cx, symtbl, body)?;
            let t = last_type_of(cx, &b);
            Ok(ir::TypedExpr {
                e: Block { body: b },
                t: t,
            })
        }
        Let {
            varname,
            typename,
            init,
        } => {
            let init_expr = typecheck_expr(cx, symtbl, *init)?;
            let init_type = init_expr.t;
            if type_matches(&init_type, &typename) {
                // Add var to symbol table, proceed
                symtbl.add_var(varname, &typename);
                Ok(ir::TypedExpr {
                    e: Let {
                        varname,
                        typename,
                        init: Box::new(init_expr),
                    },
                    t: unittype,
                })
            } else {
                let msg = format!(
                    "initializer for variable {}: expected {}, got {}",
                    cx.fetch(varname),
                    cx.fetch_type(typename).get_name(),
                    cx.fetch_type(init_type).get_name()
                );
                Err(TypeError::TypeMismatch(msg))
            }
        }
        If { cases, falseblock } => {
            let falseblock = typecheck_exprs(cx, symtbl, falseblock)?;
            let assumed_type = last_type_of(cx, &falseblock);
            let mut new_cases = vec![];
            for (cond, body) in cases {
                let cond_expr = typecheck_expr(cx, symtbl, cond)?;
                if type_matches(&cond_expr.t, &booltype) {
                    // Proceed to typecheck arms
                    let ifbody_exprs = typecheck_exprs(cx, symtbl, body)?;
                    let if_type = last_type_of(cx, &ifbody_exprs);
                    if !type_matches(&if_type, &assumed_type) {
                        let msg = format!(
                            "If block return type is {}, but else block returns {}",
                            cx.fetch_type(if_type).get_name(),
                            cx.fetch_type(assumed_type).get_name(),
                        );
                        return Err(TypeError::TypeMismatch(msg));
                    }

                    // Great, it matches
                    new_cases.push((cond_expr, ifbody_exprs));
                } else {
                    let msg = format!(
                        "If expr condition is {}, not bool",
                        cx.fetch_type(cond_expr.t).get_name(),
                    );
                    return Err(TypeError::TypeMismatch(msg));
                }
            }
            Ok(ir::TypedExpr {
                t: assumed_type,
                e: If {
                    cases: new_cases,
                    falseblock,
                },
            })
        }
        Loop { body } => {
            let b = typecheck_exprs(cx, symtbl, body)?;
            let t = last_type_of(cx, &b);
            Ok(ir::TypedExpr {
                e: Loop { body: b },
                t: t,
            })
        }
        Lambda { signature, body } => {
            symtbl.push_scope();
            // add params to symbol table
            for (paramname, paramtype) in signature.params.iter() {
                symtbl.add_var(*paramname, paramtype);
            }
            let body_expr = typecheck_exprs(cx, symtbl, body)?;
            let bodytype = last_type_of(cx, &body_expr);
            // TODO: validate/unify types
            if !type_matches(&bodytype, &signature.rettype) {
                let msg = format!(
                    "Function returns type {:?} but should be {:?}",
                    cx.fetch_type(bodytype),
                    cx.fetch_type(signature.rettype)
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
            Ok(ir::TypedExpr {
                e: Lambda {
                    signature,
                    body: body_expr,
                },
                t: lambdatype,
            })
        }
        Funcall { func, params } => {
            // First, get param types
            let given_params = typecheck_exprs(cx, symtbl, params)?;
            // Then, look up function
            let f = typecheck_expr(cx, symtbl, *func)?;
            let fdef = &*cx.fetch_type(f.t);
            match fdef {
                TypeDef::Lambda(paramtypes, rettype) => {
                    // Now, make sure all the function's params match what it wants
                    for (given, wanted) in given_params.iter().zip(paramtypes) {
                        if !type_matches(&given.t, &wanted) {
                            let msg = format!(
                                "Function wanted type {} in param but got type {}",
                                cx.fetch_type(*wanted).get_name(),
                                cx.fetch_type(given.t).get_name()
                            );
                            return Err(TypeError::TypeMismatch(msg));
                        }
                    }
                    Ok(ir::TypedExpr {
                        e: Funcall {
                            func: Box::new(f),
                            params: given_params,
                        },
                        t: **rettype,
                    })
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
        Break => Ok(expr.map(unittype)),
        Return { .. } => unimplemented!("TODO: Never type"),
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
            let ir = *plz(Expr::BinOp {
                op: BOp::Add,
                lhs: plz(Expr::Lit {
                    val: Literal::Integer(3),
                }),
                rhs: plz(Expr::Lit {
                    val: Literal::Integer(4),
                }),
            });
            assert!(type_matches(
                &typecheck_expr(cx, tbl, ir).unwrap().t,
                &t_i32
            ));

            let bad_ir = *plz(Expr::BinOp {
                op: BOp::Add,
                lhs: plz(Expr::Lit {
                    val: Literal::Integer(3),
                }),
                rhs: plz(Expr::Lit {
                    val: Literal::Bool(false),
                }),
            });
            assert!(typecheck_expr(cx, tbl, bad_ir).is_err());
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
            let ir = *plz(Expr::UniOp {
                op: UOp::Neg,
                rhs: plz(Expr::Lit {
                    val: Literal::Integer(4),
                }),
            });
            assert!(type_matches(
                &typecheck_expr(cx, tbl, ir).unwrap().t,
                &t_i32
            ));

            let bad_ir = *plz(Expr::UniOp {
                op: UOp::Neg,
                rhs: plz(Expr::Lit {
                    val: Literal::Bool(false),
                }),
            });
            assert!(typecheck_expr(cx, tbl, bad_ir).is_err());
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
        let fooname = cx.intern("foo");

        use ir::*;
        {
            let ir = *plz(Expr::Let {
                varname: fooname,
                typename: t_i32.clone(),
                init: plz(Expr::Lit {
                    val: Literal::Integer(42),
                }),
            });
            assert!(type_matches(
                &typecheck_expr(cx, tbl, ir).unwrap().t,
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
        let fname = cx.intern("foo");
        let aname = cx.intern("a");
        let bname = cx.intern("b");
        let ftype = cx.intern_type(&TypeDef::Lambda(vec![t_i32, t_i32], Box::new(t_i32)));

        use ir::*;
        {
            let ir = vec![
                *plz(Expr::Let {
                    varname: fname,
                    typename: ftype,
                    init: plz(Expr::Lambda {
                        signature: Signature {
                            params: vec![(aname, t_i32), (bname, t_i32)],
                            rettype: t_i32,
                        },
                        body: vec![*plz(Expr::BinOp {
                            op: BOp::Add,
                            lhs: plz(Expr::Var { name: aname }),
                            rhs: plz(Expr::Var { name: bname }),
                        })],
                    }),
                }),
                *plz(Expr::Funcall {
                    func: plz(Expr::Var { name: fname }),
                    params: vec![*plz(Expr::int(3)), *plz(Expr::int(4))],
                }),
            ];
            let exprs = &typecheck_exprs(cx, tbl, ir).unwrap();
            assert!(type_matches(&last_type_of(cx, exprs), &t_i32));
            // Is the variable now bound in our symbol table?
            assert_eq!(tbl.get_var(cx, fname).unwrap(), ftype);
        }
    }

    /// TODO
    #[test]
    fn test_return() {}
}
