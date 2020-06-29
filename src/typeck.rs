//! Typechecking and other semantic checking.
//! Operates on the IR.

use std::collections::HashMap;

use crate::ir;
use crate::{Cx, TypeDef, TypeSym, VarSym};

#[derive(Debug, Clone)]
pub enum TypeError {
    UnknownVar(VarSym),
    ReturnMismatch {
        fname: VarSym,
        got: TypeSym,
        expected: TypeSym,
    },
    BopTypeMismatch {
        bop: ir::BOp,
        got1: TypeSym,
        got2: TypeSym,
        expected: TypeSym,
    },
    UopTypeMismatch {
        op: ir::UOp,
        got: TypeSym,
        expected: TypeSym,
    },
    LetTypeMismatch {
        name: VarSym,
        got: TypeSym,
        expected: TypeSym,
    },
    IfTypeMismatch {
        ifpart: TypeSym,
        elsepart: TypeSym,
    },
    CondMismatch {
        got: TypeSym,
    },
    ParamMismatch {
        got: TypeSym,
        expected: TypeSym,
    },
    CallMismatch {
        got: TypeSym,
    },
    TupleRefMismatch {
        got: TypeSym,
    },
}

impl TypeError {
    pub fn format(&self, cx: &Cx) -> String {
        match self {
            TypeError::UnknownVar(sym) => format!("Unknown var: {}", cx.fetch(*sym)),
            TypeError::ReturnMismatch {
                fname,
                got,
                expected,
            } => format!(
                "Function {} returns {} but should return {}",
                cx.fetch(*fname),
                cx.fetch_type(*got).get_name(cx),
                cx.fetch_type(*expected).get_name(cx),
            ),
            TypeError::BopTypeMismatch {
                bop,
                got1,
                got2,
                expected,
            } => format!(
                "Invalid types for BOp {:?}: expected {}, got {} + {}",
                bop,
                cx.fetch_type(*expected).get_name(cx),
                cx.fetch_type(*got1).get_name(cx),
                cx.fetch_type(*got2).get_name(cx)
            ),
            TypeError::UopTypeMismatch { op, got, expected } => format!(
                "Invalid types for UOp {:?}: expected {}, got {}",
                op,
                cx.fetch_type(*expected).get_name(cx),
                cx.fetch_type(*got).get_name(cx)
            ),
            TypeError::LetTypeMismatch {
                name,
                got,
                expected,
            } => format!(
                "initializer for variable {}: expected {}, got {}",
                cx.fetch(*name),
                cx.fetch_type(*expected).get_name(cx),
                cx.fetch_type(*got).get_name(cx)
            ),
            TypeError::IfTypeMismatch { ifpart, elsepart } => format!(
                "If block return type is {}, but else block returns {}",
                cx.fetch_type(*ifpart).get_name(cx),
                cx.fetch_type(*elsepart).get_name(cx),
            ),
            TypeError::CondMismatch { got } => format!(
                "If expr condition is {}, not bool",
                cx.fetch_type(*got).get_name(cx),
            ),
            TypeError::ParamMismatch { got, expected } => format!(
                "Function wanted type {} in param but got type {}",
                cx.fetch_type(*got).get_name(cx),
                cx.fetch_type(*expected).get_name(cx)
            ),
            TypeError::CallMismatch { got } => format!(
                "Tried to call function but it is not a function, it is a {}",
                cx.fetch_type(*got).get_name(cx)
            ),
            TypeError::TupleRefMismatch { got } => format!(
                "Tried to reference tuple but didn't get a tuple, got {}",
                cx.fetch_type(*got).get_name(cx)
            ),
        }
    }
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

    /// TODO:
    /// To think about: Take a lambda and call it with the new scope,
    /// so we can never forget to pop it?
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
    fn add_var(&mut self, name: VarSym, typedef: TypeSym) {
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
    fn get_var(&self, name: VarSym) -> Result<TypeSym, TypeError> {
        for scope in self.syms.iter().rev() {
            if let Some(binding) = scope.get(&name) {
                return Ok(binding.typename.clone());
            }
        }
        Err(TypeError::UnknownVar(name))
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
/// Currently it's just, if the symbols match, the types match.
/// The symbols matching by definition means the structures match.
fn type_matches(t1: TypeSym, t2: TypeSym) -> bool {
    t1 == t2
}

pub fn typecheck(cx: &mut Cx, ir: ir::Ir<()>) -> Result<ir::Ir<TypeSym>, TypeError> {
    let symtbl = &mut Symtbl::new();
    ir.decls.iter().for_each(|d| predeclare_decl(cx, symtbl, d));
    let checked_decls = ir
        .decls
        .into_iter()
        .map(|decl| typecheck_decl(cx, symtbl, decl))
        .collect::<Result<Vec<ir::Decl<TypeSym>>, TypeError>>()?;
    Ok(ir::Ir {
        decls: checked_decls,
    })
}

/// Scan through all decl's and add any bindings to the symbol table,
/// so we don't need to do anything with forward references.
fn predeclare_decl(cx: &mut Cx, symtbl: &mut Symtbl, decl: &ir::Decl<()>) {
    match decl {
        ir::Decl::Function {
            name, signature, ..
        } => {
            // Add function to global scope
            let type_params = signature.params.iter().map(|(_name, t)| *t).collect();
            let function_type = cx.intern_type(&TypeDef::Lambda(type_params, signature.rettype));
            symtbl.add_var(*name, function_type);
        }
        ir::Decl::Const { name, typename, .. } => {
            symtbl.add_var(*name, *typename);
        }
    }
}

/// Typechecks a single decl
fn typecheck_decl(
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
            // TODO: How to handle return statements, hm?
            for (pname, ptype) in signature.params.iter() {
                symtbl.add_var(*pname, *ptype);
            }

            // This is squirrelly; basically, we want to return unit
            // if the function has no body, otherwise return the
            // type of the last expression.
            //
            // Oh gods, what in the name of Eris do we do if there's
            // a return statement here?
            // Use a Never type, it seems.
            let typechecked_exprs = typecheck_exprs(cx, symtbl, body)?;
            // TODO: This only works if there are no return statements.
            let last_expr_type = last_type_of(cx, &typechecked_exprs);

            if !type_matches(signature.rettype, last_expr_type) {
                return Err(TypeError::ReturnMismatch {
                    fname: name,
                    got: last_expr_type,
                    expected: signature.rettype,
                });
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
        } => Ok(ir::Decl::Const {
            name,
            typename,
            init: typecheck_expr(cx, symtbl, init)?,
        }),
    }
}

/// Typecheck a vec of expr's and returns them, with type annotations
/// attached.
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

/// Takes a slice of typed expr's and returns the type of the last one.
/// Returns unit if the slice is empty.
fn last_type_of(cx: &Cx, exprs: &[ir::TypedExpr<TypeSym>]) -> TypeSym {
    exprs.last().map(|e| e.t).unwrap_or_else(|| cx.unit())
}

/// Actually typecheck a single expr
fn typecheck_expr(
    cx: &mut Cx,
    symtbl: &mut Symtbl,
    expr: ir::TypedExpr<()>,
) -> Result<ir::TypedExpr<TypeSym>, TypeError> {
    use ir::Expr::*;
    let unittype = cx.unit();
    let booltype = cx.bool();
    match expr.e {
        Lit { val } => {
            let t = typecheck_literal(cx, &val)?;
            Ok(ir::TypedExpr { e: Lit { val }, t })
        }

        Var { name } => {
            let t = symtbl.get_var(name)?;
            Ok(expr.map(t))
        }
        BinOp { op, lhs, rhs } => {
            let lhs = Box::new(typecheck_expr(cx, symtbl, *lhs)?);
            let rhs = Box::new(typecheck_expr(cx, symtbl, *rhs)?);
            // Currently, our only valid binops are on numbers.
            let input_type = op.input_type(cx);
            let output_type = op.output_type(cx);
            if type_matches(lhs.t, rhs.t) && type_matches(input_type, lhs.t) {
                Ok(ir::TypedExpr {
                    e: BinOp { op, lhs, rhs },
                    t: output_type,
                })
            } else {
                Err(TypeError::BopTypeMismatch {
                    bop: op,
                    expected: input_type,
                    got1: lhs.t,
                    got2: rhs.t,
                })
            }
        }
        UniOp { op, rhs } => {
            let rhs = Box::new(typecheck_expr(cx, symtbl, *rhs)?);
            // Currently, our only valid binops are on numbers.
            let input_type = op.input_type(cx);
            let output_type = op.output_type(cx);
            if type_matches(input_type, rhs.t) {
                Ok(ir::TypedExpr {
                    e: UniOp { op, rhs },
                    t: output_type,
                })
            } else {
                Err(TypeError::UopTypeMismatch {
                    op,
                    expected: input_type,
                    got: rhs.t,
                })
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
            if type_matches(init_type, typename) {
                // Add var to symbol table, proceed
                symtbl.add_var(varname, typename);
                Ok(ir::TypedExpr {
                    e: Let {
                        varname,
                        typename,
                        init: Box::new(init_expr),
                    },
                    t: unittype,
                })
            } else {
                Err(TypeError::LetTypeMismatch {
                    name: varname,
                    got: init_type,
                    expected: typename,
                })
            }
        }
        If { cases, falseblock } => {
            let falseblock = typecheck_exprs(cx, symtbl, falseblock)?;
            let assumed_type = last_type_of(cx, &falseblock);
            let mut new_cases = vec![];
            for (cond, body) in cases {
                let cond_expr = typecheck_expr(cx, symtbl, cond)?;
                if type_matches(cond_expr.t, booltype) {
                    // Proceed to typecheck arms
                    let ifbody_exprs = typecheck_exprs(cx, symtbl, body)?;
                    let if_type = last_type_of(cx, &ifbody_exprs);
                    if !type_matches(if_type, assumed_type) {
                        return Err(TypeError::IfTypeMismatch {
                            ifpart: if_type,
                            elsepart: assumed_type,
                        });
                    }

                    // Great, it matches
                    new_cases.push((cond_expr, ifbody_exprs));
                } else {
                    return Err(TypeError::CondMismatch { got: cond_expr.t });
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
                symtbl.add_var(*paramname, *paramtype);
            }
            let body_expr = typecheck_exprs(cx, symtbl, body)?;
            let bodytype = last_type_of(cx, &body_expr);
            // TODO: validate/unify types
            if !type_matches(bodytype, signature.rettype) {
                let function_name = cx.intern("lambda");
                return Err(TypeError::ReturnMismatch {
                    fname: function_name,
                    got: bodytype,
                    expected: signature.rettype,
                });
            }
            symtbl.pop_scope();
            let lambdatype = signature.to_type(cx);
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
                        if !type_matches(given.t, *wanted) {
                            return Err(TypeError::ParamMismatch {
                                got: given.t,
                                expected: *wanted,
                            });
                        }
                    }
                    Ok(ir::TypedExpr {
                        e: Funcall {
                            func: Box::new(f),
                            params: given_params,
                        },
                        t: *rettype,
                    })
                }
                _other => Err(TypeError::CallMismatch { got: f.t }),
            }
        }
        Break => Ok(expr.map(unittype)),
        Return { .. } => todo!("TODO: Never type"),
        TupleCtor { body } => {
            let body_exprs = typecheck_exprs(cx, symtbl, body)?;
            let body_typesyms = body_exprs.iter().map(|te| te.t).collect();
            let body_type = TypeDef::Tuple(body_typesyms);
            Ok(ir::TypedExpr {
                t: cx.intern_type(&body_type),
                e: TupleCtor { body: body_exprs },
            })
        }
        TupleRef { expr, elt } => {
            let body_expr = typecheck_expr(cx, symtbl, *expr)?;
            let expr_typedef = cx.fetch_type(body_expr.t);
            if let TypeDef::Tuple(typesyms) = &*expr_typedef {
                // TODO
                assert!(elt < typesyms.len());
                Ok(ir::TypedExpr {
                    t: typesyms[elt],
                    e: TupleRef {
                        expr: Box::new(body_expr),
                        elt: elt,
                    },
                })
            } else {
                Err(TypeError::TupleRefMismatch { got: body_expr.t })
            }
        }
    }
}

fn typecheck_literal(cx: &mut Cx, lit: &ir::Literal) -> Result<TypeSym, TypeError> {
    match lit {
        ir::Literal::Integer(_) => Ok(cx.i32()),
        ir::Literal::Bool(_) => Ok(cx.bool()),
    }
}

//fn typecheck_tuple_ctr(cx: &mut Cx, body: &[ir::Expr

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;

    /// Sanity check
    #[test]
    fn test_typecheck_lit() {
        use ir;
        let cx = &mut crate::Cx::new();
        let t_i32 = cx.i32();
        let t_bool = cx.bool();

        assert_eq!(
            typecheck_literal(cx, &ir::Literal::Integer(9)).unwrap(),
            t_i32
        );
        assert_eq!(
            typecheck_literal(cx, &ir::Literal::Bool(false)).unwrap(),
            t_bool
        );
    }

    /// Test symbol table
    #[test]
    fn test_symtbl() {
        let cx = &mut crate::Cx::new();
        let t_foo = cx.intern("foo");
        let t_bar = cx.intern("bar");
        let t_i32 = cx.i32();
        let t_bool = cx.bool();
        let mut t = Symtbl::new();

        // Make sure we can get a value
        assert!(t.get_var(t_foo).is_err());
        assert!(t.get_var(t_bar).is_err());
        t.add_var(t_foo, t_i32);
        assert_eq!(t.get_var(t_foo).unwrap(), t_i32);
        assert!(t.get_var(t_bar).is_err());

        // Push scope, verify behavior is the same.
        t.push_scope();
        assert_eq!(t.get_var(t_foo).unwrap(), t_i32);
        assert!(t.get_var(t_bar).is_err());
        // Add var, make sure we can get it
        t.add_var(t_bar, t_bool);
        assert_eq!(t.get_var(t_foo).unwrap(), t_i32);
        assert_eq!(t.get_var(t_bar).unwrap(), t_bool);

        // Pop scope, make sure bar is gone.
        t.pop_scope();
        assert_eq!(t.get_var(t_foo).unwrap(), t_i32);
        assert!(t.get_var(t_bar).is_err());
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
        let t_i32 = cx.i32();
        let t_bool = cx.bool();

        let l1 = ir::Literal::Integer(3);
        let l1t = typecheck_literal(cx, &l1).unwrap();
        let l2 = ir::Literal::Bool(true);
        let l2t = typecheck_literal(cx, &l2).unwrap();
        assert!(!type_matches(l1t, l2t));

        assert!(type_matches(l1t, t_i32));
        assert!(type_matches(l2t, t_bool));
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
            assert!(type_matches(typecheck_expr(cx, tbl, ir).unwrap().t, t_i32));

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
            assert!(type_matches(typecheck_expr(cx, tbl, ir).unwrap().t, t_i32));

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
        let t_i32 = cx.i32();
        let t_unit = cx.unit();
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
            assert!(type_matches(typecheck_expr(cx, tbl, ir).unwrap().t, t_unit));
            // Is the variable now bound in our symbol table?
            assert_eq!(tbl.get_var(fooname).unwrap(), t_i32);
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
        let ftype = cx.intern_type(&TypeDef::Lambda(vec![t_i32, t_i32], t_i32));

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
            assert!(type_matches(last_type_of(cx, exprs), t_i32));
            // Is the variable now bound in our symbol table?
            assert_eq!(tbl.get_var(fname).unwrap(), ftype);
        }

        {
            use crate::parser::Parser;
            let src = "fn foo(): fn(I32):I32 = fn(x: I32):I32 = x+1 end end";
            let ast = Parser::new(cx, src).parse();
            let ir = ir::lower(&ast);
            let _ = &typecheck(cx, ir).unwrap();
        }
    }

    #[test]
    fn test_bogus_function() {
        let cx = &mut crate::Cx::new();
        use crate::parser::Parser;
        let src = "fn foo(): fn(I32):I32 = fn(x: I32):Bool = x+1 end end";
        let ast = Parser::new(cx, src).parse();
        let ir = ir::lower(&ast);
        assert!(typecheck(cx, ir).is_err());
    }

    /// TODO
    #[test]
    fn test_return() {}

    #[test]
    fn test_tuples() {
        use ir::*;
        let cx = &mut crate::Cx::new();
        let tbl = &mut Symtbl::new();
        assert_eq!(
            typecheck_expr(cx, tbl, *plz(ir::Expr::unit())).unwrap().t,
            cx.unit()
        );
    }
}
