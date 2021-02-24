//! Typechecking and other semantic checking.
//! Operates on the IR.

use crate::hir;
use crate::scope;
use crate::{Cx, TypeDef, TypeSym, VarSym};

/// A random other error type bundled up with a `Cx`,
/// so that error can be printed with info from the Cx.
#[derive(Debug, Clone)]
pub struct CxError<T>(Cx, T);

impl std::fmt::Display for CxError<TypeError> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.1.format(&self.0))
    }
}

impl std::error::Error for CxError<TypeError> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.1)
    }
}

impl From<CxError<TypeError>> for TypeError {
    fn from(e: CxError<TypeError>) -> TypeError {
        e.1
    }
}

impl std::error::Error for TypeError {}

impl std::fmt::Display for TypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Debug, Clone)]
pub enum TypeError {
    UnknownVar(VarSym),
    ReturnMismatch {
        fname: VarSym,
        got: TypeSym,
        expected: TypeSym,
    },
    BopTypeMismatch {
        bop: hir::BOp,
        got1: TypeSym,
        got2: TypeSym,
        expected: TypeSym,
    },
    UopTypeMismatch {
        op: hir::UOp,
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
    TypeMismatch {
        expr_name: String,
        got: TypeSym,
        expected: TypeSym,
    },
    MutabilityMismatch {
        expr_name: String,
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
            TypeError::TypeMismatch {
                expr_name,
                expected,
                got,
            } => format!(
                "Type mismatch in '{}' expresssion, expected {} but got {}",
                expr_name,
                cx.fetch_type(*expected).get_name(cx),
                cx.fetch_type(*got).get_name(cx)
            ),
            TypeError::MutabilityMismatch { expr_name } => {
                format!("Mutability mismatch in '{}' expresssion", expr_name)
            }
        }
    }
}

/// A variable binding
#[derive(Debug, Clone)]
pub struct VarBinding {
    name: VarSym,
    typename: TypeSym,
    mutable: bool,
}

/*
/// Symbol table.  Stores the scope stack and variable bindings.
/// Kinda dumb structure, but simple, and using a HashMap is trickier
/// 'cause we allow shadowing bindings.
///
pub struct Symtbl {
    syms: scope::Symbols<VarSym, VarBinding>,
}

*/
type Symtbl = scope::Symbols<VarSym, VarBinding>;

impl Symtbl {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a variable to the top level of the scope.
    /// Allows shadowing.
    fn add_var(&mut self, name: VarSym, typedef: TypeSym, mutable: bool) {
        let binding = VarBinding {
            name,
            typename: typedef.clone(),
            mutable,
        };
        self.add(name, binding);
    }

    /// Get the type of the given variable, or an error
    fn get_var(&self, name: VarSym) -> Result<TypeSym, TypeError> {
        Ok(self.get_binding(name)?.typename.clone())
    }

    /// Get the binding of the given variable, or an error
    fn get_binding(&self, name: VarSym) -> Result<&VarBinding, TypeError> {
        if let Some(binding) = self.get(name) {
            return Ok(binding);
        }
        Err(TypeError::UnknownVar(name))
    }
}

/// The interned value of a type inference ref
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct InfoSym(pub usize);

impl From<usize> for InfoSym {
    fn from(i: usize) -> InfoSym {
        InfoSym(i)
    }
}

impl From<InfoSym> for usize {
    fn from(i: InfoSym) -> usize {
        i.0
    }
}

/*
/// Info tag for type inference
#[derive(Debug, Hash, Clone, PartialEq, Eq)]
pub enum TypeInfo {
    /// Unknown type not inferred yet
    Unknown,
    /// Reference saying "this type is the same as that one",
    /// which may still be unknown.
    /// TODO: Symbol type needs to change.
    Ref(InfoSym),
    /// Known type.
    Known(TypeSym),
}

struct InferenceCx {
    types: intern::Interner<InfoSym, TypeInfo>,
}

impl InferenceCx {
    pub fn new() -> Self {
        let s = Self {
            types: intern::Interner::new(),
        };
        s
    }

    /// Intern the symbol.
    pub fn intern(&self, ti: &TypeInfo) -> InfoSym {
        self.types.intern(&ti.clone())
    }

    /// Get the string for a variable symbol
    pub fn fetch(&self, s: InfoSym) -> Rc<TypeInfo> {
        self.types.fetch(s)
    }

    /// Make the types of two terms equivalent, or produce an error if they're in conflict
    /// TODO: Figure out how to use this
    /// Aha, can we just make this another pass right before typechecking, fill out the IR tree
    /// with InfoSym rather than TypeSym or such, and then go from there?
    /// That might work!
    /// https://play.rust-lang.org/?version=stable&mode=debug&edition=2018&gist=ba1adacd8659de0587af68cb0e55a471
    pub fn unify(&self, a: &TypeInfo, b: &TypeInfo) -> Result<(), TypeError> {
        use TypeInfo::*;
        match (a, b) {
            // Follow references
            (Ref(a), _) => self.unify(&*self.fetch(*a), b),
            (_, Ref(b)) => self.unify(a, &*self.fetch(*b)),

            // When we don't know about a type, assume they match and
            // make the one know nothing about refer to the one we may
            // know something about
            (Unknown, _) => todo!(),
            (_, Unknown) => todo!(),

            (Known(t1), Known(t2)) if t1 == t2 => Ok(()),
            (Known(_t1), Known(_t2)) => todo!("Inference failed: Type mismatch"),

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
            (t1, t2) => todo!("Inference failed: Could not unify types"),
        }
    }

    /// Attempt to reconstruct a concrete type from a symbol.  This may
    /// fail if we don't have enough info to figure out what the type is.
    pub fn reconstruct(&self, sym: InfoSym) -> Result<TypeSym, TypeError> {
        let t = &*self.fetch(sym);
        use TypeInfo::*;
        match t {
            Unknown => todo!("No type?"),
            Ref(id) => self.reconstruct(*id),
            Known(sym) => Ok(*sym),
        }
    }
}
*/

/// Does t1 equal t2?
///
/// Currently we have no covariance or contravariance, so this is pretty simple.
/// Currently it's just, if the symbols match, the types match.
/// The symbols matching by definition means the structures match.
fn type_matches(cx: &Cx, t1: TypeSym, t2: TypeSym) -> bool {
    let td1 = &*cx.fetch_type(t1);
    let td2 = &*cx.fetch_type(t2);
    match (td1, td2) {
        // Numbers match numbers of unknown sizes...
        // Actually, maybe we need to do this with real type inference.
        /*
        (TypeDef::UnknownInt, TypeDef::SInt(_)) => true,
        (TypeDef::SInt(_), TypeDef::UnknownInt) => true,
        */
        (t1, t2) => t1 == t2,
    }
}

/// Try to actually typecheck the given HIR, and return HIR with resolved types.
pub fn typecheck(cx: &Cx, ir: hir::Ir<()>) -> Result<hir::Ir<TypeSym>, CxError<TypeError>> {
    let symtbl = &mut Symtbl::new();
    ir.decls.iter().for_each(|d| predeclare_decl(cx, symtbl, d));
    let checked_decls = ir
        .decls
        .into_iter()
        .map(|decl| typecheck_decl(cx, symtbl, decl))
        .collect::<Result<Vec<hir::Decl<TypeSym>>, CxError<TypeError>>>()?;
    Ok(hir::Ir {
        decls: checked_decls,
    })
}

/// Scan through all decl's and add any bindings to the symbol table,
/// so we don't need to do anything with forward references.
fn predeclare_decl(cx: &Cx, symtbl: &mut Symtbl, decl: &hir::Decl<()>) {
    match decl {
        hir::Decl::Function {
            name, signature, ..
        } => {
            // Add function to global scope
            let type_params = signature.params.iter().map(|(_name, t)| *t).collect();
            let function_type = cx.intern_type(&TypeDef::Lambda(type_params, signature.rettype));
            symtbl.add_var(*name, function_type, false);
        }
        hir::Decl::Const { name, typename, .. } => {
            symtbl.add_var(*name, *typename, false);
        }
    }
}

/// Typechecks a single decl
fn typecheck_decl(
    cx: &Cx,
    symtbl: &mut Symtbl,
    decl: hir::Decl<()>,
) -> Result<hir::Decl<TypeSym>, CxError<TypeError>> {
    match decl {
        hir::Decl::Function {
            name,
            signature,
            body,
        } => {
            // Push scope, typecheck and add params to symbol table
            symtbl.push_scope();
            // TODO: How to handle return statements, hm?
            for (pname, ptype) in signature.params.iter() {
                symtbl.add_var(*pname, *ptype, false);
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

            if !type_matches(cx, signature.rettype, last_expr_type) {
                return Err(CxError(
                    cx.clone(),
                    TypeError::ReturnMismatch {
                        fname: name,
                        got: last_expr_type,
                        expected: signature.rettype,
                    },
                ));
            }

            symtbl.pop_scope();
            Ok(hir::Decl::Function {
                name,
                signature,
                body: typechecked_exprs,
            })
        }
        hir::Decl::Const {
            name,
            typename,
            init,
        } => Ok(hir::Decl::Const {
            name,
            typename,
            init: typecheck_expr(cx, symtbl, init)?,
        }),
    }
}

/// Typecheck a vec of expr's and returns them, with type annotations
/// attached.
fn typecheck_exprs(
    cx: &Cx,
    symtbl: &mut Symtbl,
    exprs: Vec<hir::TypedExpr<()>>,
) -> Result<Vec<hir::TypedExpr<TypeSym>>, CxError<TypeError>> {
    exprs
        .into_iter()
        .map(|e| typecheck_expr(cx, symtbl, e))
        .collect()
}

/// Takes a slice of typed expr's and returns the type of the last one.
/// Returns unit if the slice is empty.
fn last_type_of(cx: &Cx, exprs: &[hir::TypedExpr<TypeSym>]) -> TypeSym {
    exprs.last().map(|e| e.t).unwrap_or_else(|| cx.unit())
}

/// Actually typecheck a single expr
fn typecheck_expr(
    cx: &Cx,
    symtbl: &mut Symtbl,
    expr: hir::TypedExpr<()>,
    //) -> Result<hir::TypedExpr<TypeSym>, TypeError> {
) -> Result<hir::TypedExpr<TypeSym>, CxError<TypeError>> {
    use hir::Expr::*;
    let unittype = cx.unit();
    let booltype = cx.bool();
    let rar = |e| CxError(cx.clone(), e);
    //let rar = |e| CxError(Cx::new(), e);
    match expr.e {
        Lit { val } => {
            let t = typecheck_literal(cx, &val).map_err(rar)?;
            Ok(hir::TypedExpr { e: Lit { val }, t })
        }

        Var { name } => {
            let t = symtbl.get_var(name).map_err(rar)?;
            Ok(expr.map(t))
        }
        BinOp { op, lhs, rhs } => {
            let lhs = Box::new(typecheck_expr(cx, symtbl, *lhs)?);
            let rhs = Box::new(typecheck_expr(cx, symtbl, *rhs)?);
            // Currently, our only valid binops are on numbers.
            let input_type = op.input_type(cx);
            let output_type = op.output_type(cx);
            if type_matches(cx, lhs.t, rhs.t) && type_matches(cx, input_type, lhs.t) {
                Ok(hir::TypedExpr {
                    e: BinOp { op, lhs, rhs },
                    t: output_type,
                })
            } else {
                Err(rar(TypeError::BopTypeMismatch {
                    bop: op,
                    expected: input_type,
                    got1: lhs.t,
                    got2: rhs.t,
                }))
            }
        }
        UniOp { op, rhs } => {
            let rhs = Box::new(typecheck_expr(cx, symtbl, *rhs)?);
            // Currently, our only valid binops are on numbers.
            let input_type = op.input_type(cx);
            let output_type = op.output_type(cx);
            if type_matches(cx, input_type, rhs.t) {
                Ok(hir::TypedExpr {
                    e: UniOp { op, rhs },
                    t: output_type,
                })
            } else {
                Err(rar(TypeError::UopTypeMismatch {
                    op,
                    expected: input_type,
                    got: rhs.t,
                }))
            }
        }
        Block { body } => {
            let b = typecheck_exprs(cx, symtbl, body)?;
            let t = last_type_of(cx, &b);
            Ok(hir::TypedExpr {
                e: Block { body: b },
                t,
            })
        }
        Let {
            varname,
            typename,
            init,
            mutable,
        } => {
            let init_expr = typecheck_expr(cx, symtbl, *init)?;
            let init_type = init_expr.t;
            if type_matches(cx, init_type, typename) {
                // Add var to symbol table, proceed
                symtbl.add_var(varname, typename, mutable);
                Ok(hir::TypedExpr {
                    e: Let {
                        varname,
                        typename,
                        init: Box::new(init_expr),
                        mutable,
                    },
                    t: unittype,
                })
            } else {
                Err(rar(TypeError::LetTypeMismatch {
                    name: varname,
                    got: init_type,
                    expected: typename,
                }))
            }
        }
        If { cases, falseblock } => {
            let falseblock = typecheck_exprs(cx, symtbl, falseblock)?;
            let assumed_type = last_type_of(cx, &falseblock);
            let mut new_cases = vec![];
            for (cond, body) in cases {
                let cond_expr = typecheck_expr(cx, symtbl, cond)?;
                if type_matches(cx, cond_expr.t, booltype) {
                    // Proceed to typecheck arms
                    let ifbody_exprs = typecheck_exprs(cx, symtbl, body)?;
                    let if_type = last_type_of(cx, &ifbody_exprs);
                    if !type_matches(cx, if_type, assumed_type) {
                        return Err(rar(TypeError::IfTypeMismatch {
                            ifpart: if_type,
                            elsepart: assumed_type,
                        }));
                    }

                    // Great, it matches
                    new_cases.push((cond_expr, ifbody_exprs));
                } else {
                    return Err(rar(TypeError::CondMismatch { got: cond_expr.t }));
                }
            }
            Ok(hir::TypedExpr {
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
            Ok(hir::TypedExpr {
                e: Loop { body: b },
                t,
            })
        }
        Lambda { signature, body } => {
            symtbl.push_scope();
            // add params to symbol table
            for (paramname, paramtype) in signature.params.iter() {
                symtbl.add_var(*paramname, *paramtype, false);
            }
            let body_expr = typecheck_exprs(cx, symtbl, body)?;
            let bodytype = last_type_of(cx, &body_expr);
            // TODO: validate/unify types
            if !type_matches(cx, bodytype, signature.rettype) {
                let function_name = cx.intern("lambda");
                return Err(rar(TypeError::ReturnMismatch {
                    fname: function_name,
                    got: bodytype,
                    expected: signature.rettype,
                }));
            }
            symtbl.pop_scope();
            let lambdatype = signature.to_type(cx);
            Ok(hir::TypedExpr {
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
                        if !type_matches(cx, given.t, *wanted) {
                            return Err(rar(TypeError::ParamMismatch {
                                got: given.t,
                                expected: *wanted,
                            }));
                        }
                    }
                    Ok(hir::TypedExpr {
                        e: Funcall {
                            func: Box::new(f),
                            params: given_params,
                        },
                        t: *rettype,
                    })
                }
                _other => Err(rar(TypeError::CallMismatch { got: f.t })),
            }
        }
        Break => Ok(expr.map(unittype)),
        Return { .. } => todo!("TODO: Never type"),
        TupleCtor { body } => {
            let body_exprs = typecheck_exprs(cx, symtbl, body)?;
            let body_typesyms = body_exprs.iter().map(|te| te.t).collect();
            let body_type = TypeDef::Tuple(body_typesyms);
            Ok(hir::TypedExpr {
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
                Ok(hir::TypedExpr {
                    t: typesyms[elt],
                    e: TupleRef {
                        expr: Box::new(body_expr),
                        elt,
                    },
                })
            } else {
                Err(rar(TypeError::TupleRefMismatch { got: body_expr.t }))
            }
        }
        Assign { lhs, rhs } => {
            let lhs_expr = typecheck_expr(cx, symtbl, *lhs)?;
            let rhs_expr = typecheck_expr(cx, symtbl, *rhs)?;
            // So the rules for assignments are, it's good only:
            // If the lhs is an lvalue (just variable or tuple ref currently)
            // (This basically comes down to "somewhere with a location"...)
            // If the lhs is mutable
            // If the types of lhs and rhs match.
            if !is_mutable_lvalue(symtbl, &lhs_expr.e).map_err(rar)? {
                Err(rar(TypeError::MutabilityMismatch {
                    expr_name: String::from("assignment"),
                }))
            } else if !type_matches(cx, lhs_expr.t, rhs_expr.t) {
                Err(rar(TypeError::TypeMismatch {
                    expr_name: String::from("assignment"),
                    expected: lhs_expr.t,
                    got: rhs_expr.t,
                }))
            } else {
                Ok(hir::TypedExpr {
                    t: cx.unit(),
                    e: Assign {
                        lhs: Box::new(lhs_expr),
                        rhs: Box::new(rhs_expr),
                    },
                })
            }
        }
        Deref { .. } => todo!(),
        Ref { .. } => todo!(),
    }
}

/// So, what is an lvalue?
/// Well, it's a variable,
/// or it's an lvalue in a deref expr or tupleref
fn is_mutable_lvalue<T>(symtbl: &Symtbl, expr: &hir::Expr<T>) -> Result<bool, TypeError> {
    match expr {
        hir::Expr::Var { name } => {
            let v = symtbl.get_binding(*name)?;
            Ok(v.mutable)
        }
        hir::Expr::TupleRef { expr, .. } => is_mutable_lvalue(symtbl, &expr.e),
        _ => Ok(false),
    }
}

fn typecheck_literal(cx: &Cx, lit: &hir::Literal) -> Result<TypeSym, TypeError> {
    match lit {
        hir::Literal::Integer(_) => Ok(cx.i32()),
        hir::Literal::Bool(_) => Ok(cx.bool()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;

    /// Sanity check
    #[test]
    fn test_typecheck_lit() {
        use hir;
        let cx = &mut crate::Cx::new();
        let t_i32 = cx.i32();
        let t_bool = cx.bool();

        assert!(type_matches(
            cx,
            typecheck_literal(cx, &hir::Literal::Integer(9)).unwrap(),
            t_i32
        ));
        assert_eq!(
            typecheck_literal(cx, &hir::Literal::Bool(false)).unwrap(),
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
        t.add_var(t_foo, t_i32, false);
        assert_eq!(t.get_var(t_foo).unwrap(), t_i32);
        assert!(t.get_var(t_bar).is_err());

        // Push scope, verify behavior is the same.
        t.push_scope();
        assert_eq!(t.get_var(t_foo).unwrap(), t_i32);
        assert!(t.get_var(t_bar).is_err());
        // Add var, make sure we can get it
        t.add_var(t_bar, t_bool, false);
        assert_eq!(t.get_var(t_foo).unwrap(), t_i32);
        assert_eq!(t.get_var(t_bar).unwrap(), t_bool);

        // Pop scope, make sure bar is gone.
        t.pop_scope();
        assert_eq!(t.get_var(t_foo).unwrap(), t_i32);
        assert!(t.get_var(t_bar).is_err());

        // Make sure we can shadow a value
        t.add_var(t_foo, t_i32, false);
        t.add_var(t_foo, t_bool, false);
        assert_eq!(t.get_var(t_foo).unwrap(), t_bool);
        // TODO: Check to make sure the shadowed var is still there.
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

        let l1 = hir::Literal::Integer(3);
        let l1t = typecheck_literal(cx, &l1).unwrap();
        let l2 = hir::Literal::Bool(true);
        let l2t = typecheck_literal(cx, &l2).unwrap();
        assert!(!type_matches(cx, l1t, l2t));

        assert!(type_matches(cx, l1t, t_i32));
        assert!(type_matches(cx, l2t, t_bool));
    }

    /// Test binop typechecks
    #[test]
    fn test_binop() {
        let cx = &mut crate::Cx::new();
        let t_i32 = cx.intern_type(&TypeDef::SInt(4));
        let tbl = &mut Symtbl::new();

        use hir::*;
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
                cx,
                typecheck_expr(cx, tbl, ir).unwrap().t,
                t_i32
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

        use hir::*;
        {
            let ir = *plz(Expr::UniOp {
                op: UOp::Neg,
                rhs: plz(Expr::Lit {
                    val: Literal::Integer(4),
                }),
            });
            assert!(type_matches(
                cx,
                typecheck_expr(cx, tbl, ir).unwrap().t,
                t_i32
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
        let t_i32 = cx.i32();
        let t_unit = cx.unit();
        let fooname = cx.intern("foo");

        use hir::*;
        {
            let ir = *plz(Expr::Let {
                varname: fooname,
                typename: t_i32.clone(),
                init: plz(Expr::Lit {
                    val: Literal::Integer(42),
                }),
                mutable: false,
            });
            assert!(type_matches(
                cx,
                typecheck_expr(cx, tbl, ir).unwrap().t,
                t_unit
            ));
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

        use hir::*;
        {
            let ir = vec![
                *plz(Expr::Let {
                    varname: fname,
                    typename: ftype,
                    mutable: false,
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
            assert!(type_matches(cx, last_type_of(cx, exprs), t_i32));
            // Is the variable now bound in our symbol table?
            assert_eq!(tbl.get_var(fname).unwrap(), ftype);
        }

        {
            use crate::parser::Parser;
            let src = "fn foo(): fn(I32):I32 = fn(x: I32):I32 = x+1 end end";
            let ast = Parser::new(cx, src).parse();
            let ir = hir::lower(&ast);
            let _ = &typecheck(cx, ir).unwrap();
        }
    }

    #[test]
    #[should_panic]
    fn test_bogus_function() {
        let src = "fn foo(): fn(I32):I32 = fn(x: I32):Bool = x+1 end end";
        typecheck_src(src);
    }

    /// TODO
    #[test]
    fn test_return() {}

    #[test]
    fn test_tuples() {
        use hir::*;
        let cx = &mut crate::Cx::new();
        let tbl = &mut Symtbl::new();
        assert_eq!(
            typecheck_expr(cx, tbl, *plz(hir::Expr::unit())).unwrap().t,
            cx.unit()
        );
    }

    fn typecheck_src(src: &str) {
        let cx = &mut crate::Cx::new();
        use crate::parser::Parser;
        let ast = Parser::new(cx, src).parse();
        let ir = hir::lower(&ast);
        match &typecheck(cx, ir) {
            Ok(_res) => (),
            Err(e) => {
                panic!("Typecheck failed: {}", e)
            }
        }
    }

    #[test]
    fn test_assign() {
        let src = r#"fn foo() =
    let mut x: I32 = 10
    x = 11
end"#;
        typecheck_src(src);
    }

    #[test]
    fn test_assign2() {
        let src = r#"fn foo() =
    let mut x: {I32, I32} = {10, 12}
    x.0 = 11
end"#;
        typecheck_src(src);
    }

    #[test]
    fn test_number_types() {
        let src = r#"fn foo() =
        let x: I8 = 8
        let x: I16 = 9
        let x: I32 = 10
        let y: I64 = 11
        let y: I128 = 12
end"#;
        typecheck_src(src);
    }

    #[test]
    #[should_panic]
    fn test_bad_assign1() {
        let src = r#"fn foo() =
    let x: I32 = 10
    x = 11
end"#;
        typecheck_src(src);
    }

    #[test]
    #[should_panic]
    fn test_bad_assign2() {
        let src = r#"fn foo() =
    {1,2,3}.3 = 11
end"#;
        typecheck_src(src);
    }

    #[test]
    fn test_bad_types() {
        let src = r#"fn foo() =
        let x: I32 = 10
        let mut y: I64 = 11
        y = x
end"#;
        typecheck_src(src);
    }
}
