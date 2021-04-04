//! Typechecking and other semantic checking.
//! Operates on the IR.

use std::borrow::Cow;

use crate::hir;
use crate::scope;
use crate::*;
use crate::{TypeDef, TypeSym, VarSym, INT};

impl std::error::Error for TypeError {}

impl std::fmt::Display for TypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.format())
    }
}

#[derive(Debug, Clone)]
pub enum TypeError {
    UnknownVar(VarSym),
    InferenceFailed {
        t1: InfTypeDef,
        t2: InfTypeDef,
    },
    InvalidReturn,
    Return {
        fname: VarSym,
        got: TypeSym,
        expected: TypeSym,
    },
    BopType {
        bop: hir::BOp,
        got1: TypeSym,
        got2: TypeSym,
        expected: TypeSym,
    },
    UopType {
        op: hir::UOp,
        got: TypeSym,
        expected: TypeSym,
    },
    LetType {
        name: VarSym,
        got: TypeSym,
        expected: TypeSym,
    },
    IfType {
        ifpart: TypeSym,
        elsepart: TypeSym,
    },
    Cond {
        got: TypeSym,
    },
    Param {
        got: TypeSym,
        expected: TypeSym,
    },
    Call {
        got: TypeSym,
    },
    TupleRef {
        got: TypeSym,
    },
    TypeMismatch {
        expr_name: Cow<'static, str>,
        got: TypeSym,
        expected: TypeSym,
    },
    Mutability {
        expr_name: Cow<'static, str>,
    },
}

impl TypeError {
    pub fn format(&self) -> String {
        match self {
            TypeError::UnknownVar(sym) => format!("Unknown var: {}", INT.fetch(*sym)),
            TypeError::InvalidReturn => {
                format!("return expression happened somewhere that isn't in a function!")
            }
            TypeError::InferenceFailed { t1, t2 } => {
                format!("Type inference failed: {:?} != {:?}", t1, t2)
            }
            TypeError::Return {
                fname,
                got,
                expected,
            } => format!(
                "Function {} returns {} but should return {}",
                INT.fetch(*fname),
                INT.fetch_type(*got).get_name(),
                INT.fetch_type(*expected).get_name(),
            ),
            TypeError::BopType {
                bop,
                got1,
                got2,
                expected,
            } => format!(
                "Invalid types for BOp {:?}: expected {}, got {} + {}",
                bop,
                INT.fetch_type(*expected).get_name(),
                INT.fetch_type(*got1).get_name(),
                INT.fetch_type(*got2).get_name()
            ),
            TypeError::UopType { op, got, expected } => format!(
                "Invalid types for UOp {:?}: expected {}, got {}",
                op,
                INT.fetch_type(*expected).get_name(),
                INT.fetch_type(*got).get_name()
            ),
            TypeError::LetType {
                name,
                got,
                expected,
            } => format!(
                "initializer for variable {}: expected {}, got {}",
                INT.fetch(*name),
                INT.fetch_type(*expected).get_name(),
                INT.fetch_type(*got).get_name()
            ),
            TypeError::IfType { ifpart, elsepart } => format!(
                "If block return type is {}, but else block returns {}",
                INT.fetch_type(*ifpart).get_name(),
                INT.fetch_type(*elsepart).get_name(),
            ),
            TypeError::Cond { got } => format!(
                "If expr condition is {}, not bool",
                INT.fetch_type(*got).get_name(),
            ),
            TypeError::Param { got, expected } => format!(
                "Function wanted type {} in param but got type {}",
                INT.fetch_type(*got).get_name(),
                INT.fetch_type(*expected).get_name()
            ),
            TypeError::Call { got } => format!(
                "Tried to call function but it is not a function, it is a {}",
                INT.fetch_type(*got).get_name()
            ),
            TypeError::TupleRef { got } => format!(
                "Tried to reference tuple but didn't get a tuple, got {}",
                INT.fetch_type(*got).get_name()
            ),
            TypeError::TypeMismatch {
                expr_name,
                expected,
                got,
            } => format!(
                "Type mismatch in '{}' expresssion, expected {} but got {}",
                expr_name,
                INT.fetch_type(*expected).get_name(),
                INT.fetch_type(*got).get_name()
            ),
            TypeError::Mutability { expr_name } => {
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

/*
impl InferenceCx {
    /*
    /// Take an expression, and update the type symbols to
    /// represent what we can figure out about it.
    ///
    /// Basically walk down the tree and figure out all the facts we
    /// know for the inference algorithm to work with, and attach them to
    /// the appropriate expressions.
    fn heckin_infer(&mut self, expr: hir::TypedExpr<()>) -> hir::TypedExpr<InfTypeSym> {
        use hir::Expr as E;
        match expr.e {
            E::Lit { val } => {
                let t = self.insert(Self::infer_lit(&val));
                hir::TypedExpr {
                    e: E::Lit { val },
                    t: t,
                }
            }
            E::Var { name } => todo!(),
            E::BinOp { op, lhs, rhs } => {
                let t1 = self.heckin_infer(*lhs);
                let t2 = self.heckin_infer(*rhs);
                let output_type = op.output_inftype();
                // "t1 is the same as our input type"
                // "t2 is the same as our input type"
                // That occurs in unify() which is a separate step...
                // We may not have enough info to know what t1 and t2 are yet.
                //
                // "The type of this expression is its output type"
                let t = self.insert(TypeInfo::Known(output_type));
                hir::TypedExpr {
                    e: E::BinOp {
                        op: op,
                        lhs: Box::new(t1),
                        rhs: Box::new(t2),
                    },
                    t: t,
                }
            }
            E::UniOp { op, rhs } => {
                let t1 = self.heckin_infer(*rhs);
                let output_type = op.output_inftype();
                // "The type of this expression is its output type"
                let t = self.insert(TypeInfo::Known(output_type));
                hir::TypedExpr {
                    e: E::UniOp {
                        op: op,
                        rhs: Box::new(t1),
                    },
                    t: t,
                }
            }
            E::Block { body } => todo!(),
            E::Let {
                varname,
                typename,
                init,
                mutable,
            } => {
                let t1 = self.heckin_infer(*init);
                let t = if let Some(ty) = typename {
                    // > let x: T = foo
                    // x has the type T
                    // foo has the same type as x
                    //self.insert(TypeInfo::Known(ty))
                } else {
                    // > let x = foo
                    // x has the same type as foo
                    self.insert(TypeInfo::Ref(t1.t))
                };
                hir::TypedExpr {
                    e: E::Let {
                        varname,
                        typename,
                        init: Box::new(t1),
                        mutable,
                    },
                    t: t,
                }
            }
            E::If { cases, falseblock } => todo!(),
            E::Loop { body } => todo!(),
            E::Lambda { signature, body } => todo!(),
            E::Funcall { func, params } => todo!(),
            E::Break => todo!(),
            E::Return { retval } => todo!(),
            E::TupleCtor { body } => todo!(),
            E::TupleRef { expr, elt } => todo!(),
            E::Assign { lhs, rhs } => todo!(),
            E::Deref { expr } => todo!(),
            E::Ref { expr } => todo!(),
        }
    }

    fn infer_lit(lit: &hir::Literal) -> TypeInfo {
        match lit {
            // TODO: Make this an "unknown number" type.
            hir::Literal::Integer(_) => TypeInfo::Known(InfTypeDef::SInt(4)),
            hir::Literal::SizedInteger { vl: _, bytes } => {
                TypeInfo::Known(InfTypeDef::SInt(*bytes))
            }
            hir::Literal::Bool(_) => TypeInfo::Known(InfTypeDef::Bool),
        }
    }
    */

    /// Make the types of two terms equivalent, or produce an error if they're in conflict
    /// TODO: Figure out how to use this
    /// Aha, can we just make this another pass right before typechecking, fill out the IR tree
    /// with InfoSym rather than TypeSym or such, and then go from there?
    /// That might work!
    /// https://play.rust-lang.org/?version=stable&mode=debug&edition=2018&gist=ba1adacd8659de0587af68cb0e55a471
    pub fn unify(&mut self, a: InfTypeSym, b: InfTypeSym) -> Result<(), TypeError> {
        use TypeInfo::*;
        // TODO someday: clean up clones
        let def_a = self.get(a).clone();
        let def_b = self.get(b).clone();
        match (def_a, def_b) {
            // Follow references
            (Ref(a), _) => self.unify(a, b),
            (_, Ref(b)) => self.unify(a, b),

            // When we don't know about a type, assume they match and
            // make the one know nothing about refer to the one we may
            // know something about.
            //
            // This involves updating what we know about our types.
            (Unknown, _) => {
                self.types.insert(a, Ref(b));
                Ok(())
            }
            (_, Unknown) => {
                self.types.insert(b, Ref(a));
                Ok(())
            }

            (Known(t1), Known(t2)) => self.unify_defs(t1, t2),
        }
    }

    /// Unify concrete types.
    pub fn unify_defs(&mut self, a: InfTypeDef, b: InfTypeDef) -> Result<(), TypeError> {
        use InfTypeDef::*;
        match (a, b) {
            // Primitives are easy to unify
            (SInt(sa), SInt(sb)) if sa == sb => Ok(()),
            (SInt(_sa), SInt(_sb)) => todo!(),
            (Bool, Bool) => Ok(()),
            // Never is our bottom-ish type
            (Never, _) => Ok(()),

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
            (_t1, _t2) => todo!(),
        }
    }

    /// Attempt to reconstruct a concrete type from an inferred type symbol.  This may
    /// fail if we don't have enough info to figure out what the type is.
    pub fn reconstruct(&self, cx: &Cx, sym: InfTypeSym) -> Result<TypeSym, TypeError> {
        let t = &*self.get(sym);
        use InfTypeDef as I;
        use TypeDef as C;
        use TypeInfo::*;
        let def = match t {
            Unknown => todo!("No type?"),
            Ref(id) => return self.reconstruct(cx, *id),
            Known(I::SInt(i)) => C::SInt(*i),
            Known(I::Bool) => C::Bool,
            Known(I::Never) => C::Never,
            Known(I::Tuple(items)) => {
                let new_items: Result<Vec<TypeSym>, TypeError> = items
                    .into_iter()
                    .map(|i| self.reconstruct(cx, *i))
                    .collect();
                C::Tuple(new_items?)
            }
            Known(I::Lambda(args, ret)) => {
                let new_args: Result<Vec<TypeSym>, TypeError> =
                    args.into_iter().map(|i| self.reconstruct(cx, *i)).collect();
                let new_ret = self.reconstruct(cx, **ret)?;
                C::Lambda(new_args?, new_ret)
            }
        };
        Ok(cx.intern_type(&def))
    }
}
*/

/// Does t1 equal t2?
///
/// Currently we have no covariance or contravariance, so this is pretty simple.
/// Currently it's just, if the symbols match, the types match.
/// The symbols matching by definition means the structures match.
fn type_matches(wanted: TypeSym, got: TypeSym) -> bool {
    // If we want some type, and got Never, then this is always valid
    // because we never hit the expression that expected the `wanted` type
    if got == INT.never() {
        return true;
    } else {
        wanted == got
    }
}

/// Goes from types that may be unknown to a type that is
/// totally real, and returns the real type.
/// Returns `None` if there is not enough info to decide.
fn infer_type(t1: TypeSym, t2: TypeSym) -> Option<TypeSym> {
    let t1_def = &*INT.fetch_type(t1);
    let t2_def = &*INT.fetch_type(t2);
    match (t1_def, t2_def) {
        (TypeDef::UnknownInt, TypeDef::SInt(_)) => Some(t2),
        (TypeDef::SInt(_), TypeDef::UnknownInt) => Some(t1),
        // Anything related to a never type becomes a never type,
        // because by definition control flow never actually gets
        // to that expression.
        (TypeDef::Never, _) => Some(t1),
        (_, TypeDef::Never) => Some(t2),
        // Tuples!  If we can infer the types of their members, then life is good
        (TypeDef::Tuple(tup1), TypeDef::Tuple(tup2)) => {
            // If tuples are different lengths, they never match
            if tup1.len() != tup2.len() {
                return None;
            }
            // I'm sure there's a neat comprehension to do this, but I haven't
            // been sleeping well lately.
            let mut accm = vec![];
            for (item1, item2) in tup1.iter().zip(tup2) {
                if let Some(t) = infer_type(*item1, *item2) {
                    accm.push(t);
                } else {
                    // Some item in the tuples doesn't match
                    return None;
                }
            }
            let sym = INT.intern_type(&TypeDef::Tuple(accm));
            Some(sym)
        }
        // TODO: Never and Never?  Is that valid, or not?  I THINK it is
        (tt1, tt2) if tt1 == tt2 => Some(t1),
        _ => None,
    }
}

/// Walks down the expression tree and if it hits a type that is a `UnknownInt`
/// it replaces it with the given type.  Basically we use `infer_type()` on a subexpression
/// tree to figure out what the real number type of it must be, then
/// call this to rewrite the tree.
///
/// I THINK this can never fail, if a contradiction occur it gets caught in `infer_type()`
fn reify_types(
    from: TypeSym,
    to: TypeSym,
    expr: hir::TypedExpr<TypeSym>,
) -> hir::TypedExpr<TypeSym> {
    // We're done?
    if expr.t == to {
        return expr;
    } else {
        expr.map_type(&|t| if *t == from { to } else { *t })
    }
}

/// `reify_types()`, called on the last value in a list of expressions such
/// as a function body.
fn reify_last_types(
    from: TypeSym,
    to: TypeSym,
    exprs: Vec<hir::TypedExpr<TypeSym>>,
) -> Vec<hir::TypedExpr<TypeSym>> {
    assert!(exprs.len() > 0);
    // We own the vec, so we can do whatever we want to it!  Muahahahahaha!
    let mut exprs = exprs;
    let last_expr = exprs.pop().unwrap();
    exprs.push(reify_types(from, to, last_expr));
    exprs
}

/// Try to actually typecheck the given HIR, and return HIR with resolved types.
pub fn typecheck(ir: hir::Ir<()>) -> Result<hir::Ir<TypeSym>, TypeError> {
    let symtbl = &mut Symtbl::new();
    ir.decls.iter().for_each(|d| predeclare_decl(symtbl, d));
    let checked_decls = ir
        .decls
        .into_iter()
        .map(|decl| typecheck_decl(symtbl, decl))
        .collect::<Result<Vec<hir::Decl<TypeSym>>, TypeError>>()?;
    Ok(hir::Ir {
        decls: checked_decls,
    })
}

/// Scan through all decl's and add any bindings to the symbol table,
/// so we don't need to do anything with forward references.
fn predeclare_decl(symtbl: &mut Symtbl, decl: &hir::Decl<()>) {
    match decl {
        hir::Decl::Function {
            name, signature, ..
        } => {
            // Add function to global scope
            let type_params = signature.params.iter().map(|(_name, t)| *t).collect();
            let function_type = INT.intern_type(&TypeDef::Lambda(type_params, signature.rettype));
            symtbl.add_var(*name, function_type, false);
        }
        hir::Decl::Const { name, typename, .. } => {
            symtbl.add_var(*name, *typename, false);
        }
    }
}

/// Typechecks a single decl
fn typecheck_decl(
    symtbl: &mut Symtbl,
    decl: hir::Decl<()>,
) -> Result<hir::Decl<TypeSym>, TypeError> {
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
            let typechecked_exprs = typecheck_exprs(symtbl, body, Some(signature.rettype))?;
            // Ok, so we *also* need to walk through all the expressions
            // and look for any "return" exprs (or later `?`/`try` exprs
            // also) and see make sure the return types match.
            let last_expr_type = last_type_of(&typechecked_exprs);
            if let Some(t) = infer_type(last_expr_type, signature.rettype) {
                let inferred_exprs = reify_last_types(last_expr_type, t, typechecked_exprs);
                symtbl.pop_scope();
                Ok(hir::Decl::Function {
                    name,
                    signature,
                    body: inferred_exprs,
                })
            } else {
                //if !type_matches(signature.rettype, last_expr_type) {
                Err(TypeError::Return {
                    fname: name,
                    got: last_expr_type,
                    expected: signature.rettype,
                })
            }
        }
        hir::Decl::Const {
            name,
            typename,
            init,
        } => Ok(hir::Decl::Const {
            name,
            typename,
            init: typecheck_expr(symtbl, init, None)?,
        }),
    }
}

/// Typecheck a vec of expr's and returns them, with type annotations
/// attached.
fn typecheck_exprs(
    symtbl: &mut Symtbl,
    exprs: Vec<hir::TypedExpr<()>>,
    function_rettype: Option<TypeSym>,
) -> Result<Vec<hir::TypedExpr<TypeSym>>, TypeError> {
    // For each expr, figure out what its type *should* be
    exprs
        .into_iter()
        .map(|e| typecheck_expr(symtbl, e, function_rettype))
        .collect()
}

/// Takes a slice of typed expr's and returns the type of the last one.
/// Returns unit if the slice is empty.  Returns None if the only expressions
/// are return statements or other things that don't return a value
///
/// Basically, what do we do if we get this code?
///
/// let x = something
/// return something_else
/// x
///
/// Or even just:
///
/// return 1
/// return 2
/// return 3
///
/// The type of those expr lists is `Never`,
fn last_type_of(exprs: &[hir::TypedExpr<TypeSym>]) -> TypeSym {
    exprs.last().map(|e| e.t).unwrap_or_else(|| INT.unit())
}

/// Actually typecheck a single expr
///
/// `function_rettype` is the type that `return` exprs and such must be.
fn typecheck_expr(
    symtbl: &mut Symtbl,
    expr: hir::TypedExpr<()>,
    function_rettype: Option<TypeSym>,
) -> Result<hir::TypedExpr<TypeSym>, TypeError> {
    use hir::Expr::*;
    let unittype = INT.unit();
    let booltype = INT.bool();
    // TODO: Better name maybe!
    match expr.e {
        Lit { val } => {
            let t = typecheck_literal(&val)?;
            Ok(hir::TypedExpr { e: Lit { val }, t })
        }

        Var { name } => {
            let t = symtbl.get_var(name)?;
            Ok(expr.map_type(&|_| t))
        }
        BinOp { op, lhs, rhs } => {
            // Typecheck each arm
            let lhs1 = typecheck_expr(symtbl, *lhs, function_rettype)?;
            let rhs1 = typecheck_expr(symtbl, *rhs, function_rettype)?;
            // Find out real type of each arm
            let input_type = op.input_type();
            let t_lhs = infer_type(input_type, lhs1.t).ok_or_else(|| TypeError::TypeMismatch {
                expr_name: format!("{:?}", op).into(),
                got: lhs1.t,
                expected: input_type,
            })?;
            let t_rhs = infer_type(input_type, rhs1.t).ok_or_else(|| TypeError::TypeMismatch {
                expr_name: format!("{:?}", op).into(),
                got: lhs1.t,
                expected: input_type,
            })?;
            // Both arms must be a compatible type
            if let Some(t) = infer_type(t_lhs, t_rhs) {
                let real_lhs = reify_types(t_lhs, t, lhs1);
                let real_rhs = reify_types(t_rhs, t, rhs1);
                // TODO NEXT:
                // We must also figure out the output type, though.  It depends on the binop!
                // Sometimes it depends on the input types (number + number => number, number + i32
                // => i32), and sometimes it doesn't (bool and bool => bool, no guesswork involved)
                Ok(hir::TypedExpr {
                    e: BinOp {
                        op,
                        lhs: Box::new(real_lhs),
                        rhs: Box::new(real_rhs),
                    },
                    t: op.output_type(),
                })
            } else {
                Err(TypeError::BopType {
                    bop: op,
                    expected: input_type,
                    got1: lhs1.t,
                    got2: rhs1.t,
                })
            }
        }
        UniOp { op, rhs } => {
            let rhs = typecheck_expr(symtbl, *rhs, function_rettype)?;
            // Currently, our only valid binops are on numbers.
            let input_type = op.input_type();
            if let Some(t) = infer_type(input_type, rhs.t) {
                // Type `t` is compatible with the expr's return type
                // and the op's input type.
                // We force it to be our output type too, though that
                // isn't *strictly* necessary for general purpose ops.
                let new_rhs = reify_types(rhs.t, t, rhs);
                Ok(hir::TypedExpr {
                    e: UniOp {
                        op,
                        rhs: Box::new(new_rhs),
                    },
                    t: op.output_type(),
                })
            } else {
                Err(TypeError::UopType {
                    op,
                    expected: input_type,
                    got: rhs.t,
                })
            }
        }
        Block { body } => {
            let b = typecheck_exprs(symtbl, body, function_rettype)?;
            let t = last_type_of(&b);
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
            let init_expr = typecheck_expr(symtbl, *init, function_rettype)?;
            let typename = typename.expect("TODO: Try to do type inference here if we can?");
            // If the init expression can match with the declared let type, life is good
            if let Some(t) = infer_type(typename, init_expr.t) {
                let new_init = reify_types(init_expr.t, t, init_expr);
                // Add var to symbol table, proceed
                symtbl.add_var(varname, t, mutable);
                Ok(hir::TypedExpr {
                    e: Let {
                        varname,
                        typename: Some(t),
                        init: Box::new(new_init),
                        mutable,
                    },
                    t: unittype,
                })
            } else {
                Err(TypeError::LetType {
                    name: varname,
                    got: init_expr.t,
                    expected: typename,
                })
            }
        }
        If { cases, falseblock } => {
            let falseblock = typecheck_exprs(symtbl, falseblock, function_rettype)?;
            let assumed_type = last_type_of(&falseblock);
            let mut new_cases = vec![];
            for (cond, body) in cases {
                let cond_expr = typecheck_expr(symtbl, cond, function_rettype)?;
                if type_matches(cond_expr.t, booltype) {
                    // Proceed to typecheck arms
                    let ifbody_exprs = typecheck_exprs(symtbl, body, function_rettype)?;
                    let if_type = last_type_of(&ifbody_exprs);
                    // TODO: Inference?
                    if !type_matches(if_type, assumed_type) {
                        return Err(TypeError::IfType {
                            ifpart: if_type,
                            elsepart: assumed_type,
                        });
                    }

                    // Great, it matches
                    new_cases.push((cond_expr, ifbody_exprs));
                } else {
                    return Err(TypeError::Cond { got: cond_expr.t });
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
            let b = typecheck_exprs(symtbl, body, function_rettype)?;
            let t = last_type_of(&b);
            Ok(hir::TypedExpr {
                e: Loop { body: b },
                t,
            })
        }
        // TODO: Inference?
        Lambda { signature, body } => {
            symtbl.push_scope();
            // add params to symbol table
            for (paramname, paramtype) in signature.params.iter() {
                symtbl.add_var(*paramname, *paramtype, false);
            }
            let body_expr = typecheck_exprs(symtbl, body, Some(signature.rettype))?;
            let bodytype = last_type_of(&body_expr);
            dbg!(INT.fetch_type(bodytype));
            if let Some(t) = infer_type(bodytype, signature.rettype) {
                let new_body = reify_last_types(bodytype, t, body_expr);
                symtbl.pop_scope();
                let lambdatype = signature.to_type();
                Ok(hir::TypedExpr {
                    e: Lambda {
                        signature,
                        body: new_body,
                    },
                    t: lambdatype,
                })
            } else {
                let function_name = INT.intern("lambda");
                symtbl.pop_scope();
                return Err(TypeError::Return {
                    fname: function_name,
                    got: bodytype,
                    expected: signature.rettype,
                });
            }
        }
        Funcall { func, params } => {
            // First, get param types
            let given_params = typecheck_exprs(symtbl, params, function_rettype)?;
            // Then, look up function
            let f = typecheck_expr(symtbl, *func, function_rettype)?;
            let fdef = &*INT.fetch_type(f.t);
            match fdef {
                TypeDef::Lambda(paramtypes, rettype) => {
                    // Now, make sure all the function's params match what it wants
                    let mut inferred_args = vec![];
                    for (given, wanted) in given_params.into_iter().zip(paramtypes) {
                        if let Some(t) = infer_type(given.t, *wanted) {
                            let new_given = reify_types(given.t, t, given);
                            inferred_args.push(new_given);
                        } else {
                            return Err(TypeError::Param {
                                got: given.t,
                                expected: *wanted,
                            });
                        }
                    }
                    Ok(hir::TypedExpr {
                        e: Funcall {
                            func: Box::new(f),
                            params: inferred_args,
                        },
                        t: *rettype,
                    })
                }
                // Something was called but it wasn't a function.  Wild.
                _other => Err(TypeError::Call { got: f.t }),
            }
        }
        Break => Ok(expr.map_type(&|_| unittype)),
        Return { retval } => {
            if let Some(wanted_type) = function_rettype {
                let retval_expr = typecheck_expr(symtbl, *retval, function_rettype)?;
                if let Some(t) = infer_type(retval_expr.t, wanted_type) {
                    // If you do `let x = return y` the type of `x` is `Never`,
                    // while the type of y is the return type of the function.
                    let new_retval = reify_types(retval_expr.t, t, retval_expr);
                    Ok(hir::TypedExpr {
                        t: INT.never(),
                        e: Return {
                            retval: Box::new(new_retval),
                        },
                    })
                } else {
                    Err(TypeError::TypeMismatch {
                        expr_name: "return".into(),
                        expected: wanted_type,
                        got: retval_expr.t,
                    })
                }
            } else {
                // We got a `return` expression in a place where there ain't anything to return
                // from, such as a const initializer
                Err(TypeError::InvalidReturn)
            }
        }
        // TODO: Inference?
        TupleCtor { body } => {
            let body_exprs = typecheck_exprs(symtbl, body, function_rettype)?;
            let body_typesyms = body_exprs.iter().map(|te| te.t).collect();
            let body_type = TypeDef::Tuple(body_typesyms);
            Ok(hir::TypedExpr {
                t: INT.intern_type(&body_type),
                e: TupleCtor { body: body_exprs },
            })
        }
        // TODO: Inference???
        TupleRef { expr, elt } => {
            let body_expr = typecheck_expr(symtbl, *expr, function_rettype)?;
            let expr_typedef = INT.fetch_type(body_expr.t);
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
                Err(TypeError::TupleRef { got: body_expr.t })
            }
        }
        Assign { lhs, rhs } => {
            let lhs_expr = typecheck_expr(symtbl, *lhs, function_rettype)?;
            let rhs_expr = typecheck_expr(symtbl, *rhs, function_rettype)?;
            // So the rules for assignments are, it's valid only if:
            // The LHS is an lvalue (just variable or tuple ref currently)
            // (This basically comes down to "somewhere with a location"...)
            // The LHS is mutable
            // The types of the RHS can be inferred into the type of the LHS
            if !is_mutable_lvalue(symtbl, &lhs_expr.e)? {
                Err(TypeError::Mutability {
                    expr_name: "assignment".into(),
                })
            } else if let Some(t) = infer_type(lhs_expr.t, rhs_expr.t) {
                // TODO: Do we change the type of the LHS?  I THINK we could...
                // I think by definition the type of the LHS is always known
                // and is always a real type.  So this should at worst be a noop?
                let new_lhs = reify_types(lhs_expr.t, t, lhs_expr);
                let new_rhs = reify_types(rhs_expr.t, t, rhs_expr);
                Ok(hir::TypedExpr {
                    t: INT.unit(),
                    e: Assign {
                        lhs: Box::new(new_lhs),
                        rhs: Box::new(new_rhs),
                    },
                })
            } else {
                Err(TypeError::TypeMismatch {
                    expr_name: "assignment".into(),
                    expected: lhs_expr.t,
                    got: rhs_expr.t,
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

fn typecheck_literal(lit: &hir::Literal) -> Result<TypeSym, TypeError> {
    match lit {
        hir::Literal::Integer(_) => Ok(INT.iunknown()),
        hir::Literal::SizedInteger { vl: _, bytes } => Ok(INT.intern_type(&TypeDef::SInt(*bytes))),
        hir::Literal::Bool(_) => Ok(INT.bool()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::*;
    use crate::*;

    fn typecheck_src(src: &str) -> Result<hir::Ir<TypeSym>, TypeError> {
        use crate::parser::Parser;
        let ast = Parser::new(src).parse();
        let ir = hir::lower(&mut rly, &ast);
        typecheck(ir).map_err(|e| {
            eprintln!("typecheck_src got error: {}", e);
            e
        })
    }

    macro_rules! fail_typecheck {
        ( $src: expr, $err_pat: pat ) => {
            match typecheck_src($src) {
                Ok(_) => panic!("Typecheck succeeded and should have failed!"),
                Err($err_pat) => (),
                Err(x) => panic!("Typecheck gave the wrong error: {}", x),
            }
        };
    }

    /// Test symbol table
    #[test]
    fn test_symtbl() {
        let t_foo = INT.intern("foo");
        let t_bar = INT.intern("bar");
        let t_i32 = INT.i32();
        let t_bool = INT.bool();
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
        let t_iunk = INT.iunknown();
        let t_bool = INT.bool();

        let l1 = hir::Literal::Integer(3);
        let l1t = typecheck_literal(&l1).unwrap();
        let l2 = hir::Literal::Bool(true);
        let l2t = typecheck_literal(&l2).unwrap();
        assert!(!type_matches(l1t, l2t));

        assert!(type_matches(l1t, t_iunk));
        assert!(type_matches(l2t, t_bool));
    }

    /// Test binop typechecks
    #[test]
    fn test_binop() {
        let t_i32 = INT.i32();
        let t_i16 = INT.i16();
        let t_iunk = INT.iunknown();
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
                typecheck_expr(tbl, ir, None).unwrap().t,
                t_iunk
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
            assert!(typecheck_expr(tbl, bad_ir, None).is_err());
        }
        // Inference of UnknownInt types
        {
            let ir = *plz(Expr::BinOp {
                op: BOp::Add,
                lhs: plz(Expr::Lit {
                    val: Literal::SizedInteger { vl: 3, bytes: 4 },
                }),
                rhs: plz(Expr::Lit {
                    val: Literal::Integer(4),
                }),
            });
            assert!(type_matches(
                typecheck_expr(tbl, ir, None).unwrap().t,
                t_i32
            ));

            let ir = *plz(Expr::BinOp {
                op: BOp::Add,
                lhs: plz(Expr::Lit {
                    val: Literal::Integer(4),
                }),
                rhs: plz(Expr::Lit {
                    val: Literal::SizedInteger { vl: 3, bytes: 2 },
                }),
            });
            assert!(type_matches(
                typecheck_expr(tbl, ir, None).unwrap().t,
                t_i16
            ));
        }
    }

    /// Test uniop typechecks
    #[test]
    fn test_uniop() {
        let t_i16 = INT.i16();
        let t_i32 = INT.i32();
        let t_iunk = INT.iunknown();
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
                typecheck_expr(tbl, ir, None).unwrap().t,
                t_iunk
            ));
        }
        {
            let ir = *plz(Expr::UniOp {
                op: UOp::Neg,
                rhs: plz(Expr::Lit {
                    val: Literal::SizedInteger { vl: 42, bytes: 4 },
                }),
            });
            assert!(type_matches(
                typecheck_expr(tbl, ir, None).unwrap().t,
                t_i32,
            ));
        }
        {
            let ir = *plz(Expr::UniOp {
                op: UOp::Neg,
                rhs: plz(Expr::Lit {
                    val: Literal::SizedInteger { vl: 91, bytes: 2 },
                }),
            });
            assert!(type_matches(
                typecheck_expr(tbl, ir, None).unwrap().t,
                t_i16
            ));
        }

        {
            let bad_ir = *plz(Expr::UniOp {
                op: UOp::Neg,
                rhs: plz(Expr::Lit {
                    val: Literal::Bool(false),
                }),
            });
            assert!(typecheck_expr(tbl, bad_ir, None).is_err());
        }
    }

    /// TODO
    #[test]
    fn test_block() {}

    /// Do let expr's have the right return?
    #[test]
    fn test_let() {
        let tbl = &mut Symtbl::new();
        let t_i32 = INT.i32();
        let t_unit = INT.unit();
        let fooname = INT.intern("foo");

        use hir::*;
        {
            let ir = *plz(Expr::Let {
                varname: fooname,
                typename: Some(t_i32.clone()),
                init: plz(Expr::Lit {
                    val: Literal::Integer(42),
                }),
                mutable: false,
            });
            assert!(type_matches(
                typecheck_expr(tbl, ir, None).unwrap().t,
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
        let tbl = &mut Symtbl::new();
        let t_i32 = INT.intern_type(&TypeDef::SInt(4));
        let fname = INT.intern("foo");
        let aname = INT.intern("a");
        let bname = INT.intern("b");
        let ftype = INT.intern_type(&TypeDef::Lambda(vec![t_i32, t_i32], t_i32));

        use hir::*;
        {
            let ir = vec![
                *plz(Expr::Let {
                    varname: fname,
                    typename: Some(ftype),
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
            let exprs = &typecheck_exprs(tbl, ir, None).unwrap();
            assert!(type_matches(last_type_of(exprs), t_i32));
            // Is the variable now bound in our symbol table?
            assert_eq!(tbl.get_var(fname).unwrap(), ftype);
        }

        {
            let src = "fn foo(): fn(I32):I32 = fn(x: I32):I32 = x+1 end end";
            typecheck_src(src).unwrap();
        }
    }

    #[test]
    fn test_function_return_inference() {
        //let src = "fn foo(): I32 = 3 end";
        //typecheck_src(src).unwrap();

        let src = r#"fn foo(): I32 = 3 end"#;
        typecheck_src(src).unwrap();
    }

    #[test]
    fn test_bogus_function() {
        let src = "fn foo(): fn(I32):I32 = fn(x: I32):Bool = x+1 end end";
        fail_typecheck!(src, TypeError::Return { .. });
    }

    #[test]
    fn test_rettype() {
        let src = r#"fn foo(): I32 =
        let x: I32 = 10
        return x
end"#;
        typecheck_src(src).unwrap();
    }

    #[test]
    fn test_bad_rettype1() {
        let src = r#"fn foo(): I32 =
        let x: I32 = 10
        return true
        x
end"#;
        fail_typecheck!(src, TypeError::TypeMismatch { .. });
    }

    #[test]
    fn test_bad_rettype2() {
        let src = r#"fn foo(): I32 =
        let x: I32 = 10
        return x
        true
end"#;
        fail_typecheck!(src, TypeError::Return { .. });
    }

    #[test]
    fn test_bad_rettype3() {
        let src = r#"fn foo(): I32 =
        return 3
        return true
        return 4
end"#;
        fail_typecheck!(src, TypeError::TypeMismatch { .. });
    }

    #[test]
    fn test_invalid_return() {
        let src = r#"const F: I32 = return 3
"#;
        fail_typecheck!(src, TypeError::InvalidReturn)
    }

    #[test]
    fn test_tuples() {
        use hir::*;
        let tbl = &mut Symtbl::new();
        assert_eq!(
            typecheck_expr(tbl, *plz(hir::Expr::unit()), None)
                .unwrap()
                .t,
            INT.unit()
        );
    }

    #[test]
    fn test_assign() {
        let src = r#"fn foo() =
    let mut x: I32 = 10
    x = 11
end"#;
        typecheck_src(src).unwrap();
    }

    #[test]
    fn test_assign2() {
        let src = r#"fn foo() =
    let mut x: {I32, I32} = {10, 12}
    x.0 = 11
end"#;
        typecheck_src(src).unwrap();
    }

    #[test]
    fn test_number_types() {
        let src = r#"fn foo() =
        let x: I8 = 8i8
        let x: I16 = 9i16
        let x: I32 = 10i32
        let y: I64 = 11i64
        let y: I128 = 12i128
        let x: I32 = 10
end"#;
        typecheck_src(src).unwrap();
    }

    #[test]
    fn test_bad_assign1() {
        let src = r#"fn foo() =
    let x: I32 = 10
    x = 11
end"#;
        fail_typecheck!(src, TypeError::Mutability { .. });
    }

    #[test]
    #[should_panic]
    fn test_bad_assign2() {
        let src = r#"fn foo() =
    {1,2,3}.3 = 11
end"#;
        typecheck_src(src).unwrap();
    }

    #[test]
    fn test_bad_integer_assignment() {
        let src = r#"fn foo() =
        let x: I32 = 10
        let mut y: I64 = 11i64
        y = x
end"#;
        fail_typecheck!(src, TypeError::TypeMismatch { .. });
    }

    #[test]
    fn test_bad_integer_math() {
        let src = r#"fn foo() =
        let x: I32 = 10
        let mut y: I64 = 11i64
        y + x
end"#;
        fail_typecheck!(src, TypeError::BopType { .. });
    }

    /*
    #[test]
    fn test_inference() {
        let cx = &mut crate::Cx::new();
        let mut engine = InferenceCx::new();
        // Function with unknown input
        let i = engine.insert(TypeInfo::Unknown);
        let o = engine.insert(TypeInfo::Known(InfTypeDef::SInt(4)));
        let f0 = engine.insert(TypeInfo::Known(InfTypeDef::Lambda(vec![i], Box::new(o))));

        // Function with unkown output
        let i = engine.insert(TypeInfo::Known(InfTypeDef::Bool));
        let o = engine.insert(TypeInfo::Unknown);
        let f1 = engine.insert(TypeInfo::Known(InfTypeDef::Lambda(vec![i], Box::new(o))));

        // Unify them...
        engine.unify(f0, f1).unwrap();
        // What do we know about these functions?
        let t0 = engine.reconstruct(cx, f0).unwrap();
        let t1 = engine.reconstruct(cx, f1).unwrap();

        assert_eq!(
            t0,
            cx.intern_type(&TypeDef::Lambda(vec![cx.bool()], cx.i32()),)
        );
        assert_eq!(t0, t1);
    }
    */
}
