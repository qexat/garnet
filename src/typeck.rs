//! Typechecking and other semantic checking.
//! Operates on the HIR.

use std::borrow::Cow;

use crate::hir::{self, ISymtbl, VarBinding};
use crate::{TypeDef, TypeId, TypeSym, VarSym, INT};

#[derive(Debug, Clone)]
pub enum TypeError {
    UnknownVar(VarSym),
    UnknownType(VarSym),
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
        expected: TypeSym,
        got: TypeSym,
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
    StructRef {
        fieldname: VarSym,
        got: TypeSym,
    },
    StructField {
        expected: Vec<VarSym>,
        got: Vec<VarSym>,
    },
    EnumVariant {
        expected: Vec<VarSym>,
        got: VarSym,
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

impl std::error::Error for TypeError {}

impl std::fmt::Display for TypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.format())
    }
}

impl TypeError {
    pub fn format(&self) -> String {
        match self {
            TypeError::UnknownVar(sym) => format!("Unknown var: {}", INT.fetch(*sym)),
            TypeError::UnknownType(sym) => format!("Unknown type: {}", INT.fetch(*sym)),
            TypeError::InvalidReturn => {
                "return expression happened somewhere that isn't in a function!".to_string()
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
                "initializer for variable {}: expected {} ({:?}), got {} ({:?})",
                INT.fetch(*name),
                INT.fetch_type(*expected).get_name(),
                *expected,
                INT.fetch_type(*got).get_name(),
                *got,
            ),
            TypeError::IfType { expected, got } => format!(
                "If block return type is {}, but we thought it should be something like {}",
                INT.fetch_type(*expected).get_name(),
                INT.fetch_type(*got).get_name(),
            ),
            TypeError::Cond { got } => format!(
                "If expr condition is {}, not bool",
                INT.fetch_type(*got).get_name(),
            ),
            TypeError::Param { got, expected } => format!(
                "Function wanted type {} in param but got type {}",
                INT.fetch_type(*expected).get_name(),
                INT.fetch_type(*got).get_name()
            ),
            TypeError::Call { got } => format!(
                "Tried to call function but it is not a function, it is a {}",
                INT.fetch_type(*got).get_name()
            ),
            TypeError::TupleRef { got } => format!(
                "Tried to reference tuple but didn't get a tuple, got {}",
                INT.fetch_type(*got).get_name()
            ),
            TypeError::StructRef { fieldname, got } => format!(
                "Tried to reference field {} of struct, but struct is {}",
                INT.fetch(*fieldname),
                INT.fetch_type(*got).get_name(),
            ),
            TypeError::StructField { expected, got } => format!(
                "Invalid field in struct constructor: expected {:?}, but got {:?}",
                expected, got
            ),
            TypeError::EnumVariant { expected, got } => {
                let expected_names: Vec<String> = expected
                    .into_iter()
                    .map(|nm| (&*INT.fetch(*nm)).clone())
                    .collect();
                format!(
                    "Unknown enum variant '{}', valid ones are {:?}",
                    INT.fetch(*got),
                    expected_names,
                )
            }
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

impl TypeDef {
    /// Returns true if the type does not contain a ForAll somewhere in it.
    fn is_mono(&self) -> bool {
        todo!()
        /*
        match self {
            TypeDef::Unit => true,
            TypeDef::TypeVar(_) => true,
            TypeDef::ExistentialVar(_) => true,
            TypeDef::ForAll(_, _) => false,
            TypeDef::Function(a, b) => a.iter().all(|x| x.is_mono()) && b.is_mono(),
            TypeDef::Bool => true,
            TypeDef::SInt(_) => true,
            TypeDef::UnknownInt => true,
            TypeDef::Never => true,
            TypeDef::Tuple(ts) => ts.iter().all(|x| x.is_mono()),
        }
            */
    }

    /// Returns any unsolved type vars that exist in the type.
    fn free_vars(&self) -> Vec<TypeSym> {
        todo!()
        /*
        match self {
            TypeDef::Unit => vec![],
            TypeDef::TypeVar(_) => vec![],
            TypeDef::ExistentialVar(id) => vec![id.clone()],
            TypeDef::ForAll(_, t) => t.free_vars(),
            TypeDef::Function(a, b) => {
                // ahahahaha this is the most terrible way to concatenate
                // lists ever.  Rust is *not a fan* of the functional definition
                // of this.
                let mut ret = a.iter().flat_map(|x| x.free_vars()).collect::<Vec<_>>();
                ret.extend(b.free_vars());
                ret
            }
            TypeDef::Bool => vec![],
            TypeDef::SInt(_) => vec![],
            TypeDef::UnknownInt => vec![],
            TypeDef::Never => vec![],
            TypeDef::Tuple(ts) => ts.iter().flat_map(|x| x.free_vars()).collect::<Vec<_>>(),
        }
            */
    }

    /// Instantiate a type, I think.
    /// Whatever that means.
    /// Substitute a type variable for a new type apparently,
    /// though that new type may be another type variable, or
    /// even the same one.
    fn instantiate(&self, v: &TypeId, s: &TypeSym) -> TypeSym {
        // TODO: I THINK this signature is correct, verify
        todo!()
        /*
        match self {
            TypeDef::Unit => TypeDef::Unit,
            TypeDef::TypeVar(v2) => {
                if v == v2 {
                    s.clone()
                } else {
                    self.clone()
                }
            }
            TypeDef::ExistentialVar(_id) => self.clone(),
            TypeDef::ForAll(v2, t) => TypeDef::ForAll(*v2, Box::new(t.instantiate(v, s))),
            TypeDef::Function(a, b) => {
                let params = a.iter().map(|x| x.instantiate(v, s)).collect::<Vec<_>>();
                TypeDef::Function(params, Box::new(b.instantiate(v, s)))
            }
            TypeDef::Bool => TypeDef::Bool,
            TypeDef::SInt(i) => TypeDef::SInt(*i),
            TypeDef::UnknownInt => TypeDef::UnknownInt,
            TypeDef::Never => TypeDef::Never,
            TypeDef::Tuple(ts) => {
                TypeDef::Tuple(ts.iter().map(|t| t.instantiate(v, s)).collect::<Vec<_>>())
            }
        }
        */
    }
}

/// The kinds of type facts we may know while type checking stuff.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ContextItem {
    /// Term variable typings x : A
    TermVar(TypeSym),
    Assump(VarSym, TypeSym),
    /// Existential type variables α-hat: Unsolved
    ExistentialVar(TypeId),
    /// And solved: α-hat = τ
    SolvedExistentialVar(TypeId, TypeSym),
    /// Marker: ◮α-hat
    Marker(TypeId),
}

impl ContextItem {
    /// Returns whether the id is an assumption for this context item?
    fn is_assump(&self, id: VarSym) -> bool {
        match self {
            ContextItem::Assump(x, _) => *x == id,
            _ => false,
        }
    }

    /// Returns whether the id is a solution for this context item?
    fn is_solution(&self, id: &TypeId) -> bool {
        match self {
            ContextItem::SolvedExistentialVar(u, _) => u == id,
            _ => false,
        }
    }
}

/// Type checking context.  Contains what is known about the types
/// being inferred/checked within the current decl/expression(?).
// TODO:
//
// Currently the implementation is very inefficient, doing lots of cloning of itself and
// splitting/merging is inner vec in brute force ways.  It may be better to switch to `im::Vector`,
// but it also may need a fairly large context before it matters.  It may also be more efficient
// to replace markers with stack of vec's or such.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct TCContext {
    items: Vec<ContextItem>,
}

impl TCContext {
    /// Add a new item to the context, consuming it.
    fn add(mut self, t: ContextItem) -> TCContext {
        self.items.push(t);
        self
    }

    fn add_all(mut self, t: impl IntoIterator<Item = ContextItem>) -> TCContext {
        self.items.extend(t);
        self
    }

    fn contains(&self, t: &ContextItem) -> bool {
        self.items.contains(t)
    }

    /// Concatenates two contexts, consuming them and returning a new one.
    fn concat(mut self, c2: TCContext) -> TCContext {
        self.items.extend(c2.items.into_iter());
        self
    }

    /// splits the context in two at the given item, if it exists,
    /// omitting that item from either side.
    ///
    /// Very crude and inefficient, but screw it.  We can make it better with
    /// an immutable list or something later.
    fn hole(mut self, member: &ContextItem) -> Option<(TCContext, TCContext)> {
        let idx = self.items.iter().position(|x| x == member)?;
        let rest = self.items.split_off(idx);
        // Data.Sequence.drop n ditches the first n items,
        // and I THINK what we need to do here is make sure the
        // item we give it is removed from either side.
        //
        // Yes, this is correct, verified it against the reference impl
        let rest = (&rest[1..]).into_iter().cloned().collect();
        Some((self, TCContext { items: rest }))
    }

    /// Splits the context into three, around the given items
    fn hole2(
        self,
        mem1: &ContextItem,
        mem2: &ContextItem,
    ) -> Option<(TCContext, TCContext, TCContext)> {
        let (a, ctx) = self.hole(mem1)?;
        let (b, c) = ctx.hole(mem2)?;
        Some((a, b, c))
    }

    /// ???
    ///
    /// Returns whether there is a type assumption for the given var ID?
    fn assump(&self, id: VarSym) -> Option<TypeSym> {
        let mut assumptions = self.items.iter().filter(|x| x.is_assump(id));
        match (assumptions.next(), assumptions.next()) {
            (Some(ContextItem::Assump(_, t)), None) => Some(t.clone()),
            (None, None) => None,
            other => panic!("ctxAssump: multiple types for variable: {:?}", other),
        }
    }

    fn solution(&self, id: &TypeId) -> Option<TypeSym> {
        let mut solutions = self.items.iter().filter(|x| x.is_solution(id));
        eprintln!(
            "Solutions for {:?}: {:?}",
            id,
            solutions.clone().collect::<Vec<_>>()
        );
        match (solutions.next(), solutions.next()) {
            (Some(ContextItem::SolvedExistentialVar(_, t)), None) => Some(t.clone()),
            (None, None) => None,
            other => panic!("ctxAssump: multiple types for variable: {:?}", other),
        }
    }

    /// Return all the things in the context up until the given item, exclusive
    /// (I think)
    fn until(&self, mem: &ContextItem) -> TCContext {
        self.clone().hole(mem).unwrap().0
    }

    /// Returns all the things in the context up until the given item, inclusive.
    fn until_after(&self, mem: &ContextItem) -> TCContext {
        let idx = self
            .items
            .iter()
            .position(|x| x == mem)
            .unwrap_or(self.items.len());
        let mut new = self.clone();
        new.items.truncate(idx);
        new
    }

    fn type_is_well_formed(&self, t: TypeSym) -> Result<(), TypeError> {
        todo!()
        /*
            match t {
                Type::Unit => Ok(()),
                Type::TypeVar(t_id) => {
                    if self.contains(&ContextItem::TermVar(*t_id)) {
                        Ok(())
                    } else {
                        let msg = format!("Unbound type variable {:?}", t_id);
                        Err(msg)
                    }
                }

                Type::ExistentialVar(te_id) => {
                    let has_solution = self.solution(te_id).is_some();
                    if self.contains(&ContextItem::ExistentialVar(*te_id)) || has_solution {
                        Ok(())
                    } else {
                        let msg = format!("Unbound existential variable {:?}", te_id);
                        Err(msg)
                    }
                }
                Type::ForAll(t_id, ty) => {
                    let ctx = self.clone().add(ContextItem::TermVar(*t_id));
                    ctx.type_is_well_formed(ty)
                }
                Type::Function(ty_a, ty_b) => {
                    /*
                    for t in ty_a {
                        let _ = self.type_is_well_formed(t)?;
                    }
                    */
                    ty_a.iter()
                        .map(|t| self.type_is_well_formed(t))
                        .collect::<Result<_, _>>()?;
                    self.type_is_well_formed(ty_b)
                }
                Type::Bool => Ok(()),
                Type::SInt(_) => Ok(()),
                // TODO: ...maybe not, since we don't know exactly what type it is?
                Type::UnknownInt => Ok(()),
                Type::Never => Ok(()),
                Type::Tuple(ts) => ts
                    .iter()
                    .map(|t| self.type_is_well_formed(t))
                    .collect::<Result<_, _>>(),
            }
        */
    }

    /// Apply a substitution, basically attempting to solve for the unknowns of a type(?).
    /// Fig 8 of the paper.
    fn subst(&self, ts: TypeSym) -> TypeSym {
        let t = INT.fetch_type(ts);
        todo!()
        /*
            match t {
                TypeDef::Unit => TypeDef::Unit,
                TypeDef::TypeVar(_t_id) => t.clone(),
                TypeDef::ExistentialVar(te_id) => {
                    /*
                    // maybe t (applySubst ctx) (ctxSolution ctx v)
                    // maybe :: b->(a->b) -> Maybe a -> b
                    //     Applies the second argument to the third, when it is Just x, otherwise returns the first argument.
                    if let Some(solution) = self.solution(te_id) {
                        self.subst(&solution)
                    } else {
                        t.clone()
                    }
                    */
                    self.solution(te_id)
                        .map(|x| self.subst(&x))
                        .unwrap_or(t.clone())
                }
                TypeDef::Function(ty_a, ty_b) => {
                    let new_params = ty_a.iter().map(|x| self.subst(x)).collect();
                    TypeDef::Function(new_params, Box::new(self.subst(ty_b)))
                }
                TypeDef::ForAll(v, t) => TypeDef::ForAll(v.clone(), Box::new(self.subst(t))),
                TypeDef::Bool => TypeDef::Bool,
                TypeDef::SInt(i) => TypeDef::SInt(*i),
                TypeDef::UnknownInt => TypeDef::UnknownInt,
                TypeDef::Never => TypeDef::Never,
                TypeDef::Tuple(ts) => TypeDef::Tuple(ts.iter().map(|x| self.subst(x)).collect()),
            }
        */
    }
}

#[derive(Clone, Default)]
struct CheckState {
    ctx: TCContext,
    next_existential_var: usize,
}

impl CheckState {
    /// Creates an arbitrary synthetic type identifier for labelling
    /// unsolved type variables
    fn next_existential_var(&mut self) -> TypeId {
        self.next_existential_var += 1;
        TypeId(self.next_existential_var)
    }

    fn type_sub(&mut self, t1: TypeSym, t2: TypeSym) -> Result<(), TypeError> {
        //eprintln!("Doing type_sub on {:?}, {:?}", t1, t2);
        let td1 = INT.fetch_type(t1);
        let td2 = INT.fetch_type(t2);
        match (td1, td2) {
            /*
             * TODO
            (TypeDef::Unit, TypeDef::Unit) => Ok(()),
            (TypeDef::Bool, TypeDef::Bool) => Ok(()),
            (TypeDef::SInt(i1), TypeDef::SInt(i2)) if i1 == i2 => Ok(()),
            (TypeDef::TypeVar(a), TypeDef::TypeVar(b)) if a == b => Ok(()),
            (TypeDef::ExistentialVar(a), TypeDef::ExistentialVar(b)) if a == b => Ok(()),
            (TypeDef::Lambda(a1, b1), TypeDef::Lambda(a2, b2)) => {
                for (param1, param2) in a1.iter().zip(a2) {
                    self.type_sub(param1.clone(), param2)?;
                }
                self.type_sub(*b1, *b2)
            }
            (TypeDef::ForAll(v, a), b) => {
                let heckin_a_hat = self.next_existential_var();
                let heckin_a_prime = a.instantiate(&v, &TypeDef::ExistentialVar(heckin_a_hat));
                let marker = ContextItem::Marker(heckin_a_hat);
                self.ctx = self
                    .ctx
                    .clone()
                    .add(marker.clone())
                    .add(ContextItem::ExistentialVar(heckin_a_hat));
                self.type_sub(heckin_a_prime, b)?;
                self.ctx = self.ctx.until(&marker);
                Ok(())
            }
            (a, TypeDef::ForAll(v, b)) => {
                let var = ContextItem::TermVar(v.clone());
                self.ctx = self.ctx.clone().add(var.clone());
                self.type_sub(a, *b)?;
                self.ctx = self.ctx.until(&var);
                Ok(())
            }
            (TypeDef::ExistentialVar(a_hat), a) if !a.free_vars().contains(&a_hat) => {
                eprintln!("Doing instL");
                let x = self.instantiate_l(a_hat, a);
                eprintln!("Done with instL");
                x
            }
            (a, TypeDef::ExistentialVar(a_hat)) if !a.free_vars().contains(&a_hat) => {
                self.instantiate_r(a, a_hat)
            }
            */
            (a, b) => Err(TypeError::TypeMismatch {
                expr_name: "todo".into(),
                got: t1,
                expected: t2,
            }),
        }
    }

    fn instantiate_l(&mut self, a_hat: TypeId, t: TypeSym) -> Result<(), TypeError> {
        eprintln!("  Instantiate_l on {:?} and {:?}", a_hat, t);
        // The lexi-lambda Haskell impl for these functions does some kind of
        // big screwy pattern-match, so I'm trying to take it apart into pieces here.
        // go ctx -- InstLSolve
        //  | True <- isMono t
        //  , Just (l, r) <- ctxHole (CtxEVar â) ctx
        //  , Right _ <- l ⊢ t
        //  = putCtx $ l |> CtxSolved â t <> r
        todo!()
        /*
        if t.is_mono() {
            eprintln!("  Is mono");
            if let Some((l, r)) = self.ctx.clone().hole(&ContextItem::ExistentialVar(a_hat)) {
                // Apparently Either::Right is Ok in Haskell,
                // and Left is Err?
                // ("Right" also means "correct", per the docs)
                if l.type_is_well_formed(&t).is_ok() {
                    eprintln!("  {:?} is well formed in context {:?}", &t, l);
                    let ctx = l.add(ContextItem::SolvedExistentialVar(a_hat, t.clone()));
                    let ctx = ctx.concat(r);
                    self.ctx = ctx;
                    return Ok(());
                } else {
                    eprintln!(
                        "  whoops, {:?} is not well formed in context {:?}, let's carry on",
                        &t, l
                    );
                }
            }
        }

            match &t {
                // go ctx -- InstLReach
                //    | TEVar â' <- t
                //    , Just (l, m, r) <- ctxHole2 (CtxEVar â) (CtxEVar â') ctx
                //    = putCtx $ l |> CtxEVar â <> m |> CtxSolved â' (TEVar â) <> r
                Type::ExistentialVar(a_hat_fuckin_prime) => {
                    let ev_a = ContextItem::ExistentialVar(a_hat.clone());
                    let ev_b = ContextItem::ExistentialVar(a_hat_fuckin_prime.clone());
                    if let Some((l, m, r)) = self.ctx.clone().hole2(&ev_a, &ev_b) {
                        /*
                        let ctx = l.add(ev_a.clone());
                        let ctx = ctx.concat(m);
                        let ctx = ctx.add(ContextItem::SolvedExistentialVar(
                            a_hat_fuckin_prime.clone(),
                            Type::ExistentialVar(a_hat.clone()),
                        ));
                        let ctx = ctx.concat(r);
                        self.ctx = ctx;
                        */

                        let ctx = l
                            .add(ev_a)
                            .concat(m)
                            .add(ContextItem::SolvedExistentialVar(
                                *a_hat_fuckin_prime,
                                Type::ExistentialVar(a_hat),
                            ))
                            .concat(r);
                        eprintln!("  InstLReach gave result {:#?}", ctx);
                        self.ctx = ctx;
                        return Ok(());
                    }
                }
                //  go ctx -- InstLArr
                // | Just (l, r) <- ctxHole (CtxEVar â) ctx
                // , TArr a b <- t
                // = do â1 <- freshEVar
                //   â2 <- freshEVar
                //   putCtx $ l |> CtxEVar â2 |> CtxEVar â1 |> CtxSolved â (TArr (TEVar â1) (TEVar â2)) <> r
                //   instR a â1
                //   ctx' <- getCtx
                //   instL â2 (applySubst ctx' b)
                Type::Function(params, b) => {
                    if let Some((l, r)) = self.ctx.clone().hole(&ContextItem::ExistentialVar(a_hat)) {
                        let a_hat_1s = params
                            .iter()
                            .map(|_| self.next_existential_var())
                            .collect::<Vec<_>>();
                        let a_hat_2 = self.next_existential_var();
                        let solved = Type::Function(
                            a_hat_1s.iter().map(|a| Type::ExistentialVar(*a)).collect(),
                            Box::new(Type::ExistentialVar(a_hat_2)),
                        );
                        self.ctx = l
                            .add(ContextItem::ExistentialVar(a_hat_2))
                            .add_all(a_hat_1s.iter().map(|a| ContextItem::ExistentialVar(*a)))
                            .add(ContextItem::SolvedExistentialVar(a_hat, solved))
                            .concat(r);
                        for (a, param) in a_hat_1s.iter().zip(params) {
                            self.instantiate_r(param.clone(), *a)?;
                        }
                        let new_t = self.ctx.subst(&*b);
                        return self.instantiate_l(a_hat_2, new_t);
                    }
                    /*
                    eprintln!("  Non-mono function");
                    if let Some((l, r)) = self.ctx.clone().hole(&ContextItem::ExistentialVar(a_hat)) {
                        let a_hat_1 = self.next_existential_var();
                        let a_hat_2 = self.next_existential_var();
                        let solved = Type::Function(
                            Box::new(Type::ExistentialVar(a_hat_1)),
                            Box::new(Type::ExistentialVar(a_hat_2)),
                        );
                        self.ctx = l
                            .add(ContextItem::ExistentialVar(a_hat_2))
                            .add(ContextItem::ExistentialVar(a_hat_1))
                            .add(ContextItem::SolvedExistentialVar(a_hat, solved))
                            .concat(r);
                        self.instantiate_r((**a).clone(), a_hat_1)?;
                        let new_t = self.ctx.subst(&*b);
                        return self.instantiate_l(a_hat_2, new_t);
                    }
                    */
                }
                // go ctx -- InstLArrR
                // | TAll b s <- t
                // = do putCtx $ ctx |> CtxVar b
                //     instL â s
                //     Just (ctx', _) <- ctxHole (CtxVar b) <$> getCtx
                //     putCtx ctx'
                Type::ForAll(b, s) => {
                    eprintln!("  Non-mono forall");
                    let var = ContextItem::TermVar(b.clone());
                    self.ctx = self.ctx.clone().add(var.clone());
                    self.instantiate_l(a_hat, (**s).clone())?;
                    let (ctx_prime, _) = self.ctx.clone().hole(&var).expect("Should never fail?");
                    self.ctx = ctx_prime;
                    return Ok(());
                }
                _other => (),
            }
            eprintln!("  instantiate_l didn't find a solution");
            Err(format!(
                "Failed to instantiate {:?} to {:?} in instantiate_l",
                &a_hat, &t
            ))
        */
    }

    /// This is similar to instantiate_l
    fn instantiate_r(&mut self, t: TypeSym, a_hat: TypeId) -> Result<(), TypeError> {
        //  go ctx -- InstRSolve
        //    | True <- isMono t
        //    , Just (l, r) <- ctxHole (CtxEVar â) ctx
        //    , Right _ <- l ⊢ t
        //    = putCtx $ l |> CtxSolved â t <> r
        let td = INT.fetch_type(t);
        if td.is_mono() {
            if let Some((l, r)) = self.ctx.clone().hole(&ContextItem::ExistentialVar(a_hat)) {
                // Apparently Either::Right is Ok in Haskell,
                // and Left is Err?
                // ("Right" also means "correct", per the docs)
                if l.type_is_well_formed(t).is_ok() {
                    let ctx = l
                        .add(ContextItem::SolvedExistentialVar(a_hat, t.clone()))
                        .concat(r);
                    self.ctx = ctx;
                    return Ok(());
                }
            }
        }

        todo!()
        /*
        match &t {
            //  go ctx -- InstRReach
            //    | TEVar â' <- t
            //    , Just (l, m, r) <- ctxHole2 (CtxEVar â) (CtxEVar â') ctx
            //    = putCtx $ l |> CtxEVar â <> m |> CtxSolved â' (TEVar â) <> r
            Type::ExistentialVar(a_hat_fuckin_prime) => {
                let ev_a = ContextItem::ExistentialVar(a_hat);
                let ev_b = ContextItem::ExistentialVar(*a_hat_fuckin_prime);
                if let Some((l, m, r)) = self.ctx.clone().hole2(&ev_a, &ev_b) {
                    let ctx = l
                        .add(ev_a.clone())
                        .concat(m)
                        .add(ContextItem::SolvedExistentialVar(
                            *a_hat_fuckin_prime,
                            Type::ExistentialVar(a_hat),
                        ))
                        .concat(r);
                    self.ctx = ctx;
                    return Ok(());
                }
            }
            // go ctx -- InstRArr
            // | Just (l, r) <- ctxHole (CtxEVar â) ctx
            // , TArr a b <- t
            // = do â1 <- freshEVar
            //     â2 <- freshEVar
            //     putCtx $ l |> CtxEVar â2 |> CtxEVar â1 |> CtxSolved â (TArr (TEVar â1) (TEVar â2)) <> r
            //     instL â1 a
            //     ctx' <- getCtx
            //     instR (applySubst ctx' b) â2
            Type::Function(params, rettype) => {
                if let Some((l, r)) = self.ctx.clone().hole(&ContextItem::ExistentialVar(a_hat)) {
                    let a_hat_1s = params
                        .iter()
                        .map(|_| self.next_existential_var())
                        .collect::<Vec<_>>();
                    let a_hat_2 = self.next_existential_var();
                    let solved = Type::Function(
                        a_hat_1s.iter().map(|a| Type::ExistentialVar(*a)).collect(),
                        Box::new(Type::ExistentialVar(a_hat_2)),
                    );
                    self.ctx = l
                        .add(ContextItem::ExistentialVar(a_hat_2))
                        .add_all(a_hat_1s.iter().map(|a| ContextItem::ExistentialVar(*a)))
                        .add(ContextItem::SolvedExistentialVar(a_hat, solved))
                        .concat(r);
                    for (a, param) in a_hat_1s.iter().zip(params) {
                        self.instantiate_l(*a, param.clone())?;
                    }
                    let new_t = self.ctx.subst(&*rettype);
                    return self.instantiate_r(new_t, a_hat_2);
                }
            }
            // go ctx -- InstRArrL
            //  | TAll b s <- t
            //  = do â' <- freshEVar
            //       putCtx $ ctx |> CtxMarker â' |> CtxEVar â'
            //       instR (inst (b, TEVar â') s) â
            //       Just (ctx', _) <- ctxHole (CtxMarker â') <$> getCtx
            //       putCtx ctx'
            Type::ForAll(b, s) => {
                let a_hat_fucking_prime = self.next_existential_var();
                let ahatp_var = ContextItem::ExistentialVar(a_hat_fucking_prime);
                let ahatp_marker = ContextItem::Marker(a_hat_fucking_prime);
                let ctx = self
                    .ctx
                    .clone()
                    .add(ahatp_marker.clone())
                    .add(ahatp_var.clone());
                self.ctx = ctx;
                let inst_type = s.instantiate(&b, &Type::ExistentialVar(a_hat_fucking_prime));
                self.instantiate_r(inst_type, a_hat)?;
                let (ctx_prime, _) = self
                    .ctx
                    .clone()
                    .hole(&ahatp_marker)
                    .expect("Should never happen?");
                self.ctx = ctx_prime;
                return Ok(());
            }
            _other => (),
        }
        Err(format!(
            "Failed to instantiate {:?} to {:?} in instantiate_r",
            &a_hat, &t
        ))
            */
    }

    /// Returns Ok if the result of the expression matches the given type,
    /// an error otherwise.
    ///
    /// TODO: Should return TypedExpr<TypeSym>
    fn check(&mut self, expr: &hir::TypedExpr<()>, t: &TypeSym) -> Result<(), TypeError> {
        todo!()
        /*
        match (expr, t) {
            (Ast::Unit, Type::Unit) => return Ok(()),
            (Ast::Bool(_), Type::Bool) => return Ok(()),
            (e, Type::ForAll(v, a)) => {
                let check_state = &mut CheckState {
                    ctx: self.ctx.clone().add(ContextItem::TermVar(v.clone())),
                    ..*self
                };
                return check_state.check(e, a);
            }
            (Ast::Lambda(params, body), Type::Function(fnparams, fnrettype)) => {
                let mut new_ctx = self.ctx.clone();
                for (xx, aa) in params.iter().zip(fnparams) {
                    new_ctx = new_ctx.add(ContextItem::Assump(xx.clone(), aa.clone()));
                }

                let check_state = &mut CheckState {
                    ctx: new_ctx,
                    ..*self
                };
                return check_state.check(body, fnrettype);
            }
            (e, b) => {
                eprintln!("Checking thing: {:?} {:?}", e, b);
                let a = self.infer(e)?;
                eprintln!("Type of {:?} is: {:?}", e, a);
                let x = self.type_sub(self.ctx.subst(&a), self.ctx.subst(b));
                eprintln!("Type subbed: {:?}, {:?}", x, self.ctx);
                x
            }
        }
            */
    }

    /// Infers the type of the expression.
    ///
    /// TODO: Should return TypedExpr<TypeSym>
    fn infer(&mut self, expr: &hir::TypedExpr<()>) -> Result<TypeSym, TypeError> {
        todo!()
        /*
            match expr {
                Ast::Unit => Ok(Type::Unit),
                Ast::Bool(_) => Ok(Type::Bool),
                Ast::Int(_) => Ok(Type::UnknownInt),
                // TODO: Verify int sizes make sense
                Ast::SizedInt(_, size) => Ok(Type::SInt(*size)),
                Ast::BinOp(op, l, r) => {
                    use BinOp::*;
                    // Find the type the binop expects
                    match op {
                        And | Or | Xor => {
                            let target_type = Type::Bool;
                            self.check(l, &target_type)?;
                            self.check(r, &target_type)?;
                            // For these the source type and target type are always the same,
                            // not always true for things such as ==
                            Ok(target_type)
                        }
                        Add | Sub | Mul | Div => {
                            // If we have a numerical operation, we find the types
                            // of the arguments and make sure they are matching numeric
                            // types.
                            let tl = self.infer(l)?;
                            let tr = self.infer(r)?;
                            match (&tl, &tr) {
                                (Type::UnknownInt, Type::UnknownInt) => Ok(Type::UnknownInt),
                                (Type::SInt(s1), Type::SInt(s2)) if s1 == s2 => Ok(Type::SInt(*s1)),
                                // Infer types of unknown ints
                                // TODO: Unknown vars will have existential types or such, which also
                                // need to be handled somehow..
                                (Type::SInt(_), _) => {
                                    self.check(r, &tl)?;
                                    Ok(tl)
                                }
                                (_, Type::SInt(_)) => {
                                    self.check(r, &tr)?;
                                    Ok(tr)
                                }
                                (_, _) => Err(format!(
                                    "Numerical op {:?} has un-matching types: {:?} and {:?}",
                                    op, tl, tr
                                )),
                            }
                        }
                    }
                }
                Ast::ExistentialVar(x) => {
                    let ty = self
                        .ctx
                        .assump(x)
                        .ok_or(format!("Unbound variable {:?}", x));
                    ty
                    // TODO: Figure out MBones's eager instantiation
                    // vs the lazy instantiation default.
                    //let tid = self.next_existential_var();
                    //Ok(self.instantiate(tid, ty))
                }
                Ast::TypeAnnotation(e, a) => {
                    self.ctx.type_is_well_formed(a)?;
                    self.check(e, a)?;
                    return Ok(a.clone());
                }
                Ast::Lambda(params, body) => {
                    eprintln!("Inferring lambda: {:?}", expr);
                    let rettype = self.next_existential_var();
                    let mut paramtypes = vec![];
                    // The order of these matters: We have to do all the params and the rettype,
                    // then the assumptions about the params.
                    for _ in params {
                        let paramtype = self.next_existential_var();
                        self.ctx = self.ctx.clone().add(ContextItem::ExistentialVar(paramtype));
                        eprintln!(" ahat: {:?}, ahat': {:?}", paramtype, rettype);
                        eprintln!(" ctx: {:?}", &self.ctx);
                        paramtypes.push(paramtype);
                    }
                    self.ctx = self.ctx.clone().add(ContextItem::ExistentialVar(rettype));
                    for (nm, ty) in params.iter().zip(paramtypes.iter()) {
                        self.ctx = self
                            .ctx
                            .clone()
                            .add(ContextItem::Assump(nm.clone(), Type::ExistentialVar(*ty)));
                    }
                    // self.check() modifies our ctx so we have to snip off everything in it
                    // after the last thing we actually want without altering the things before
                    // it.  So we add an `until_after()` method, which also works correctly on
                    // functions with 0 params.
                    self.check(body, &Type::ExistentialVar(rettype))?;
                    eprintln!("Check done: {:?}", self.ctx);
                    self.ctx = self.ctx.until_after(&ContextItem::ExistentialVar(rettype));
                    eprintln!("until done: {:?}", self.ctx);
                    let paramtypes = paramtypes
                        .iter()
                        .map(|paramtype| Type::ExistentialVar(*paramtype))
                        .collect();
                    return Ok(Type::Function(
                        paramtypes,
                        Box::new(Type::ExistentialVar(rettype)),
                    ));
                    /*
                    let paramtype = self.next_existential_var();
                    let rettype = self.next_existential_var();
                    self.ctx = self
                        .ctx
                        .clone()
                        .add(ContextItem::ExistentialVar(paramtype))
                        .add(ContextItem::ExistentialVar(rettype))
                        .add(ContextItem::Assump(
                            params.clone(),
                            Type::ExistentialVar(paramtype),
                        ));
                    eprintln!(" ahat: {:?}, ahat': {:?}", paramtype, rettype);
                    eprintln!(" ctx: {:?}", &self.ctx);
                    self.check(body, &Type::ExistentialVar(rettype))?;
                    eprintln!("Check done: {:?}", self.ctx);
                    self.ctx = self.ctx.until(&ContextItem::Assump(
                        params.clone(),
                        Type::ExistentialVar(paramtype),
                    ));
                    eprintln!("until done: {:?}", self.ctx);
                    return Ok(Type::Function(
                        Box::new(Type::ExistentialVar(paramtype)),
                        Box::new(Type::ExistentialVar(rettype)),
                    ));
                    */
                }
                Ast::Call(f, params) => {
                    let ftype = self.infer(f)?;
                    let t = self.ctx.subst(&ftype);
                    self.infer_application(&t, &*params)
                }
                Ast::TupleLit(vls) => {
                    // TODO: Not sure this is at all correct.
                    let ts = vls
                        .iter()
                        .map(|v| self.infer(v))
                        .collect::<Result<Vec<_>, _>>()?;
                    Ok(Type::Tuple(ts))
                }
            }
        */
    }

    /// This does type inference to a function call expression, t(expr).
    /// application as in function application, not as in a program
    ///
    /// t is the function's inferred type, exprs are the args
    fn infer_application(
        &mut self,
        t: &TypeSym,
        exprs: &[hir::TypedExpr<()>],
    ) -> Result<TypeSym, TypeError> {
        todo!()
        /*
            match t {
                Type::ForAll(v, a) => {
                    let a_hat = self.next_existential_var();
                    self.ctx = self.ctx.clone().add(ContextItem::ExistentialVar(a_hat));
                    let b = a.instantiate(v, &Type::ExistentialVar(a_hat));
                    self.infer_application(&b, exprs)
                }
                Type::ExistentialVar(a_hat) => {
                    let rettype = self.next_existential_var();
                    let rettype_var = ContextItem::ExistentialVar(rettype);
                    let (l, r) = self
                        .ctx
                        .clone()
                        .hole(&ContextItem::ExistentialVar(*a_hat))
                        .expect("Should not fail");
                    let mut paramtypes = vec![];
                    let mut paramtype_vars = vec![];
                    let mut paramtype_types = vec![];
                    for _ in exprs {
                        let paramtype = self.next_existential_var();
                        let paramtype_var = ContextItem::ExistentialVar(paramtype);
                        paramtypes.push(paramtype);
                        paramtype_vars.push(paramtype_var);
                        paramtype_types.push(Type::ExistentialVar(paramtype));
                    }
                    let t = Type::Function(
                        paramtype_types.clone(),
                        Box::new(Type::ExistentialVar(rettype)),
                    );
                    self.ctx = l
                        .add(rettype_var)
                        .add_all(paramtype_vars)
                        .add(ContextItem::SolvedExistentialVar(*a_hat, t))
                        .concat(r);
                    for (expr, ty) in exprs.iter().zip(paramtype_types) {
                        self.check(expr, &ty)?;
                    }
                    Ok(Type::ExistentialVar(rettype))
                    /*
                    let paramtype = self.next_existential_var();
                    let rettype = self.next_existential_var();
                    let paramtype_var = ContextItem::ExistentialVar(paramtype);
                    let rettype_var = ContextItem::ExistentialVar(rettype);
                    let (l, r) = self
                        .ctx
                        .clone()
                        .hole(&ContextItem::ExistentialVar(*a_hat))
                        .expect("Should not fail");
                    let t = Type::Function(
                        Box::new(Type::ExistentialVar(paramtype)),
                        Box::new(Type::ExistentialVar(rettype)),
                    );
                    self.ctx = l
                        .add(rettype_var)
                        .add(paramtype_var)
                        .add(ContextItem::SolvedExistentialVar(*a_hat, t))
                        .concat(r);
                    self.check(exprs, &Type::ExistentialVar(paramtype))?;
                    Ok(Type::ExistentialVar(rettype))
                    */
                }
                Type::Function(params, rettype) => {
                    for (p, e) in params.iter().zip(exprs) {
                        self.check(e, p)?;
                    }
                    Ok((**rettype).clone())
                }
                other => Err(format!(
                    "Cannot apply function of type {:?} to args {:?}",
                    other, exprs
                )),
            }
        */
    }
}

/// TODO: Having all these methods here instead of with the defintion is jank,
/// do something about it.  They're here so we have the types in scope and don't
/// make a circular dependency, which is a REAL GOOD reason to do it this way, but
/// also a big fat code smell.
impl ISymtbl {
    /// Create new symbol table with some built-in functions.
    ///
    /// Also see the prelude defined in `backend/rust.rs`
    pub fn new_with_defaults() -> Self {
        let mut x = Self::default();
        // We add a built-in function for printing, currently.
        {
            let name = INT.intern("__println");
            let typesym = INT.intern_type(&TypeDef::Lambda(vec![INT.i32()], INT.unit()));
            x.add_var(name, typesym, false);
        }
        {
            let name = INT.intern("__println_bool");
            let typesym = INT.intern_type(&TypeDef::Lambda(vec![INT.bool()], INT.unit()));
            x.add_var(name, typesym, false);
        }
        {
            let name = INT.intern("__println_i64");
            let typesym = INT.intern_type(&TypeDef::Lambda(vec![INT.i64()], INT.unit()));
            x.add_var(name, typesym, false);
        }
        {
            let name = INT.intern("__println_i16");
            let typesym = INT.intern_type(&TypeDef::Lambda(vec![INT.i16()], INT.unit()));
            x.add_var(name, typesym, false);
        }
        x
    }

    fn add_type(&mut self, name: VarSym, typedef: TypeSym) {
        self.types.insert(name, typedef);
    }

    fn get_typedef(&mut self, name: VarSym) -> Option<TypeSym> {
        self.types.get(&name).cloned()
    }

    /*
    /// Looks up a typedef and if it is `Named` try to keep looking
    /// it up until we find the actual concrete type.  Returns None
    /// if it can't.
    ///
    /// TODO: This is weird, currently necessary for structs though.
    fn follow_typedef(&mut self, name: VarSym) -> Option<TypeSym> {
        match self.types.get(&name) {
            Some(tsym) => match &*INT.fetch_type(*tsym) {
                &TypeDef::Named(vsym) => self.follow_typedef(vsym),
                _other => Some(*tsym),
            },
            None => None,
        }
    }

    /// Take a typesym and look it up to a concrete type definition of some kind.
    /// Recursively follows named types, which might or might not be a good idea...
    ///
    /// TODO: This is weird, currently necessary for structs though.
    fn resolve_typedef(&mut self, t: TypeSym) -> Option<std::sync::Arc<TypeDef>> {
        let tdef = INT.fetch_type(t);
        match &*tdef {
            TypeDef::Named(vsym) => self.follow_typedef(*vsym).map(|sym| INT.fetch_type(sym)),
            _other => Some(tdef),
        }
    }
    */

    /// Returns Ok if the type exists, or a TypeError of the appropriate
    /// kind if it does not.
    fn type_exists(&mut self, tsym: TypeSym) -> Result<(), TypeError> {
        match &*INT.fetch_type(tsym) {
            /*
             * TODO
            TypeDef::Named(name) => {
                if self.types.get(name).is_some() {
                    Ok(())
                } else {
                    Err(TypeError::UnknownType(*name))
                }
            }
            */
            // Primitive type
            _ => Ok(()),
        }
    }

    /// Add a variable to the top level of the scope.
    /// Shadows the old var if it already exists in that scope.
    fn add_var(&mut self, name: VarSym, typedef: TypeSym, mutable: bool) {
        let binding = VarBinding {
            name,
            typename: typedef,
            mutable,
        };
        self.vars.insert(name, binding);
    }

    /// Get the type of the given variable, or an error
    fn get_var(&self, name: VarSym) -> Result<TypeSym, TypeError> {
        Ok(self.get_binding(name)?.typename)
    }

    /// Get (a clone of) the binding of the given variable, or an error
    fn get_binding(&self, name: VarSym) -> Result<VarBinding, TypeError> {
        if let Some(binding) = self.vars.get(&name) {
            return Ok(binding.clone());
        }
        Err(TypeError::UnknownVar(name))
    }

    fn binding_exists(&self, name: VarSym) -> bool {
        self.get_binding(name).is_ok()
    }
}

fn typecheck_decl(
    symtbl: &mut ISymtbl,
    decl: hir::Decl<()>,
) -> Result<hir::Decl<TypeSym>, TypeError> {
    let ctx = &mut CheckState::default();
    match decl {
        hir::Decl::Function {
            name,
            signature,
            body,
        } => {
            // Push scope, typecheck and add params to symbol table
            let symtbl = &mut symtbl.clone();
            for (pname, ptype) in signature.params.iter() {
                symtbl.add_var(*pname, *ptype, false);
            }
            let mut last_type = INT.unit();
            for expr in body {
                let inferred_t = ctx.infer(&expr)?;
                last_type = ctx.ctx.subst(inferred_t);
            }
            return Ok(last_type);
            // Below here is old

            /*

            // Push scope, typecheck and add params to symbol table
            let symtbl = &mut symtbl.clone();
            for (pname, ptype) in signature.params.iter() {
                symtbl.add_var(*pname, *ptype, false);
            }

            // This is squirrelly; basically, we want to return unit
            // if the function has no body, otherwise return the
            // type of the last expression.
            //
            // If there's a return expr, we just return the Never type
            // for it and it all shakes out to work.
            let typechecked_exprs = typecheck_exprs(symtbl, body, Some(signature.rettype))?;
            // Ok, so we *also* need to walk through all the expressions
            // and look for any "return" exprs (or later `?`/`try` exprs
            // also) and see make sure the return types match.
            let last_expr_type = last_type_of(&typechecked_exprs);
            if let Some(t) = infer_type(last_expr_type, signature.rettype) {
                let inferred_exprs = reify_last_types(last_expr_type, t, typechecked_exprs);
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
                */
        }
        hir::Decl::Const {
            name,
            typename,
            init,
        } => {
            // Make sure the const's type exists
            symtbl.type_exists(typename)?;
            let symtbl = &mut symtbl.clone();
            let inferred_t = ctx.infer(&init)?;
            let last_type = ctx.ctx.subst(inferred_t);
            assert!(INT.fetch_type(inferred_t).is_mono());
            Ok(hir::Decl::Const {
                name,
                typename,
                init: todo!("Consistantify TypedExpr nonsense"),
            })
        }
        // Ok, we are declaring a new type.  We need to make sure that the typedecl
        // it's using is real.  We've already checked to make sure it's not a duplicate.
        hir::Decl::TypeDef { name, typedecl } => {
            // Make sure the body of the typedef is a real type.
            symtbl.type_exists(typedecl)?;
            Ok(hir::Decl::TypeDef { name, typedecl })
        }
        // Don't need to do anything here since we generate these in the lowering
        // step and have already verified no names clash.
        hir::Decl::Constructor { name, signature } => {
            Ok(hir::Decl::Constructor { name, signature })
        }
    }
}

/// Scan through all decl's and add any bindings to the symbol table,
/// so we don't need to do anything with forward references.
fn predeclare_decl(symtbl: &mut ISymtbl, decl: &hir::Decl<()>) {
    match decl {
        hir::Decl::Function {
            name, signature, ..
        } => {
            if symtbl.binding_exists(*name) {
                panic!("Tried to redeclare function {}!", INT.fetch(*name));
            }
            // Add function to global scope
            let type_params = signature.params.iter().map(|(_name, t)| *t).collect();
            let function_type = INT.intern_type(&TypeDef::Lambda(type_params, signature.rettype));
            symtbl.add_var(*name, function_type, false);
        }
        hir::Decl::Const { name, typename, .. } => {
            if symtbl.binding_exists(*name) {
                panic!("Tried to redeclare const {}!", INT.fetch(*name));
            }
            symtbl.add_var(*name, *typename, false);
        }
        hir::Decl::TypeDef { name, typedecl } => {
            // Gotta make sure there's no duplicate declarations
            // This kinda has to happen here rather than in typeck()
            if symtbl.get_typedef(*name).is_some() {
                panic!("Tried to redeclare type {}!", INT.fetch(*name));
            }
            symtbl.add_type(*name, *typedecl);
        }
        hir::Decl::Constructor { name, signature } => {
            {
                if symtbl.get_var(*name).is_ok() {
                    panic!(
                        "Aieeee, redeclaration of function/type constructor named {}",
                        INT.fetch(*name)
                    );
                }
                let type_params = signature.params.iter().map(|(_name, t)| *t).collect();
                let function_type =
                    INT.intern_type(&TypeDef::Lambda(type_params, signature.rettype));
                symtbl.add_var(*name, function_type, false);
            }

            // Also we need to add a deconstructor function.  This is kinda a placeholder, but,
            // should work for now.
            {
                let deconstruct_name = INT.intern(format!("{}_unwrap", INT.fetch(*name)));
                if symtbl.get_var(deconstruct_name).is_ok() {
                    panic!(
                        "Aieeee, redeclaration of function/type destructure named {}",
                        INT.fetch(deconstruct_name)
                    );
                }
                let type_params = vec![signature.rettype];
                let rettype = signature.params[0].1;
                let function_type = INT.intern_type(&TypeDef::Lambda(type_params, rettype));
                symtbl.add_var(deconstruct_name, function_type, false);
            }
        }
    }
}

pub fn typecheck(ir: hir::Ir<()>) -> Result<hir::Ir<TypeSym>, TypeError> {
    let symtbl = &mut ISymtbl::new_with_defaults();
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
/*

/// Does t1 equal t2?
///
/// Currently we have no covariance or contravariance, so this is pretty simple.
/// Currently it's just, if the symbols match, the types match.
/// The symbols matching by definition means the structures match.
///
/// If we want some type, and got Never, then this is always true
/// because we never hit the expression that expected the `wanted` type
fn type_matches(wanted: TypeSym, got: TypeSym) -> bool {
    //println!("Testing {:?} against {:?}", wanted, got);
    if got == INT.never() {
        true
    } else {
        wanted == got
    }
}

/// Goes from types that may be unknown to a type that is
/// totally real, and fits the constraints of both input
/// types, and returns the real type.
///
/// Returns `None` if there is not enough info to decide,
/// or if the types are not compatible with each other.
fn infer_type(t1: TypeSym, t2: TypeSym) -> Option<TypeSym> {
    let t1_def = &*INT.fetch_type(t1);
    let t2_def = &*INT.fetch_type(t2);
    match (t1_def, t2_def) {
        (TypeDef::UnknownInt, TypeDef::SInt(_)) => Some(t2),
        (TypeDef::SInt(_), TypeDef::UnknownInt) => Some(t1),
        // Anything related to a never type becomes a never type,
        // because by definition control flow never actually gets
        // to that expression.
        //
        // ...hang on, is that correct?  Think of an `if/else` where
        // one of the arms returns Never.  One arm never returns, the
        // other will return a thing.  So the only sensible type for
        // the expression is the type of the non-Never arm.
        //
        // heckin covariance and contravariance, I hate you.
        //
        // What about Never and Never?  Is that valid, or not?  I THINK it is
        (TypeDef::Never, _) => Some(t2),
        (_, TypeDef::Never) => Some(t1),
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
        // TODO: Structs
        /*
        (TypeDef::Struct { name: n1, .. }, TypeDef::Struct { name: n2, .. }) if n1 == n2 => {
            Some(t1)
        }
        // TODO: This is kinda fucky and I hate it; if a type is named we need to resolve
        // it to a real struct type of some kind
        (TypeDef::Named(n1), TypeDef::Struct { name: n2, .. }) if n1 == n2 => Some(t2),
        (TypeDef::Struct { name: n1, .. }, TypeDef::Named(n2)) if n1 == n2 => Some(t1),
        */
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
        expr
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
    assert!(!exprs.is_empty());
    // We own the vec, so we can do whatever we want to it!  Muahahahahaha!
    let mut exprs = exprs;
    let last_expr = exprs.pop().unwrap();
    exprs.push(reify_types(from, to, last_expr));
    exprs
}

/// Try to actually typecheck the given HIR, and return HIR with resolved types.
pub fn typecheck(ir: hir::Ir<()>) -> Result<hir::Ir<TypeSym>, TypeError> {
    let symtbl = &mut ISymtbl::new_with_defaults();
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
fn predeclare_decl(symtbl: &mut ISymtbl, decl: &hir::Decl<()>) {
    match decl {
        hir::Decl::Function {
            name, signature, ..
        } => {
            if symtbl.binding_exists(*name) {
                panic!("Tried to redeclare function {}!", INT.fetch(*name));
            }
            // Add function to global scope
            let type_params = signature.params.iter().map(|(_name, t)| *t).collect();
            let function_type = INT.intern_type(&TypeDef::Lambda(type_params, signature.rettype));
            symtbl.add_var(*name, function_type, false);
        }
        hir::Decl::Const { name, typename, .. } => {
            if symtbl.binding_exists(*name) {
                panic!("Tried to redeclare const {}!", INT.fetch(*name));
            }
            symtbl.add_var(*name, *typename, false);
        }
        hir::Decl::TypeDef { name, typedecl } => {
            // Gotta make sure there's no duplicate declarations
            // This kinda has to happen here rather than in typeck()
            if symtbl.get_typedef(*name).is_some() {
                panic!("Tried to redeclare type {}!", INT.fetch(*name));
            }
            symtbl.add_type(*name, *typedecl);
        }
        hir::Decl::Constructor { name, signature } => {
            {
                if symtbl.get_var(*name).is_ok() {
                    panic!(
                        "Aieeee, redeclaration of function/type constructor named {}",
                        INT.fetch(*name)
                    );
                }
                let type_params = signature.params.iter().map(|(_name, t)| *t).collect();
                let function_type =
                    INT.intern_type(&TypeDef::Lambda(type_params, signature.rettype));
                symtbl.add_var(*name, function_type, false);
            }

            // Also we need to add a deconstructor function.  This is kinda a placeholder, but,
            // should work for now.
            {
                let deconstruct_name = INT.intern(format!("{}_unwrap", INT.fetch(*name)));
                if symtbl.get_var(deconstruct_name).is_ok() {
                    panic!(
                        "Aieeee, redeclaration of function/type destructure named {}",
                        INT.fetch(deconstruct_name)
                    );
                }
                let type_params = vec![signature.rettype];
                let rettype = signature.params[0].1;
                let function_type = INT.intern_type(&TypeDef::Lambda(type_params, rettype));
                symtbl.add_var(deconstruct_name, function_type, false);
            }
        }
    }
}

/// Typechecks a single decl
fn typecheck_decl(
    symtbl: &mut ISymtbl,
    decl: hir::Decl<()>,
) -> Result<hir::Decl<TypeSym>, TypeError> {
    match decl {
        hir::Decl::Function {
            name,
            signature,
            body,
        } => {
            // Push scope, typecheck and add params to symbol table
            let symtbl = &mut symtbl.clone();
            for (pname, ptype) in signature.params.iter() {
                symtbl.add_var(*pname, *ptype, false);
            }

            // This is squirrelly; basically, we want to return unit
            // if the function has no body, otherwise return the
            // type of the last expression.
            //
            // If there's a return expr, we just return the Never type
            // for it and it all shakes out to work.
            let typechecked_exprs = typecheck_exprs(symtbl, body, Some(signature.rettype))?;
            // Ok, so we *also* need to walk through all the expressions
            // and look for any "return" exprs (or later `?`/`try` exprs
            // also) and see make sure the return types match.
            let last_expr_type = last_type_of(&typechecked_exprs);
            if let Some(t) = infer_type(last_expr_type, signature.rettype) {
                let inferred_exprs = reify_last_types(last_expr_type, t, typechecked_exprs);
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
        } => {
            // Make sure the const's type exists
            symtbl.type_exists(typename)?;
            let symtbl = &mut symtbl.clone();
            Ok(hir::Decl::Const {
                name,
                typename,
                init: typecheck_expr(symtbl, init, None)?,
            })
        }
        // Ok, we are declaring a new type.  We need to make sure that the typedecl
        // it's using is real.  We've already checked to make sure it's not a duplicate.
        hir::Decl::TypeDef { name, typedecl } => {
            // Make sure the body of the typedef is a real type.
            symtbl.type_exists(typedecl)?;
            Ok(hir::Decl::TypeDef { name, typedecl })
        }
        // Don't need to do anything here since we generate these in the lowering
        // step and have already verified no names clash.
        hir::Decl::Constructor { name, signature } => {
            Ok(hir::Decl::Constructor { name, signature })
        }
    }
}

/// Typecheck a vec of expr's and returns them, with type annotations
/// attached.
fn typecheck_exprs(
    symtbl: &mut ISymtbl,
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
    symtbl: &mut ISymtbl,
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
            Ok(hir::TypedExpr {
                e: Lit { val },
                t,
                s: symtbl.clone(),
            })
        }
        EnumLit { val, ty } => {
            // Verify that the type given is actually an enum
            let tdef = INT.fetch_type(ty);
            match &*tdef {
                // Verify that the variant given exists in the enum
                TypeDef::Enum { variants } => {
                    if let Some((enum_sym, enum_val)) = variants
                        .iter()
                        .find(|(enum_sym, _enum_val)| *enum_sym == val)
                    {
                        /*
                        // We do some lowering here just 'cause it's easy to?
                        // As noted elsewhere, our enums are always i32 for now
                        let integer_val = hir::Literal::SizedInteger {
                            vl: (*enum_val) as i128,
                            bytes: 4,
                        };
                        Ok(hir::TypedExpr {
                            e: Lit { val: integer_val },
                            t: ty,
                            s: symtbl.clone(),
                        })
                        */
                        Ok(hir::TypedExpr {
                            e: EnumLit { val, ty },
                            t: ty,
                            s: symtbl.clone(),
                        })
                    } else {
                        Err(TypeError::EnumVariant {
                            expected: variants.into_iter().map(|(a, _)| *a).collect(),
                            got: val,
                        })
                    }
                }
                TypeDef::Named(vsym) => todo!("Aha"),
                other => {
                    dbg!(INT.intern_type(other));
                    dbg!(ty);
                    Err(TypeError::TypeMismatch {
                        // TODO: More information
                        expr_name: "enum literal".into(),
                        got: INT.intern_type(other),
                        expected: ty,
                    })
                }
            }
        }

        Var { name } => {
            let t = symtbl.get_var(name)?;
            Ok(hir::TypedExpr {
                e: Var { name },
                t: t,
                s: symtbl.clone(),
            })
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
                // We must also figure out the output type, though.  It depends on the binop!
                // Sometimes it depends on the input types (number + number => number, number + i32
                // => i32), and sometimes it doesn't (bool and bool => bool, no guesswork involved)
                Ok(hir::TypedExpr {
                    e: BinOp {
                        op,
                        lhs: Box::new(real_lhs),
                        rhs: Box::new(real_rhs),
                    },
                    t: op.output_type(t),
                    s: symtbl.clone(),
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
                    t: op.output_type(t),
                    s: symtbl.clone(),
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
            let mut symtbl = symtbl.clone();
            let b = typecheck_exprs(&mut symtbl, body, function_rettype)?;
            let t = last_type_of(&b);
            Ok(hir::TypedExpr {
                e: Block { body: b },
                t,
                s: symtbl,
            })
        }
        Let {
            varname,
            typename,
            init,
            mutable,
        } => {
            let init_expr = typecheck_expr(symtbl, *init, function_rettype)?;
            // If the init expression can match with the declared let type, life is good
            if let Some(t) = infer_type(typename, init_expr.t) {
                let new_init = reify_types(init_expr.t, t, init_expr);
                // Add var to symbol table, proceed
                symtbl.add_var(varname, t, mutable);
                Ok(hir::TypedExpr {
                    e: Let {
                        varname,
                        typename: t,
                        init: Box::new(new_init),
                        mutable,
                    },
                    t: unittype,
                    s: symtbl.clone(),
                })
            } else {
                dbg!(&init_expr.t, &typename);
                dbg!(INT.fetch_type(init_expr.t), INT.fetch_type(typename));

                Err(TypeError::LetType {
                    name: varname,
                    got: init_expr.t,
                    expected: typename,
                })
            }
        }
        If { cases } => {
            // This is kinda oogly, what we really need to do is find the ret type
            // of *every* arm, unify them (aka find a type they all agree on), and then
            // reify them all to have that type.
            //
            // ...hang on, that doesn't sound TOO hard.
            let mut assumed_type = INT.never();
            let mut new_cases = vec![];
            for (cond, body) in cases.into_iter() {
                // First we just check whether all the conds return bools
                // ...hmm, typecheck_expr() consumes `cond`, which is kinda awkward.
                let cond_expr = typecheck_expr(symtbl, cond, function_rettype)?;
                if !type_matches(cond_expr.t, booltype) {
                    return Err(TypeError::Cond { got: cond_expr.t });
                }

                // Now we get the type of the body...
                let mut symtbl = symtbl.clone();
                let body_exprs = typecheck_exprs(&mut symtbl, body, function_rettype)?;
                let body_type = last_type_of(&body_exprs);
                // Does it have a type that can match with our other types?
                if let Some(t) = infer_type(body_type, assumed_type) {
                    assumed_type = t;
                    let new_body = reify_last_types(body_type, t, body_exprs);
                    new_cases.push((cond_expr, new_body));
                } else {
                    return Err(TypeError::IfType {
                        got: body_type,
                        expected: assumed_type,
                    });
                }
            }
            Ok(hir::TypedExpr {
                e: If { cases: new_cases },
                t: assumed_type,
                s: symtbl.clone(),
            })
        }
        Loop { body } => {
            let mut symtbl = symtbl.clone();
            let b = typecheck_exprs(&mut symtbl, body, function_rettype)?;
            let t = last_type_of(&b);
            Ok(hir::TypedExpr {
                e: Loop { body: b },
                t,
                s: symtbl,
            })
        }
        // TODO: Inference?
        Lambda { signature, body } => {
            let mut symtbl = symtbl.clone();
            // add params to symbol table
            for (paramname, paramtype) in signature.params.iter() {
                symtbl.add_var(*paramname, *paramtype, false);
            }
            let body_expr = typecheck_exprs(&mut symtbl, body, Some(signature.rettype))?;
            let bodytype = last_type_of(&body_expr);
            if let Some(t) = infer_type(bodytype, signature.rettype) {
                let new_body = reify_last_types(bodytype, t, body_expr);
                let lambdatype = signature.to_type();
                Ok(hir::TypedExpr {
                    e: Lambda {
                        signature,
                        body: new_body,
                    },
                    t: lambdatype,
                    s: symtbl,
                })
            } else {
                let function_name = INT.intern("lambda");
                Err(TypeError::Return {
                    fname: function_name,
                    got: bodytype,
                    expected: signature.rettype,
                })
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
                        s: symtbl.clone(),
                    })
                }
                // Something was called but it wasn't a function.  Wild.
                _other => Err(TypeError::Call { got: f.t }),
            }
        }
        Break => Ok(hir::TypedExpr {
            e: Break,
            t: INT.never(),
            s: expr.s.clone(),
        }),
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
                        s: symtbl.clone(),
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
        TupleCtor { body } => {
            // Inference for this is done by `infer_types()` on the member types.
            let body_exprs = typecheck_exprs(symtbl, body, function_rettype)?;
            let body_typesyms = body_exprs.iter().map(|te| te.t).collect();
            let body_type = TypeDef::Tuple(body_typesyms);
            Ok(hir::TypedExpr {
                t: INT.intern_type(&body_type),
                e: TupleCtor { body: body_exprs },
                s: symtbl.clone(),
            })
        }
        StructCtor { body, types } => {
            let mut symtbl = symtbl.clone();
            // Add all the types declared in the struct to the local type scope
            /*
            for (nm, ty) in &types {
                dbg!("Adding type", nm, ty);
                symtbl.add_type(*nm, *ty);
            }
            */
            // Typecheck the expressions in the body of the structure.
            let body_exprs: Vec<(VarSym, hir::TypedExpr<TypeSym>)> = body
                .iter()
                .map(|(nm, vl)| {
                    let expr = typecheck_expr(&mut symtbl, vl.clone(), function_rettype)?;
                    Ok((*nm, expr))
                })
                .collect::<Result<_, _>>()?;

            // Construct a struct type with the types of the body in it.
            let struct_type = INT.intern_type(&TypeDef::Struct {
                fields: body_exprs
                    .iter()
                    .map(|(nm, typed_expr)| (*nm, typed_expr.t))
                    .collect(),
                typefields: types.iter().map(|(nm, _ty)| *nm).collect(),
            });

            Ok(hir::TypedExpr {
                t: struct_type,
                e: StructCtor {
                    body: body_exprs,
                    types,
                },
                s: symtbl,
            })
        }
        // TODO: Inference???
        // Not sure we need it, any integer type should be fine for tuple lookups...
        TupleRef { expr: e, elt } => {
            let body_expr = typecheck_expr(symtbl, *e, function_rettype)?;
            let expr_typedef = INT.fetch_type(body_expr.t);
            if let TypeDef::Tuple(typesyms) = &*expr_typedef {
                // TODO
                // ...what do we have to do here?
                assert!(elt < typesyms.len());
                Ok(hir::TypedExpr {
                    t: typesyms[elt],
                    e: TupleRef {
                        expr: Box::new(body_expr),
                        elt,
                    },
                    s: symtbl.clone(),
                })
            } else {
                Err(TypeError::TupleRef { got: body_expr.t })
            }
        }
        StructRef { expr: e, elt } => {
            let body_expr = typecheck_expr(symtbl, *e, function_rettype)?;
            //let expr_typedef = INT.fetch_type(body_expr.t);
            let expr_typedef = symtbl
                .resolve_typedef(body_expr.t)
                .expect("Type does not exist?");
            match &*expr_typedef {
                TypeDef::Struct { fields, .. } => {
                    if let Some((_name, vl)) = fields.iter().find(|(nm, _)| **nm == elt) {
                        // The referenced field exists in the struct type
                        Ok(hir::TypedExpr {
                            t: *vl,
                            e: StructRef {
                                expr: Box::new(body_expr),
                                elt,
                            },
                            s: symtbl.clone(),
                        })
                    } else {
                        Err(TypeError::StructRef {
                            fieldname: elt,
                            got: body_expr.t,
                        })
                    }
                }
                // We should never get a named type out of this.
                _other => Err(TypeError::TypeMismatch {
                    expr_name: "struct ref".into(),
                    got: body_expr.t,
                    expected: INT.unit(),
                }),
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
                    s: symtbl.clone(),
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
fn is_mutable_lvalue<T>(symtbl: &ISymtbl, expr: &hir::Expr<T>) -> Result<bool, TypeError> {
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
        let ast = Parser::new("unittest", src).parse();
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
        let t = &mut ISymtbl::new_with_defaults();

        // Make sure we can get a value
        assert!(t.get_var(t_foo).is_err());
        assert!(t.get_var(t_bar).is_err());
        t.add_var(t_foo, t_i32, false);
        assert_eq!(t.get_var(t_foo).unwrap(), t_i32);
        assert!(t.get_var(t_bar).is_err());

        // Push scope, verify behavior is the same.
        {
            let mut t = t.clone();
            assert_eq!(t.get_var(t_foo).unwrap(), t_i32);
            assert!(t.get_var(t_bar).is_err());
            // Add var, make sure we can get it
            t.add_var(t_bar, t_bool, false);
            assert_eq!(t.get_var(t_foo).unwrap(), t_i32);
            assert_eq!(t.get_var(t_bar).unwrap(), t_bool);
        }
        // Pop scope, make sure bar is gone.
        assert_eq!(t.get_var(t_foo).unwrap(), t_i32);
        assert!(t.get_var(t_bar).is_err());

        // Make sure we can shadow a value
        t.add_var(t_foo, t_i32, false);
        t.add_var(t_foo, t_bool, false);
        assert_eq!(t.get_var(t_foo).unwrap(), t_bool);
        // TODO: Check to make sure the shadowed var is still there.
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
        let tbl = &mut ISymtbl::new_with_defaults();

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
        let tbl = &mut ISymtbl::new_with_defaults();

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
        let tbl = &mut ISymtbl::new_with_defaults();
        let t_i32 = INT.i32();
        let t_unit = INT.unit();
        let fooname = INT.intern("foo");

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
        let tbl = &mut ISymtbl::new_with_defaults();
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
        let tbl = &mut ISymtbl::new_with_defaults();
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
        let x: I8 = 8_I8
        let x: I16 = 9_I16
        let x: I32 = 10_I32
        let y: I64 = 11_I64
        let y: I128 = 12_I128
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
        let mut y: I64 = 11_I64
        y = x
end"#;
        fail_typecheck!(src, TypeError::TypeMismatch { .. });
    }

    #[test]
    fn test_bad_integer_math() {
        let src = r#"fn foo() =
        let x: I32 = 10
        let mut y: I64 = 11_I64
        y + x
end"#;
        fail_typecheck!(src, TypeError::BopType { .. });
    }
}
*/
