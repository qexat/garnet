//! Typechecking and other semantic checking.
//! Operates on the HIR.
//!
//! This is mostly based on
//! https://ahnfelt.medium.com/type-inference-by-example-793d83f98382
//!
//! We go through and generate a type variable for each expression,
//! then unify to solve them, probably in a more-or-less-bidirectional
//! fashion.

use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use crate::hir::{self, Expr};
use crate::{TypeDef, TypeSym, VarSym, INT};

#[derive(Debug, Clone)]
pub enum TypeError {
    UnknownVar(VarSym),
    AlreadyDefined(VarSym),
    TypeMismatch {
        expr_name: Cow<'static, str>,
        got: TypeSym,
        expected: TypeSym,
    },
    AmbiguousType {
        expr_name: Cow<'static, str>,
    },
    /*
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
    Mutability {
        expr_name: Cow<'static, str>,
    },
    */
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
            TypeError::AlreadyDefined(sym) => format!(
                "Type, var, const, or function already defined: {}",
                INT.fetch(*sym)
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
            TypeError::AmbiguousType { expr_name } => {
                format!("Ambiguous/unknown type for expression '{}'", expr_name)
            } /*
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
              TypeError::Mutability { expr_name } => {
                  format!("Mutability mismatch in '{}' expresssion", expr_name)
              }
              */
        }
    }
}

/// A variable binding
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarBinding {
    pub name: VarSym,
    pub typevar: TypeVar,
    pub mutable: bool,
}

/// A generated type variable.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct TypeVar(usize);

/// What a type var may be equal to.
///
/// I tend to think of these as "facts we know about our types", but apparently "constraints" is a
/// more proper term.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Constraint {
    /// var 1 == var 2
    TypeVar(TypeVar),
    /// var 1 == some known type
    TypeSym(TypeSym),
}

/// A unique var ID used to unambiguously identify variables regardless of scope.
///
/// They need to be unique at least per compilation unit, but since we don't really do compilation
/// units other than a single file right now, that's fine.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
struct UniqueVar(usize);

/// Immutable symbol table.  We just keep one of these associated with every expression,
/// which describes that expression's scope.  Since it's immutable, cloning and modifying
/// it only changes the parts that are necessary.
///
/// This means that we always can look at any expression and see its entire scope, and
/// keeping track of scope pushes/pops is basically implicit since there's no mutable state
/// involved.
///
/// TODO NEXT: Might need to make this UniqueVar -> VarBinding, and also have a
/// reverse lookup table too?
#[derive(Clone, Debug, PartialEq)]
pub struct Symtbl {
    vars: im::HashMap<UniqueVar, VarBinding>,
    typenames: im::HashMap<VarSym, TypeSym>,
}

impl Default for Symtbl {
    fn default() -> Self {
        Symtbl {
            vars: Default::default(),
            typenames: Default::default(),
        }
    }
}

impl Symtbl {
    fn add_type(&mut self, name: VarSym, typedef: TypeSym) {
        self.typenames.insert(name, typedef);
    }

    fn get_type(&mut self, name: VarSym) -> Option<TypeSym> {
        self.typenames.get(&name).cloned()
    }

    /*
    fn add_type_var(&mut self, name: VarSym, id: TypeId) {
        self.type_vars.insert(name, id);
    }

    fn get_type_var(&mut self, name: VarSym) -> Option<TypeId> {
        self.type_vars.get(&name).cloned()
    }
    */

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

    /*
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
    */

    /// Add a variable to the top level of the scope.
    /// Shadows the old var if it already exists in that scope.
    fn add_binding(&mut self, name: UniqueVar, binding: VarBinding) {
        self.vars.insert(name, binding);
    }

    /// Get the type of the given variable, or an error.
    /// The type will be a type variable that must then be unified, I guess.
    fn get_var_type(&self, name: UniqueVar) -> Result<TypeVar, TypeError> {
        Ok(self.get_binding(name)?.typevar)
    }

    /// Get (a clone of) the binding of the given variable, or an error
    fn get_binding(&self, name: UniqueVar) -> Result<VarBinding, TypeError> {
        if let Some(binding) = self.vars.get(&name) {
            return Ok(binding.clone());
        }
        //Err(TypeError::UnknownVar(name))
        panic!("Unique var is unbound: {:?}", name);
    }

    fn binding_exists(&self, name: UniqueVar) -> bool {
        self.get_binding(name).is_ok()
    }
}

/// Tracks variables and names in scope.
/// So the way we do this is is a little weird 'cause we can't
/// forget scope info.  We needd to know stuff about specific variables
/// that may be used to solve type inference stuff above their scope
/// or even before they're declared.
///
/// SO.  This is the usual scope-as-stack-of-vars type thing.
/// We use this to do our normal scope cehcking, and also create a
/// mapping from exprid to a unique variable id for every variable
/// use and declaration.
///
/// That way we can have a list of all the variables in each
/// function regardless of name, and just totally ignore scoping
/// from then on.
///
/// Uses internal mutability and returns a guard value to manage push/pops
/// because otherwise life is kinda awful.  Just do `let _guard = scope.push()`
/// and it will pop automatically when the _guard value is dropped.
#[derive(Clone, Debug)]
struct Scope {
    vars: Rc<RefCell<Vec<HashMap<VarSym, UniqueVar>>>>,
}

impl Default for Scope {
    /// We start with an empty toplevel scope existing.
    fn default() -> Scope {
        Scope {
            vars: Rc::new(RefCell::new(vec![HashMap::new()])),
        }
    }
}

pub struct ScopeGuard {
    scope: Scope,
}

impl Drop for ScopeGuard {
    fn drop(&mut self) {
        self.scope.pop();
    }
}

impl Scope {
    fn push(&self) -> ScopeGuard {
        self.vars.borrow_mut().push(HashMap::new());
        ScopeGuard {
            scope: self.clone(),
        }
    }

    fn pop(&self) {
        self.vars.borrow_mut().pop().expect("Scope stack underflow");
    }

    fn add_var(&self, var: VarSym, unique: UniqueVar) {
        self.vars
            .borrow_mut()
            .last_mut()
            .expect("Scope stack underflow")
            .insert(var, unique);
    }

    /// Checks whether the var exists in the currently alive scopes
    fn get_var_with_scope(&self, var: VarSym) -> Option<UniqueVar> {
        for scope in self.vars.borrow().iter().rev() {
            let v = scope.get(&var);
            if v.is_some() {
                return v.cloned();
            }
        }
        return None;
    }

    fn var_is_bound(&self, var: VarSym) -> bool {
        self.get_var_with_scope(var).is_some()
    }
}

/// Top level type checking context struct.
#[derive(Clone, Debug)]
pub struct Tck {
    /// A mapping containing a type variable for each expression.
    /// Then we update the type variables with what we know about them
    /// as we typecheck/infer.
    exprtypes: HashMap<hir::Eid, TypeVar>,

    /// What info we know about our typevars.
    /// We may have multiple non-identical constraints for each type var,
    /// so we keep a set of them.
    constraints: HashMap<TypeVar, Constraint>,

    /// Symbol table.  All known vars and types.
    symtbl: Symtbl,

    /// The current scope stack.
    /// This is its own type 'cause it's a little clearer
    /// where we're manipulating scope vs. all vars.
    ///
    /// So the Scope gives us varsym -> uniquevar, and
    /// symtbl gives us uniquevar -> binding
    scope: Scope,

    /// Index of the next type var gensym.
    next_typevar: usize,

    /// Index of the next unique var symbol.  These could be
    /// unique per function or globally, for now they're globally
    /// because there's no real reason not to be.
    next_uniquevar: usize,
}

impl Default for Tck {
    /// Create a new Tck with whatever default types etc we need.
    fn default() -> Self {
        let mut x = Tck {
            exprtypes: Default::default(),
            constraints: Default::default(),
            symtbl: Default::default(),
            scope: Default::default(),
            next_typevar: 0,
            next_uniquevar: 0,
        };
        // We add a built-in function for printing, currently.
        {
            let name = INT.intern("__println");
            let typesym = INT.intern_type(&TypeDef::Lambda {
                generics: vec![],
                params: vec![INT.i32()],
                rettype: INT.unit(),
            });
            x.create_var(name, Some(typesym), false);
        }
        {
            let name = INT.intern("__println_bool");
            let typesym = INT.intern_type(&TypeDef::Lambda {
                generics: vec![],
                params: vec![INT.bool()],
                rettype: INT.unit(),
            });
            x.create_var(name, Some(typesym), false);
        }
        {
            let name = INT.intern("__println_i64");
            let typesym = INT.intern_type(&TypeDef::Lambda {
                generics: vec![],
                params: vec![INT.i64()],
                rettype: INT.unit(),
            });
            x.create_var(name, Some(typesym), false);
        }
        {
            let name = INT.intern("__println_i16");
            let typesym = INT.intern_type(&TypeDef::Lambda {
                generics: vec![],
                params: vec![INT.i16()],
                rettype: INT.unit(),
            });
            x.create_var(name, Some(typesym), false);
        }
        x
    }
}

impl Tck {
    /// Takes a typevar and follows whatever constraints
    /// we have on it until it hits a real type.  Returns
    /// None if it can't do that.
    ///
    /// Panics if the typevars form a loop, which should never happen.
    pub fn follow_typevar(&self, tv: TypeVar) -> Option<TypeSym> {
        fn helper(tck: &Tck, tv: TypeVar, original_typevar: TypeVar) -> Option<TypeSym> {
            let constraint = tck.constraints.get(&tv)?;
            match constraint {
                Constraint::TypeVar(tv2) => {
                    assert!(*tv2 != original_typevar);
                    helper(tck, *tv2, original_typevar)
                }
                Constraint::TypeSym(ts) => Some(*ts),
            }
        }

        helper(self, tv, tv)
    }

    /// Create a new type var and associate it with the given expression ID
    pub fn create_exprtype(&mut self, eid: hir::Eid) -> TypeVar {
        let tv = self.new_typevar();
        let old = self.exprtypes.insert(eid, tv);
        if old.is_some() {
            panic!("Tried to bind Eid twice, how unique is it?!");
        }
        tv
    }

    fn new_typevar(&mut self) -> TypeVar {
        let tv = TypeVar(self.next_typevar);
        self.next_typevar += 1;
        tv
    }

    fn new_uniquevar(&mut self) -> UniqueVar {
        let uv = UniqueVar(self.next_typevar);
        self.next_uniquevar += 1;
        uv
    }

    /// Try getting the typevar for the given expression.
    /// Returns None if it does not have one.
    pub fn get_typevar_for_expression(&self, expr: &hir::TypedExpr) -> Option<TypeVar> {
        self.exprtypes.get(&expr.id).cloned()
    }

    /// If the var's type is unknown, then we create a type var for it.
    ///
    /// If the var has a type associated with it, then we create a type
    /// var for it and list it as a known fact.
    ///
    /// We return the UniqueVar ID so we can add it to a Scope as well.
    /// We might just want to handle that here, we will see.
    fn create_var(&mut self, name: VarSym, ty: Option<TypeSym>, mutable: bool) -> UniqueVar {
        let uniquevar = self.new_uniquevar();
        let tv = self.new_typevar();
        let binding = VarBinding {
            name,
            typevar: tv,
            mutable,
        };
        // If we know the real type of the variable, save it
        if let Some(t) = ty {
            self.add_constraint(tv, Constraint::TypeSym(t));
        }
        self.symtbl.add_binding(uniquevar, binding);
        uniquevar
    }

    fn add_constraint(&mut self, tv: TypeVar, constraint: Constraint) {
        println!("Adding constraint {:?} to typevar {:?}", constraint, tv);
        use std::collections::hash_map::Entry;
        match self.constraints.entry(tv) {
            Entry::Occupied(o) => {
                panic!(
                    "Duplicate type var {:?} with constraints {:?} and {:?}, should never happen?",
                    tv,
                    o.get(),
                    constraint
                )
            }
            Entry::Vacant(v) => v.insert(constraint),
        };
    }
}

/// Scan through all decl's and add any bindings to the symbol table,
/// so we don't need to do anything with forward references.
fn predeclare_decl(tck: &mut Tck, decl: &hir::Decl) -> Result<(), TypeError> {
    match decl {
        hir::Decl::Function {
            name, signature, ..
        } => {
            // Does var exist in the global scope?
            if tck.scope.var_is_bound(*name) {
                return Err(TypeError::AlreadyDefined(*name));
            }
            // Add function to global scope
            let function_type = signature.to_type();
            tck.create_var(*name, Some(function_type), false);
        }
        hir::Decl::Const { name, typename, .. } => {
            if tck.scope.var_is_bound(*name) {
                return Err(TypeError::AlreadyDefined(*name));
            }
            tck.create_var(*name, Some(*typename), false);
        }
        hir::Decl::TypeDef { name, typedecl } => {
            // Gotta make sure there's no duplicate declarations
            // This kinda has to happen here rather than in typeck()
            if tck.symtbl.get_type(*name).is_some() {
                return Err(TypeError::AlreadyDefined(*name));
            }
            tck.symtbl.add_type(*name, *typedecl);
            todo!("Predeclare typedef");
        }
        hir::Decl::Constructor { name, signature } => {
            {
                if tck.scope.var_is_bound(*name) {
                    return Err(TypeError::AlreadyDefined(*name));
                }
                let function_type = signature.to_type();
                tck.create_var(*name, Some(function_type), false);
            }

            // Also we need to add a deconstructor function.  This is kinda a placeholder, but,
            // should work for now.
            {
                let deconstruct_name = INT.intern(format!("{}_unwrap", INT.fetch(*name)));
                if tck.scope.var_is_bound(deconstruct_name) {
                    return Err(TypeError::AlreadyDefined(*name));
                }
                let type_params = vec![signature.rettype];
                let rettype = signature.params[0].1;
                let function_type = INT.intern_type(&TypeDef::Lambda {
                    generics: vec![],
                    params: type_params,
                    rettype,
                });
                tck.create_var(deconstruct_name, Some(function_type), false);
            }
            todo!("Predeclare constructor");
        }
    }
    Ok(())
}

// TODO: Might have to create an existential var or something instead of iunknown()
// Yeah it should give you a constraint, not a type
fn infer_literal(lit: &hir::Literal) -> Result<TypeSym, TypeError> {
    match lit {
        hir::Literal::Integer(_) => Ok(INT.iunknown()),
        hir::Literal::SizedInteger { vl: _, bytes } => Ok(INT.intern_type(&TypeDef::SInt(*bytes))),
        hir::Literal::Bool(_) => Ok(INT.bool()),
    }
}

/// Called "reconstruct" in zesterer's
/// "Type inference in less than 100 lines of Rust"
///
/// or maybe called "subst" in "Complete & Easy"
///
/// ...we're gonna have to have a TypeDef that has
/// typevars rather than varsym's in it, aren't we.
fn try_solve_type(tck: &mut Tck, tv: TypeVar) -> Option<TypeSym> {
    let tsym = tck.follow_typevar(tv)?;
    let tdef = &*INT.fetch_type(tsym);
    match tdef {
        TypeDef::UnknownInt => Some(tsym),
        TypeDef::SInt(_) => Some(tsym),
        TypeDef::Tuple(items) if items.len() == 0 => Some(tsym),
        other => todo!("Solve {:?}", other),
    }
}

/// Try to set v1 equal to v2
fn try_unify(tck: &mut Tck, v1: TypeVar, v2: TypeVar) -> Result<(), TypeError> {
    let t1 = *tck.constraints.get(&v1).expect("Shouldn't happen?");
    let t2 = *tck.constraints.get(&v2).expect("Shouldn't happen?");
    match (t1, t2) {
        // Follow any references
        (Constraint::TypeVar(t1var), _) => try_unify(tck, t1var, v2),
        (_, Constraint::TypeVar(t2var)) => try_unify(tck, v1, t2var),
        (Constraint::TypeSym(s1), Constraint::TypeSym(s2)) => {
            // Resolve type syms and unify from there
            let tdef1 = &*INT.fetch_type(s1);
            let tdef2 = &*INT.fetch_type(s2);
            match (tdef1, tdef2) {
                // Primitives are easy
                (TypeDef::Bool, TypeDef::Bool) => Ok(()),
                (TypeDef::SInt(i1), TypeDef::SInt(i2)) if i1 == i2 => Ok(()),
                (TypeDef::UnknownInt, TypeDef::UnknownInt) => Ok(()),
                // If we have a known an unknown int, update the unknown
                // one to match the known one
                (TypeDef::UnknownInt, TypeDef::SInt(_i2)) => {
                    tck.constraints.insert(v1, t2);
                    Ok(())
                }
                (TypeDef::SInt(_i1), TypeDef::UnknownInt) => {
                    tck.constraints.insert(v2, t1);
                    Ok(())
                }
                (s1, s2) => panic!(
                    "Type mismatch trying to unify concrete types {:?} and {:?}",
                    s1, s2
                ),
            }
        }
        (a, b) => panic!("Type mismatch trying to unify {:?} and {:?}", a, b),
    }
}

/// The infer part of our vaguely-bidi type checker.
/// It outputs a constraint -- basically the expression
/// given must either resolve to a known type, or an unknown
/// type where it says "I don't know what type this expression is,
/// but I know it must be the same as `TypeVar(foo)`"
fn infer_expr(
    tck: &mut Tck,
    expr: &hir::TypedExpr,
    rettype: TypeVar,
) -> Result<Constraint, TypeError> {
    todo!("infer_expr")
}

/// The "check" step of our vaguely-bidi type inference.  This is where
/// checking an expr starts.
///
/// rettype is the function return type, any non-local exits (`return`,
/// `?`, etc) need to know that type.
fn check_expr(
    tck: &mut Tck,
    expr: &hir::TypedExpr,
    expected: TypeVar,
    rettype: TypeVar,
) -> Result<(), TypeError> {
    //let tdef = &*INT.fetch_type(expected);
    match &expr.e {
        Expr::Lit { val } => {
            let lit_type = infer_literal(val)?;
            let lit_typevar = tck.create_exprtype(expr.id);
            tck.add_constraint(lit_typevar, Constraint::TypeSym(lit_type));
            // Ok so we know that typevar must be the inferred type...
            // So do we just unify every time we infer something?
            try_unify(tck, expected, lit_typevar)?;
            /*
            match &*INT.fetch_type(lit_type) {
                TypeDef::UnknownInt => {
                    // Ok, the type of this expr is now that int type
                    todo!()
                    //tck.set_type(expr.id, t);
                    //Ok(())
                }
                TypeDef::SInt(size2) => {
                    todo!()
                }
                _ => Err(TypeError::TypeMismatch {
                    expr_name: format!("{:?}", &expr.e).into(),
                    got: lit_type,
                    expected: INT.iunknown(),
                }),
            }
            */
        }
        Expr::Let {
            varname,
            typename,
            init,
            mutable,
        } => {
            let uniquevar = tck.create_var(*varname, Some(*typename), *mutable);
            // get the type var for that particular variable
            let var_type = tck
                .symtbl
                .get_var_type(uniquevar)
                .expect("Should never happen");
            check_expr(tck, init, var_type, rettype)?;
            tck.add_constraint(expected, Constraint::TypeSym(INT.unit()));
        }
        e => {
            eprintln!("Checking thing: {:?} ", e);
            let a = infer_expr(tck, expr, rettype)?;
            eprintln!("Type of {:?} is: {:?}", e, a);
            // What is a substitution? It's a ~~miserable pile of secrets~~
            // map from type variables to types.
            todo!()
            //let x = self.type_sub(self.ctx.subst(a), self.ctx.subst(t));
            //eprintln!("Type subbed: {:?}, {:?}", x, self.ctx);
            //x
        }
    }
    Ok(())
}

/// Check multiple expressions, assuming the last one matches
/// `expected` and the others match `unit`
fn check_exprs(
    tck: &mut Tck,
    exprs: &[hir::TypedExpr],
    expected: TypeVar,
    rettype: TypeVar,
) -> Result<(), TypeError> {
    let last_expr_idx = exprs.len();
    // This loop is the sort of thing that feels like there should
    // be some turbo fancy combinator for it and it really isn't
    // worth the trouble.
    // I guess it'd be easier with recursion but eh
    if last_expr_idx > 0 {
        for expr in &exprs[..(last_expr_idx - 1)] {
            // We create a type variable for the expr, then
            // check to collect constraint info on it
            let tv = tck.create_exprtype(expr.id);
            check_expr(tck, expr, tv, rettype)?;
        }
        // Set the type of the last expression to be the expected type
        let last_expr = &exprs[last_expr_idx - 1];
        let tv = tck.create_exprtype(last_expr.id);
        check_expr(tck, last_expr, tv, rettype)?;
        tck.add_constraint(expected, Constraint::TypeVar(tv));
        //try_unify(tck, tv, rettype)?;
    } else {
        // Body is empty, so the return type must be unit
        tck.add_constraint(rettype, Constraint::TypeSym(INT.unit()));
    }
    Ok(())
}

fn typecheck_decl(tck: &mut Tck, decl: &hir::Decl) -> Result<(), TypeError> {
    match decl {
        hir::Decl::Function {
            name,
            signature,
            body,
        } => {
            let _guard = tck.scope.push();
            // Add function params to symbol table and scope
            for (var, ty) in &signature.params {
                let uniquevar = tck.create_var(*var, Some(*ty), false);
                tck.scope.add_var(*var, uniquevar);
            }
            let rettype = tck.new_typevar();
            check_exprs(tck, &body, rettype, rettype)?;
            // Now that we've checked everything in the
            // function, we need to reconstruct and attempt to solve any
            // unknowns, and error if there are any unknowns left over.
            for expr in body {
                let tv = tck
                    .get_typevar_for_expression(expr)
                    .expect(&format!("No typevar for expression {:?}, aieeee", expr));
                if try_solve_type(tck, tv).is_none() {
                    return Err(TypeError::AmbiguousType {
                        expr_name: format!("{:?}", expr).into(),
                    });
                }
            }
            Ok(())
        }
        _ => todo!("Typecheck decl type {:?}", decl),
        /*
        hir::Decl::Const { name, typename, .. } => {
            if tck.symtbl.binding_exists(*name) {
                return Err(TypeError::AlreadyDefined(*name));
            }
            tck.add_var(*name, Some(*typename), false);
        }
        hir::Decl::TypeDef { name, typedecl } => {
            // Gotta make sure there's no duplicate declarations
            // This kinda has to happen here rather than in typeck()
            if tck.symtbl.get_type(*name).is_some() {
                return Err(TypeError::AlreadyDefined(*name));
            }
            tck.symtbl.add_type(*name, *typedecl);
            todo!("Predeclare typedef");
        }
        hir::Decl::Constructor { name, signature } => {
            {
                if tck.symtbl.get_var(*name).is_ok() {
                    return Err(TypeError::AlreadyDefined(*name));
                }
                let function_type = signature.to_type();
                tck.add_var(*name, Some(function_type), false);
            }

            // Also we need to add a deconstructor function.  This is kinda a placeholder, but,
            // should work for now.
            {
                let deconstruct_name = INT.intern(format!("{}_unwrap", INT.fetch(*name)));
                if tck.symtbl.get_var(deconstruct_name).is_ok() {
                    return Err(TypeError::AlreadyDefined(*name));
                }
                let type_params = vec![signature.rettype];
                let rettype = signature.params[0].1;
                let function_type = INT.intern_type(&TypeDef::Lambda {
                    generics: vec![],
                    params: type_params,
                    rettype,
                });
                tck.add_var(deconstruct_name, Some(function_type), false);
            }
            todo!("Predeclare constructor");
        }
        */
    }
}

pub fn typecheck(ir: hir::Ir) -> Result<Tck, TypeError> {
    let mut tck = Tck::default();
    for decl in &ir.decls {
        predeclare_decl(&mut tck, decl)?;
    }
    for decl in &ir.decls {
        typecheck_decl(&mut tck, decl)?;
    }
    Ok(tck)
}
