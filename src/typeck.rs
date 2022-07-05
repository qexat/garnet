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
use std::collections::HashMap;
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
    Mutability {
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
            TypeError::UnknownVar(sym) => format!("Unknown var: {}", sym.val()),
            TypeError::AlreadyDefined(sym) => format!(
                "Type, var, const, or function already defined: {}",
                sym.val()
            ),
            TypeError::TypeMismatch {
                expr_name,
                expected,
                got,
            } => format!(
                "Type mismatch in '{}' expresssion, expected {} but got {}",
                expr_name,
                expected.val().get_name(),
                got.val().get_name()
            ),
            TypeError::AmbiguousType { expr_name } => {
                format!("Ambiguous/unknown type for expression '{}'", expr_name)
            }
            TypeError::Mutability { expr_name } => {
                format!("Mutability mismatch in '{}' expresssion", expr_name)
            } /*
              TypeError::UnknownType(sym) => format!("Unknown type: {}", *sym.val()),
              TypeError::InvalidReturn => {
                  "return expression happened somewhere that isn't in a function!".to_string()
              }
              TypeError::Return {
                  fname,
                  got,
                  expected,
              } => format!(
                  "Function {} returns {} but should return {}",
                  *fname.val(),
                  *got.val().get_name(),
                  *expected.val().get_name(),
              ),
              TypeError::BopType {
                  bop,
                  got1,
                  got2,
                  expected,
              } => format!(
                  "Invalid types for BOp {:?}: expected {}, got {} + {}",
                  bop,
                  *expected.val().get_name(),
                  *got1.val().get_name(),
                  *got2.val().get_name()
              ),
              TypeError::UopType { op, got, expected } => format!(
                  "Invalid types for UOp {:?}: expected {}, got {}",
                  op,
                  *expected.val().get_name(),
                  *got.val().get_name()
              ),
              TypeError::LetType {
                  name,
                  got,
                  expected,
              } => format!(
                  "initializer for variable {}: expected {} ({:?}), got {} ({:?})",
                  *name.val(),
                  *expected.val().get_name(),
                  *expected,
                  *got.val().get_name(),
                  *got,
              ),
              TypeError::IfType { expected, got } => format!(
                  "If block return type is {}, but we thought it should be something like {}",
                  *expected.val().get_name(),
                  *got.val().get_name(),
              ),
              TypeError::Cond { got } => format!(
                  "If expr condition is {}, not bool",
                  *got.val().get_name(),
              ),
              TypeError::Param { got, expected } => format!(
                  "Function wanted type {} in param but got type {}",
                  *expected.val().get_name(),
                  *got.val().get_name()
              ),
              TypeError::Call { got } => format!(
                  "Tried to call function but it is not a function, it is a {}",
                  *got.val().get_name()
              ),
              TypeError::TupleRef { got } => format!(
                  "Tried to reference tuple but didn't get a tuple, got {}",
                  *got.val().get_name()
              ),
              TypeError::StructRef { fieldname, got } => format!(
                  "Tried to reference field {} of struct, but struct is {}",
                  *fieldname.val(),
                  *got.val().get_name(),
              ),
              TypeError::StructField { expected, got } => format!(
                  "Invalid field in struct constructor: expected {:?}, but got {:?}",
                  expected, got
              ),
              TypeError::EnumVariant { expected, got } => {
                  let expected_names: Vec<String> = expected
                      .into_iter()
                      .map(|nm| (&**nm.val()).clone())
                      .collect();
                  format!(
                      "Unknown enum variant '{}', valid ones are {:?}",
                      *got.val(),
                      expected_names,
                  )
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

impl Constraint {
    // Shortcuts for common known types
    fn bool() -> Self {
        Constraint::TypeSym(INT.bool())
    }
    fn unit() -> Self {
        Constraint::TypeSym(INT.unit())
    }
    fn iunknown() -> Self {
        Constraint::TypeSym(INT.iunknown())
    }
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
    /// Looks up a typedef and if it is `Named` try to keep looking
    /// it up until we find the actual concrete type.  Returns None
    /// if it can't.
    ///
    /// TODO: This is weird, currently necessary for structs though.
    fn follow_typedef(&mut self, name: VarSym) -> Option<TypeSym> {
        match self.types.get(&name) {
            Some(tsym) => match &**tsym.val() {
                &TypeDef::Named(vsym) => self.follow_typedef(vsym),
                _other => Some(*tsym),
            },
            None => None,
        }
    }
    */

    /*
    /// Take a typesym and look it up to a concrete type definition of some kind.
    /// Recursively follows named types, which might or might not be a good idea...
    ///
    /// TODO: This is weird, currently necessary for structs though.
    fn resolve_typedef(&mut self, t: TypeSym) -> Option<std::sync::Arc<TypeDef>> {
        let tdef = t.val();
        match &*tdef {
            TypeDef::NamedType(vsym) => self.resolve_typedef(*vsym).map(|sym| sym.val()),
            _other => Some(tdef),
        }
    }
    */

    /*
    /// Returns Ok if the type exists, or a TypeError of the appropriate
    /// kind if it does not.
    fn type_exists(&mut self, tsym: TypeSym) -> Result<(), TypeError> {
        match &*tsym.val() {
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

    /// Add a variable to the symbol table.  Panics on duplicates,
    /// since UniqueVar's are supposed to be, you know, unique.
    fn add_binding(&mut self, name: UniqueVar, binding: VarBinding) {
        if self.vars.insert(name, binding).is_some() {
            panic!("Unique var is not unique enough: {:?}", name)
        }
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

    fn _binding_exists(&self, name: UniqueVar) -> bool {
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
    /// We don't need a stack for generic types, they only
    /// occur at top-level constructs.  (Nested functions get
    /// lambda-lifted before now, remember.)
    /// They're kept here rather than in the Symtbl though,
    /// because they're context-dependent.
    generic_types: HashMap<VarSym, TypeVar>,
}

impl Default for Scope {
    /// We start with an empty toplevel scope existing.
    fn default() -> Scope {
        Scope {
            vars: Rc::new(RefCell::new(vec![HashMap::new()])),
            generic_types: HashMap::new(),
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

    fn add_generic(&mut self, name: VarSym, tvar: TypeVar) {
        self.generic_types.insert(name, tvar);
    }

    fn get_generic(&mut self, name: VarSym) -> Option<TypeVar> {
        self.generic_types.get(&name).cloned()
    }

    fn clear_generics(&mut self) {
        self.generic_types.clear();
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
    pub fn create_exprtype(&mut self, expr: &hir::TypedExpr) -> TypeVar {
        let tv = self.new_typevar();
        let old = self.exprtypes.insert(expr.id, tv);
        if old.is_some() {
            panic!(
                "Tried to bind {:?} twice, how unique is it?!\nExpr is {:?}",
                expr.id, expr
            );
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
            println!("Adding constraint {:?} to typevar {:?}", t, tv);
            self.add_constraint(tv, Constraint::TypeSym(t));
        }
        self.symtbl.add_binding(uniquevar, binding);
        self.scope.add_var(name, uniquevar);
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

    /// So, what is an lvalue?
    /// Well, it's a variable,
    /// or it's an lvalue in a deref expr or tupleref
    fn is_mutable_lvalue(&self, expr: &hir::Expr) -> Result<bool, TypeError> {
        match expr {
            hir::Expr::Var { name } => {
                let uniquevar = self
                    .scope
                    .get_var_with_scope(*name)
                    .ok_or(TypeError::UnknownVar(*name))?;
                let binding = self.symtbl.get_binding(uniquevar)?;
                Ok(binding.mutable)
            }
            hir::Expr::StructRef { expr, .. } => self.is_mutable_lvalue(&expr.e),
            _ => Ok(false),
        }
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
                let deconstruct_name = INT.intern(format!("{}_unwrap", *name.val()));
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
pub fn try_solve_type(tck: &Tck, tv: TypeVar) -> Option<TypeSym> {
    let tsym = tck.follow_typevar(tv)?;
    let tdef = &*tsym.val();
    match tdef {
        TypeDef::UnknownInt => Some(tsym),
        TypeDef::SInt(_) => Some(tsym),
        TypeDef::Tuple(items) if items.len() == 0 => Some(tsym),
        // Simple case first
        TypeDef::Lambda {
            generics,
            params: _,
            rettype: _,
        } if generics.len() == 0 => Some(tsym),
        TypeDef::Lambda {
            generics,
            params: _,
            rettype: _,
        } => Some(tsym),
        TypeDef::Never => Some(tsym),
        TypeDef::NamedTypeVar(..) => Some(tsym),
        other => todo!("Solve {:?}", other),
    }
}

/// Try to set v1 equal to v2
fn try_unify(tck: &mut Tck, v1: TypeVar, v2: TypeVar) -> Result<(), TypeError> {
    println!("Unifying {:?} and {:?}", v1, v2);
    let t1 = *tck.constraints.get(&v1).unwrap_or_else(|| {
        panic!(
            "Type variable 1 {:?} has no constraints, should not happen!",
            v1
        )
    });
    let t2 = *tck.constraints.get(&v2).unwrap_or_else(|| {
        panic!(
            "Type variable 2 {:?} has no constraints, should not happen!",
            v2
        )
    });
    match (t1, t2) {
        // Follow any references
        (Constraint::TypeVar(t1var), _) => try_unify(tck, t1var, v2),
        (_, Constraint::TypeVar(t2var)) => try_unify(tck, v1, t2var),
        // Un-intern type syms and unify from there
        (Constraint::TypeSym(s1), Constraint::TypeSym(s2)) => {
            let tdef1 = &*s1.val();
            let tdef2 = &*s2.val();
            use TypeDef::*;
            match (tdef1, tdef2) {
                // Primitives are easy
                (Bool, Bool) => Ok(()),
                (SInt(i1), SInt(i2)) if i1 == i2 => Ok(()),
                (UnknownInt, UnknownInt) => Ok(()),
                // If we have a known an unknown int, update the unknown
                // one to match the known one
                (UnknownInt, SInt(_i2)) => {
                    tck.constraints.insert(v1, t2);
                    Ok(())
                }
                (SInt(_i1), UnknownInt) => {
                    tck.constraints.insert(v2, t1);
                    Ok(())
                }
                // If the types are actually literally equal then there's
                // not really any way for them to not be identical?
                (s1, s2) if s1 == s2 => Ok(()),
                // Never types!  They never occur, so they always match
                // anything.
                // I think?  Hmmmm...
                (Never, _) => Ok(()),
                (_, Never) => Ok(()),
                (NamedTypeVar(nm), thing) => {
                    let tvar = tck.scope.get_generic(*nm).expect("Unbound generic type");
                    tck.constraints.insert(tvar, t2);
                    Ok(())
                }

                (s1, s2) => panic!(
                    "Type mismatch trying to unify concrete types {:?} and {:?}",
                    s1, s2
                ),
            }
        }
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
    rettype: TypeSym,
) -> Result<Constraint, TypeError> {
    match &expr.e {
        Expr::Lit { val } => {
            let lit_type = infer_literal(val)?;
            Ok(Constraint::TypeSym(lit_type))
        }
        Expr::Var { name } => {
            let uvar = tck
                .scope
                .get_var_with_scope(*name)
                .ok_or(TypeError::UnknownVar(*name))?;
            let typevar = tck.symtbl.get_binding(uvar)?.typevar;
            Ok(Constraint::TypeVar(typevar))
        }
        Expr::BinOp { op, lhs, rhs } => {
            use crate::ast::BOp::*;
            match op {
                // Logical operations always take and return bool
                And | Or | Xor => {
                    let boolconstraint = Constraint::bool();
                    check_expr(tck, lhs, boolconstraint, rettype)?;
                    check_expr(tck, rhs, boolconstraint, rettype)?;

                    Ok(boolconstraint)
                }
                // If we have a numerical operation, we find the types
                // of the arguments and make sure they are matching numeric
                // types.
                /*
                Add | Sub | Mul | Div | Mod => {
                    // Input constraints: LHS and RHS must be the same type
                    let lhs_typevar = tck.create_exprtype(lhs);
                    let rhs_typevar = tck.create_exprtype(rhs);
                    tck.add_constraint(lhs_typevar, Constraint::TypeVar(rhs_typevar));
                    // Input constraint: RHS must be a number (and thus
                    // LHS must be a number)
                    tck.add_constraint(rhs_typevar, Constraint::iunknown());
                    // Output constraint: This operation returns
                    // some kind of integer that is the same as the
                    // input types.
                    let ret = Ok(Constraint::TypeVar(lhs_typevar));

                    let _lhs_constraint = infer_expr(tck, lhs, rettype)?;
                    let _rhs_constraint = infer_expr(tck, rhs, rettype)?;
                    try_unify(tck, lhs_typevar, rhs_typevar)?;

                    ret
                }
                */
                other => todo!("binop: {:?}", other),
            }
        }
        Expr::Funcall { .. } => {
            todo!("infer funcall")
        }
        Expr::If { cases } => {
            let boolconstraint = Constraint::bool();
            //let body_rettype = Constraint::TypeVar(tck.new_typevar());
            assert_ne!(cases.len(), 0,  "The length of cases can not be 0 because we always have at least an if case and an else cases added in lowering");
            let mut last_constraint = Constraint::unit();
            for (cond, body) in cases {
                check_expr(tck, cond, boolconstraint, rettype)?;
                //check_exprs(tck, body, body_rettype, rettype)?;
                last_constraint = infer_exprs(tck, body, rettype)?;
            }
            let expr_typevar = tck.new_typevar();
            tck.add_constraint(expr_typevar, last_constraint);
            Ok(Constraint::TypeVar(expr_typevar))
        }
        Expr::UniOp { op, rhs } => {
            use crate::ast::UOp::*;
            match op {
                Neg => {
                    let intconstraint = Constraint::iunknown();
                    check_expr(tck, rhs, intconstraint, rettype)?;
                    Ok(intconstraint)
                }
                Not => {
                    let boolconstraint = Constraint::bool();
                    check_expr(tck, rhs, boolconstraint, rettype)?;
                    Ok(boolconstraint)
                }
                other => todo!("Infer uni op {:?}", other),
            }
        }
        Expr::Let { .. } => {
            let unitconstraint = Constraint::unit();
            check_expr(tck, expr, unitconstraint, rettype)?;
            Ok(unitconstraint)
        }
        e => {
            todo!("infer_expr: {:?}", e)
        }
    }
}

/// Just like check_exprs(), but starts off with inferring.
fn infer_exprs(
    tck: &mut Tck,
    exprs: &[hir::TypedExpr],
    rettype: TypeSym,
) -> Result<Constraint, TypeError> {
    let last_expr_idx = exprs.len();
    if last_expr_idx > 0 {
        for expr in &exprs[..(last_expr_idx - 1)] {
            let _ignore = infer_expr(tck, expr, rettype)?;
        }
        let last_expr = &exprs[last_expr_idx - 1];
        infer_expr(tck, last_expr, rettype)
    } else {
        // Body is empty, so the return type must be unit
        if rettype == INT.unit() {
            Ok(Constraint::unit())
        } else {
            Err(TypeError::TypeMismatch {
                expr_name: "empty block".into(),
                got: rettype,
                expected: INT.unit(),
            })
        }
    }
}

/// The "check" step of our vaguely-bidi type inference.  This is where
/// checking an expr starts.
///
/// rettype is the function return type, any non-local exits (`return`,
/// `?`, etc) need to know that type.
///
/// The process:
///  * Create typevar for expr
///  * Infer types for subexpr's if necessary
///  * xor check types for subexpr's if we know what they should be
///  * If we inferred, unify "expected" with the subexpression's typevar
fn check_expr(
    tck: &mut Tck,
    expr: &hir::TypedExpr,
    expected: Constraint,
    rettype: TypeSym, // function rettype is always known for sure
) -> Result<(), TypeError> {
    let expr_typevar = tck.create_exprtype(expr);
    // create a 'dummy' type var for the expected constraint,
    // because unification always has to work on TypeVar's, not
    // Constraint's, so it can update the variables involved.
    let expected_var = tck.new_typevar();
    tck.add_constraint(expected_var, expected);
    println!("Checking expression {:?}", expr);
    println!(
        "Expr typevar: {:?}, expected type {:?}",
        expr_typevar, expected
    );
    match &expr.e {
        Expr::Lit { .. } => {
            let constraint = infer_expr(tck, expr, rettype)?;
            println!("Inferred {:?} for expr {:?}", constraint, expr_typevar);
            tck.add_constraint(expr_typevar, constraint);
            println!("Lit typevar is {:?}", expr_typevar);
            try_unify(tck, expected_var, expr_typevar)?;
        }
        Expr::Let {
            varname,
            typename,
            init,
            mutable,
        } => {
            // create our unique var name and
            // get the type var for that particular variable
            let uniquevar = tck.create_var(*varname, Some(*typename), *mutable);
            let var_type = tck
                .symtbl
                .get_var_type(uniquevar)
                .expect("Should never happen");
            // Check the init expression and make sure it matches with the
            // declared type
            check_expr(tck, init, Constraint::TypeVar(var_type), *typename)?;
            // The return type of the `let` expression is unit, so
            // set it to that and make sure it unifies with the expected
            // type.
            tck.add_constraint(expr_typevar, Constraint::unit());
            try_unify(tck, expected_var, expr_typevar)?;
        }
        Expr::Var { name } => {
            // Get the unique var for this var in our scope
            let uvar = tck
                .scope
                .get_var_with_scope(*name)
                .ok_or(TypeError::UnknownVar(*name))?;
            // find its type variable and see if it unifies with
            // our expected type
            let typevar = tck.symtbl.get_binding(uvar)?.typevar;
            println!("Unique var {:?} has typevar {:?}", uvar, typevar);
            // We gotta remember: We have typevars for variables, and typevars
            // for expressions, and they are *not the same* because variables
            // can be declared in function args which are not expressions.
            // So here we make them depend on each other so we can unify em.
            tck.add_constraint(expr_typevar, Constraint::TypeVar(typevar));
            try_unify(tck, typevar, expected_var)?;
        }
        Expr::BinOp { op, lhs, rhs } => {
            use crate::ast::BOp::*;
            match op {
                // Logical operations always take and return bool
                And | Or | Xor => {
                    // Make sure our subexpr's return bool
                    let boolconstraint = Constraint::bool();
                    check_expr(tck, lhs, boolconstraint, rettype)?;
                    check_expr(tck, rhs, boolconstraint, rettype)?;
                    // Make it so the current expr returns bool and check whether
                    // that unifies with the expected type
                    tck.add_constraint(expr_typevar, boolconstraint);
                    try_unify(tck, expr_typevar, expected_var)?;
                }
                // If we have a numerical operation, we find the types
                // of the arguments and make sure they are matching numeric
                // types.
                Add | Sub | Mul | Div | Mod => {
                    // Input constraints: LHS and RHS must be the same type
                    // and must be numbers
                    //
                    // Sooo I think we check one side of the expr is
                    // some kind of integer,
                    // then check that the other side is compatible with it?
                    let numconstraint = Constraint::iunknown();
                    check_expr(tck, lhs, numconstraint, rettype)?;
                    let lhs_var = tck.get_typevar_for_expression(lhs).expect("Can't happen?");
                    check_expr(tck, rhs, Constraint::TypeVar(lhs_var), rettype)?;

                    // Make sure the return type matches the type for the inputs
                    tck.add_constraint(expr_typevar, Constraint::TypeVar(lhs_var));
                    try_unify(tck, expr_typevar, expected_var)?;
                }
                // Comparison takes two values that must be the same type of number,
                // and returns a bool
                Gt | Lt | Gte | Lte => {
                    let numconstraint = Constraint::iunknown();
                    check_expr(tck, lhs, numconstraint, rettype)?;
                    let lhs_var = tck.get_typevar_for_expression(lhs).expect("Can't happen?");
                    check_expr(tck, rhs, Constraint::TypeVar(lhs_var), rettype)?;

                    let boolconstraint = Constraint::bool();
                    tck.add_constraint(expr_typevar, boolconstraint);
                    try_unify(tck, expr_typevar, expected_var)?;
                }
                // Equality take two values that must be the same type,
                // and return a bool
                Eq | Neq => {
                    let lhs_constraint = infer_expr(tck, lhs, rettype)?;
                    check_expr(tck, rhs, lhs_constraint, rettype)?;

                    let boolconstraint = Constraint::bool();
                    tck.add_constraint(expr_typevar, boolconstraint);
                    try_unify(tck, expr_typevar, expected_var)?;
                }
            }
        }
        Expr::Funcall { func, params } => {
            // Find type of function
            let func_constraint = infer_expr(tck, &*func, rettype)?;
            let func_typevar = tck.create_exprtype(&*func);
            tck.add_constraint(func_typevar, func_constraint);
            // Now, the function type may be anything, including not a function.
            // but, the thing is, for now our function types can *not*
            // have generic types or typevars, we *always* know the signature
            // of each of a function.

            let func_typesym = try_solve_type(tck, func_typevar).unwrap();
            let func_typedef = &*func_typesym.val();
            // And if it's actually a function...
            match func_typedef {
                TypeDef::Lambda {
                    params: param_types,
                    rettype: call_rettype,
                    generics: generics,
                } => {
                    // Ok, when we call a function with generic types,
                    // do we try to substitute those types for what we
                    // give them and then just check, or do we literally
                    // instantiate a new function with the types substituted
                    // and then typecheck as normal?
                    //
                    // It has to be the first because all we may actually
                    // know about the generic is "it is this type var".

                    // We go through the params we were given and check
                    // that they match the function's sig
                    assert_eq!(params.len(), param_types.len());
                    for (p, t) in params.iter().zip(param_types) {
                        // (the 'rettype' here is the enclosing function's rettype,
                        // not the one we're calling!)
                        check_expr(tck, p, Constraint::TypeSym(*t), rettype)?;
                    }
                    // Then we make sure the function's rettype equals the
                    // expected type
                    tck.add_constraint(expr_typevar, Constraint::TypeSym(*call_rettype));
                    try_unify(tck, expr_typevar, expected_var)?;
                }
                other => {
                    return Err(TypeError::TypeMismatch {
                        expr_name: format!("{:?}", expr).into(),
                        got: INT.intern_type(other),
                        expected: INT.intern_type(func_typedef),
                    })
                }
            }
        }
        // Ok so we need to assert that for each
        // IfCase, the condition returns a bool and the
        // bodies all return the same type.
        // The HIR has no falseblock, it just always inserts
        // `if(true)` at the end
        Expr::If { cases } => {
            let boolconstraint = Constraint::bool();
            tck.add_constraint(expr_typevar, expected);
            for (cond, body) in cases {
                check_expr(tck, cond, boolconstraint, rettype)?;
                check_exprs(tck, body, Constraint::TypeVar(expr_typevar), rettype)?;
            }
            try_unify(tck, expr_typevar, expected_var)?;
        }
        Expr::Block { body } => {
            tck.add_constraint(expr_typevar, expected);
            check_exprs(tck, body, Constraint::TypeVar(expr_typevar), rettype)?;
            try_unify(tck, expr_typevar, expected_var)?;
        }
        Expr::UniOp { op, rhs } => {
            use crate::ast::UOp;
            match op {
                UOp::Neg => {
                    let intconstraint = Constraint::iunknown();
                    check_expr(tck, rhs, intconstraint, rettype)?;
                    // Return type of this expression is the same as the input type
                    let rhs_var = tck.get_typevar_for_expression(rhs).expect("Can't happen?");
                    tck.add_constraint(expr_typevar, Constraint::TypeVar(rhs_var));
                    try_unify(tck, expr_typevar, expected_var)?;
                }
                UOp::Not => {
                    let boolconstraint = Constraint::bool();
                    check_expr(tck, rhs, boolconstraint, rettype)?;
                    tck.add_constraint(expr_typevar, boolconstraint);
                    try_unify(tck, expr_typevar, expected_var)?;
                }
                other => todo!("Uni op: {:?}", other),
            }
        }
        Expr::Loop { body } => {
            let unitconstraint = Constraint::unit();
            check_exprs(tck, body, unitconstraint, rettype)?;

            // Loops return unit (for now)
            tck.add_constraint(expr_typevar, unitconstraint);
            try_unify(tck, expr_typevar, expected_var)?;
        }
        Expr::Assign { lhs, rhs } => {
            // Similar to `let`
            // hm right now our only lvalues are variables, so this is simple
            // for now but will get more complicated.
            if !tck.is_mutable_lvalue(&lhs.e)? {
                return Err(TypeError::Mutability {
                    expr_name: "assignment".into(),
                });
            }
            let lconstraint = infer_expr(tck, lhs, rettype)?;
            let lhs_typevar = tck.create_exprtype(lhs);
            tck.add_constraint(lhs_typevar, lconstraint);
            check_expr(tck, rhs, lconstraint, rettype)?;
            let rhs_typevar = tck.get_typevar_for_expression(rhs).expect("Can't happen?");
            try_unify(tck, lhs_typevar, rhs_typevar)?;
            // An assignment always returns unit
            tck.add_constraint(expr_typevar, Constraint::unit());
            try_unify(tck, expected_var, expr_typevar)?;
        }
        Expr::Break => {
            // Break's just return unit (for now)
            // TODO: Make sure we're actually in a loop
            let unitconstraint = Constraint::unit();
            tck.add_constraint(expr_typevar, unitconstraint);
            try_unify(tck, expr_typevar, expected_var)?;
        }
        Expr::StructCtor { body } if body.len() == 0 => {
            // This is just a unit literal
            let unitconstraint = Constraint::unit();
            tck.add_constraint(expr_typevar, unitconstraint);
            try_unify(tck, expr_typevar, expected_var)?;
        }
        Expr::Return { retval } => {
            let retconstraint = Constraint::TypeSym(rettype);
            check_expr(tck, retval, retconstraint, rettype)?;
            tck.add_constraint(expr_typevar, Constraint::TypeSym(INT.never()));
            try_unify(tck, expr_typevar, expected_var)?;
        }
        e => {
            todo!("check_expr for expr {:?}", e)
            /*
            eprintln!("Checking thing: {:?} ", e);
            let a = infer_expr(tck, expr, rettype)?;
            eprintln!("Type of {:?} is: {:?}", e, a);
                */
            // What is a substitution? It's a ~~miserable pile of secrets~~
            // map from type variables to types.
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
    expected: Constraint,
    rettype: TypeSym,
) -> Result<(), TypeError> {
    let last_expr_idx = exprs.len();
    // This loop is the sort of thing that feels like there should
    // be some turbo fancy combinator for it and it really isn't
    // worth the trouble.
    // I guess it'd be easier with recursion but eh
    if last_expr_idx > 0 {
        for expr in &exprs[..(last_expr_idx - 1)] {
            // We expect exprs in the body of the value to return unit
            // ...but we want to allow them to be anything...
            // So... I guess we check that they are equivalent to the
            // bottom type?
            //
            // TODO: I'm REALLY NOT SURE using the Never type here is
            // at all correct, but it works for now?
            // Maybe try inferring the expr's instead.
            // Another option would be to do basically what Rust does
            // with ; and wrap contiguous expressions in an `ignore` function,
            // but that feels kinda weird, especially since we don't
            // require semicolons.
            let cons = Constraint::TypeSym(INT.never());
            check_expr(tck, expr, cons, rettype)?;
        }
        // Check that the last expression has the expected
        // return type
        let last_expr = &exprs[last_expr_idx - 1];
        check_expr(tck, last_expr, expected, rettype)?;
        Ok(())
    } else {
        // Body is empty, so the return type must be unit
        if rettype == INT.unit() {
            Ok(())
        } else {
            Err(TypeError::TypeMismatch {
                expr_name: "empty block".into(),
                got: rettype,
                expected: INT.unit(),
            })
        }
    }
}

fn typecheck_decl(tck: &mut Tck, decl: &hir::Decl) -> Result<(), TypeError> {
    match decl {
        hir::Decl::Function {
            name: _name,
            signature,
            body,
        } => {
            let _guard = tck.scope.push();
            // Add function params to symbol table and scope
            for (var, ty) in &signature.params {
                let uniquevar = tck.create_var(*var, Some(*ty), false);
                tck.scope.add_var(*var, uniquevar);
            }
            // Add generic params to scope
            tck.scope.clear_generics();
            for name in &signature.generics {
                let tv = tck.new_typevar();
                tck.scope.add_generic(*name, tv);
            }
            check_exprs(
                tck,
                &body,
                Constraint::TypeSym(signature.rettype),
                signature.rettype,
            )?;
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
                let deconstruct_name = INT.intern(format!("{}_unwrap", *name.val()));
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

pub fn typecheck(ir: &hir::Ir) -> Result<Tck, TypeError> {
    let mut tck = Tck::default();
    for decl in &ir.decls {
        predeclare_decl(&mut tck, decl)?;
    }
    for decl in &ir.decls {
        typecheck_decl(&mut tck, decl)?;
    }
    Ok(tck)
}
