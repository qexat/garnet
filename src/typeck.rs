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
            /*
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

/// Immutable symbol table.  We just keep one of these associated with every expression,
/// which describes that expression's scope.  Since it's immutable, cloning and modifying
/// it only changes the parts that are necessary.
///
/// This means that we always can look at any expression and see its entire scope, and
/// keeping track of scope pushes/pops is basically implicit since there's no mutable state
/// involved.
#[derive(Clone, Debug, PartialEq)]
pub struct Symtbl {
    pub vars: im::HashMap<VarSym, VarBinding>,
    pub typenames: im::HashMap<VarSym, TypeSym>,
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
    fn add_binding(&mut self, name: VarSym, binding: VarBinding) {
        self.vars.insert(name, binding);
    }

    /// Get the type of the given variable, or an error.
    /// The type will be a type variable that must then be unified, I guess.
    fn get_var(&self, name: VarSym) -> Result<TypeVar, TypeError> {
        Ok(self.get_binding(name)?.typevar)
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

/// A unique var ID used to unambiguously identify variables regardless of scope.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
struct UniqueVar(usize);

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
/// Uses internal mutability and returns a guard type to manage push/pops
/// because otherwise life is kinda awful.
#[derive(Clone, Debug)]
struct Scope {
    vars: Rc<RefCell<Vec<HashMap<hir::Eid, UniqueVar>>>>,
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

    fn add_var(&self, eid: hir::Eid, unique: UniqueVar) {
        self.vars
            .borrow_mut()
            .last_mut()
            .expect("Scope stack underflow")
            .insert(eid, unique);
    }

    /// Checks whether the var exists in the currently alive scopes
    fn get_var_with_scope(&self, eid: hir::Eid) -> Option<UniqueVar> {
        for scope in self.vars.borrow().iter().rev() {
            let v = scope.get(&eid);
            if v.is_some() {
                return v.cloned();
            }
        }
        return None;
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
    constraints: HashMap<TypeVar, HashSet<Constraint>>,
    /// Symbol table.  Current vars and types in scope.
    symtbl: Symtbl,

    /// Contains the EID -> uniquevar mappings for all the
    /// variables in current compilation unit.  Maybe it
    /// would be simple to alpha-rename vars at some point
    /// to give them unique ID's explicitly?  We need
    /// scope checking info for that anyway; it's a bit
    /// more of a low-level thing.
    /// idk.
    all_vars: HashMap<hir::Eid, UniqueVar>,
    /// The current scope stack.
    /// This is its own type 'cause it's a little clearer
    /// where we're manipulating scope vs. all vars.
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
            all_vars: Default::default(),
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
            x.add_var(name, Some(typesym), false);
        }
        {
            let name = INT.intern("__println_bool");
            let typesym = INT.intern_type(&TypeDef::Lambda {
                generics: vec![],
                params: vec![INT.bool()],
                rettype: INT.unit(),
            });
            x.add_var(name, Some(typesym), false);
        }
        {
            let name = INT.intern("__println_i64");
            let typesym = INT.intern_type(&TypeDef::Lambda {
                generics: vec![],
                params: vec![INT.i64()],
                rettype: INT.unit(),
            });
            x.add_var(name, Some(typesym), false);
        }
        {
            let name = INT.intern("__println_i16");
            let typesym = INT.intern_type(&TypeDef::Lambda {
                generics: vec![],
                params: vec![INT.i16()],
                rettype: INT.unit(),
            });
            x.add_var(name, Some(typesym), false);
        }
        x
    }
}

impl Tck {
    /// Gets the concrete type associated with a particular expr ID.
    /// Assumes that we have already found a solution for it.
    ///
    /// Panics if expr id does not exist.
    pub fn get_solved_type(&self, eid: hir::Eid) -> TypeSym {
        todo!("actually solve types")
        /*
        *self
            .exprtypes
            .get(&eid)
            .expect("Tried to get type for Eid that doesn't exist!")
            */
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

    /// If the var's type is unknown, then we create a type var for it.
    ///
    /// If the var has a type associated with it, then we create a type
    /// var for it and list it as a known fact.
    pub fn add_var(&mut self, name: VarSym, ty: Option<TypeSym>, mutable: bool) {
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
        self.symtbl.add_binding(name, binding);
    }

    pub fn add_constraint(&mut self, tv: TypeVar, constraint: Constraint) {
        let entry = self.constraints.entry(tv).or_insert(HashSet::new());
        entry.insert(constraint);
    }
}

/// Scan through all decl's and add any bindings to the symbol table,
/// so we don't need to do anything with forward references.
fn predeclare_decl(tck: &mut Tck, decl: &hir::Decl) -> Result<(), TypeError> {
    match decl {
        hir::Decl::Function {
            name, signature, ..
        } => {
            if tck.symtbl.binding_exists(*name) {
                return Err(TypeError::AlreadyDefined(*name));
            }
            // Add function to global scope
            let function_type = signature.to_type();
            tck.add_var(*name, Some(function_type), false);
        }
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
    }
    Ok(())
}

/// The "check" step of our vaguely-bidi type inference.  This is where
/// checking an expr starts.
///
/// rettype is the function return type, any non-local exits (`return`,
/// `?`, etc) need to have that type.
fn expr_check(
    tck: &mut Tck,
    expr: &hir::TypedExpr,
    expected: TypeSym,
    rettype: TypeSym,
) -> Result<(), TypeError> {
    todo!("Expr_check")
}

fn typecheck_decl(tck: &mut Tck, decl: &hir::Decl) -> Result<(), TypeError> {
    match decl {
        hir::Decl::Function {
            name,
            signature,
            body,
        } => {
            let _guard = tck.scope.push();

            // TODO NEXT: Add signature to known-types

            let last_expr_idx = body.len();
            // This is the sort of thing that feels like there should
            // be some turbo fancy combinator for it and it really isn't
            // worth the trouble.
            if last_expr_idx > 0 {
                for expr in &body[..(last_expr_idx - 1)] {
                    expr_check(tck, expr, INT.unit(), signature.rettype)?;
                }
                expr_check(
                    tck,
                    &body[last_expr_idx - 1],
                    signature.rettype,
                    signature.rettype,
                )?;
            } else {
                // Body is empty, is the return type Unit?
                if signature.rettype != INT.unit() {
                    return Err(TypeError::TypeMismatch {
                        expr_name: Cow::from("empty function body"),
                        got: INT.unit(),
                        expected: signature.rettype,
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
