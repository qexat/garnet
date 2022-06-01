//! Typechecking and other semantic checking.
//! Operates on the HIR.
//!
//! This is mostly based on
//! https://ahnfelt.medium.com/type-inference-by-example-793d83f98382
//!
//! We go through and generate a type variable for each expression,
//! then unify to solve them, probably in a more-or-less-bidirectional
//! fashion.

use std::collections::{HashMap, HashSet};

use crate::hir::{self, Expr};
use crate::{TypeDef, TypeSym, VarSym, INT};

#[derive(Debug, Clone)]
pub enum TypeError {
    UnknownVar(VarSym),
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
    TypeMismatch {
        expr_name: Cow<'static, str>,
        got: TypeSym,
        expected: TypeSym,
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
    fn add_var(&mut self, name: VarSym, binding: VarBinding) {
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

/// Top level type checking context struct.
#[derive(Clone, Debug)]
pub struct Tck {
    /// A mapping containing a type variable for each expression.
    /// Then we update the type variables with what we know about them
    /// as we typecheck/infer.
    exprtypes: HashMap<hir::Eid, TypeVar>,

    /// What info we know about our typevars.
    /// We may have multiple constraints for each type var, we keep
    /// a set of them.
    constraints: HashMap<TypeVar, HashSet<Constraint>>,
    /// Symbol table.  Current vars and types in scope.
    symtbl: Symtbl,

    /// Index of the next type var gensym.
    next_typevar: usize,
}

impl Default for Tck {
    /// Create a new Tck with whatever default types etc we need.
    fn default() -> Self {
        let mut x = Tck {
            exprtypes: Default::default(),
            constraints: Default::default(),
            symtbl: Default::default(),
            next_typevar: 0,
        };
        // We add a built-in function for printing, currently.
        {
            let name = INT.intern("__println");
            let typesym = INT.intern_type(&TypeDef::Lambda {
                generics: vec![],
                params: vec![INT.i32()],
                rettype: INT.unit(),
            });
            x.add_var(name, typesym, false);
        }
        {
            let name = INT.intern("__println_bool");
            let typesym = INT.intern_type(&TypeDef::Lambda {
                generics: vec![],
                params: vec![INT.bool()],
                rettype: INT.unit(),
            });
            x.add_var(name, typesym, false);
        }
        {
            let name = INT.intern("__println_i64");
            let typesym = INT.intern_type(&TypeDef::Lambda {
                generics: vec![],
                params: vec![INT.i64()],
                rettype: INT.unit(),
            });
            x.add_var(name, typesym, false);
        }
        {
            let name = INT.intern("__println_i16");
            let typesym = INT.intern_type(&TypeDef::Lambda {
                generics: vec![],
                params: vec![INT.i16()],
                rettype: INT.unit(),
            });
            x.add_var(name, typesym, false);
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

    pub fn new_typevar(&mut self) -> TypeVar {
        let tv = TypeVar(self.next_typevar);
        self.next_typevar += 1;
        tv
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
            mutable: false,
        };
        if let Some(t) = ty {
            self.add_constraint(tv, t)
        }
        self.symtbl.insert(name, binding);
    }

    pub fn add_constraint(&mut self) {
        todo!()
    }
}

/// Scan through all decl's and add any bindings to the symbol table,
/// so we don't need to do anything with forward references.
fn predeclare_decl(tck: &mut Tck, decl: &hir::Decl) {
    match decl {
        hir::Decl::Function {
            name, signature, ..
        } => {
            if tck.symtbl.binding_exists(*name) {
                panic!("Tried to redeclare function {}!", INT.fetch(*name));
            }
            // Add function to global scope
            let type_params = signature.params.iter().map(|(_name, t)| *t).collect();
            let function_type = INT.intern_type(&TypeDef::Lambda {
                generics: signature.generics.clone(),
                params: type_params,
                rettype: signature.rettype,
            });
            tck.symtbl.add_var(*name, function_type, false);
            tck.ctx = tck.ctx.clone().add_var(*name, function_type, false);
        }
        hir::Decl::Const { name, typename, .. } => {
            if tck.symtbl.binding_exists(*name) {
                panic!("Tried to redeclare const {}!", INT.fetch(*name));
            }
            tck.symtbl.add_var(*name, *typename, false);
            tck.ctx = tck.ctx.clone().add_var(*name, *typename, false);
        }
        hir::Decl::TypeDef { name, typedecl } => {
            // Gotta make sure there's no duplicate declarations
            // This kinda has to happen here rather than in typeck()
            if tck.symtbl.get_typedef(*name).is_some() {
                panic!("Tried to redeclare type {}!", INT.fetch(*name));
            }
            tck.symtbl.add_type(*name, *typedecl);
            todo!("Predeclare typedef");
        }
        hir::Decl::Constructor { name, signature } => {
            {
                if tck.symtbl.get_var(*name).is_ok() {
                    panic!(
                        "Aieeee, redeclaration of function/type constructor named {}",
                        INT.fetch(*name)
                    );
                }
                let type_params = signature.params.iter().map(|(_name, t)| *t).collect();
                let function_type = INT.intern_type(&TypeDef::Lambda {
                    generics: vec![],
                    params: type_params,
                    rettype: signature.rettype,
                });
                tck.symtbl.add_var(*name, function_type, false);
            }

            // Also we need to add a deconstructor function.  This is kinda a placeholder, but,
            // should work for now.
            {
                let deconstruct_name = INT.intern(format!("{}_unwrap", INT.fetch(*name)));
                if tck.symtbl.get_var(deconstruct_name).is_ok() {
                    panic!(
                        "Aieeee, redeclaration of function/type destructure named {}",
                        INT.fetch(deconstruct_name)
                    );
                }
                let type_params = vec![signature.rettype];
                let rettype = signature.params[0].1;
                let function_type = INT.intern_type(&TypeDef::Lambda {
                    generics: vec![],
                    params: type_params,
                    rettype,
                });
                tck.symtbl.add_var(deconstruct_name, function_type, false);
            }
            todo!("Predeclare constructor");
        }
    }
}

pub fn typecheck(ir: hir::Ir) -> Result<Tck, TypeError> {
    let mut tck = Tck::default();
    /*
    ir.decls.iter().for_each(|d| predeclare_decl(&mut tck, d));
    let checked_decls = ir
        .decls
        .into_iter()
        .map(|decl| typecheck_decl(&mut tck, decl))
        .collect::<Result<Vec<hir::Decl>, TypeError>>()?;
    */
    Ok(tck)
}
