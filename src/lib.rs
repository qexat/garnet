//! Garnet compiler guts.
//#![deny(missing_docs)]

use std::borrow::Cow;
use std::collections::HashMap;

pub mod ast;
pub mod intern;
pub mod ir;
pub mod typeck;

/// The interned name of a type
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct TypeSym(pub intern::Sym);

impl TypeSym {
    pub fn new(cx: &mut crate::Cx, name: &str) -> Self {
        TypeSym(cx.intern(name))
    }
}

/// The interned name of a variable/value
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct VarSym(pub intern::Sym);

impl VarSym {
    pub fn new(cx: &mut crate::Cx, name: &str) -> Self {
        VarSym(cx.intern(name))
    }
}

/// For now this is what we use as a type...
/// This doesn't include the name, just the properties of it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeDef {
    /// Signed integer with the given number of bytes
    SInt(usize),
    Bool,
    Tuple(Vec<TypeDef>),
    Lambda(Vec<TypeDef>, Box<TypeDef>),
}

impl TypeDef {
    pub fn get_name(&self) -> Cow<'static, str> {
        match self {
            TypeDef::SInt(4) => Cow::Borrowed("i32"),
            TypeDef::SInt(s) => panic!("Undefined integer size {}!", s),
            TypeDef::Bool => Cow::Borrowed("bool"),
            TypeDef::Tuple(v) => {
                if v.len() == 0 {
                    Cow::Borrowed("()")
                } else {
                    panic!("Can't yet define tuple {:?}", v)
                }
            }
            TypeDef::Lambda(params, _rettype) => panic!("Can't yet name lambda {:?}", params),
        }
    }
}

/// Compilation context.  Contains things like symbol tables.
/// Probably keeps most stages of compilation around to do
/// things like output error messages or whatever?
pub struct Cx {
    /// Interned symbols
    syms: intern::Interner,
    /// Known types
    types: HashMap<TypeSym, TypeDef>,
}

impl Cx {
    pub fn new() -> Self {
        let mut s = Cx {
            syms: intern::Interner::default(),
            types: HashMap::new(),
        };
        s.populate_builtin_types();
        s
    }

    /// Shortcut for getting an interned symbol.
    pub fn intern(&mut self, s: impl AsRef<str>) -> intern::Sym {
        self.syms.intern(s)
    }

    pub fn unintern(&self, s: intern::Sym) -> &str {
        self.syms.get(s)
    }

    /// Fill type table with whatever builtin types we have.
    fn populate_builtin_types(&mut self) {
        let types: HashMap<TypeSym, TypeDef> =
            [TypeDef::SInt(4), TypeDef::Bool, TypeDef::Tuple(vec![])]
                .iter()
                .cloned()
                .map(|t| (TypeSym(self.intern(t.get_name())), t))
                .collect();
        self.types = types;
    }

    /// Returns the symbol naming the given type, or none if it's
    /// not defined.
    pub fn get_typename(&mut self, name: &str) -> Option<TypeSym> {
        let s = TypeSym(self.syms.intern(name));
        if self.types.contains_key(&s) {
            Some(s)
        } else {
            None
        }
    }
}
