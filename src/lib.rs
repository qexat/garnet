//! Garnet compiler guts.
//#![deny(missing_docs)]

use std::borrow::Cow;

pub mod ast;
pub mod backend;
pub mod intern;
pub mod ir;
pub mod typeck;

/// The interned name of a type
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct TypeSym(pub usize);

impl TypeSym {
    pub fn new(cx: &mut crate::Cx, name: &TypeDef) -> Self {
        cx.intern_type(name)
    }
}

impl From<usize> for TypeSym {
    fn from(i: usize) -> TypeSym {
        TypeSym(i)
    }
}

impl From<TypeSym> for usize {
    fn from(i: TypeSym) -> usize {
        i.0
    }
}

/// The interned name of a variable/value
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct VarSym(pub usize);

impl VarSym {
    pub fn new(cx: &mut crate::Cx, name: &str) -> Self {
        cx.intern(name)
    }
}

impl From<usize> for VarSym {
    fn from(i: usize) -> VarSym {
        VarSym(i)
    }
}

impl From<VarSym> for usize {
    fn from(i: VarSym) -> usize {
        i.0
    }
}

/// For now this is what we use as a type...
/// This doesn't include the name, just the properties of it.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeDef {
    /// Unknown type not inferred yet
    Unknown,
    /// Reference saying "this type is the same as that one",
    /// which may still be unknown
    Ref(TypeSym),
    /// Signed integer with the given number of bytes
    SInt(usize),
    Bool,
    /// We can infer types for tuples
    Tuple(Vec<TypeSym>),
    Lambda(Vec<TypeSym>, Box<TypeSym>),
}

/* TODO:
 * Do we HAVE to separate concrete from maybe-unknown types?
 * How do we divide this up?
 * How does the interning work out?
 * Think about this.
/// The interned name of a variable/value
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct TypeInfoId(pub usize);

/// Information known about a possibly-unknown type.
pub enum TypeInfo {
    /// Unknown type not inferred yet
    Unknown,
    /// Reference saying "this type is the same as that one",
    /// which may still be unknown
    Ref(TypeInfoId),
    /// Signed integer with the given number of bytes
    Known(TypeSym),
}
*/

impl TypeDef {
    pub fn get_name(&self) -> Cow<'static, str> {
        match self {
            TypeDef::Unknown => Cow::Borrowed("unknown!"),
            TypeDef::Ref(s) => Cow::Owned(format!("'{}", s.0)),
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
    syms: intern::Interner<VarSym, String>,
    /// Known types
    types: intern::Interner<TypeSym, TypeDef>,
}

impl Cx {
    pub fn new() -> Self {
        let mut s = Cx {
            syms: intern::Interner::new(),
            types: intern::Interner::new(),
        };
        s.populate_builtin_types();
        s
    }

    /// Shortcut for getting an interned symbol.
    pub fn intern(&mut self, s: impl AsRef<str>) -> VarSym {
        self.syms.intern(&s.as_ref().to_owned())
    }

    pub fn unintern(&self, s: VarSym) -> &str {
        self.syms.get(s)
    }

    pub fn intern_type(&mut self, s: &TypeDef) -> TypeSym {
        self.types.intern(&s)
    }

    pub fn unintern_type(&self, s: TypeSym) -> &TypeDef {
        self.types.get(s)
    }

    /// Fill type table with whatever builtin types we have.
    fn populate_builtin_types(&mut self) {
        /*
        let types: HashMap<TypeSym, TypeDef> =
            [TypeDef::SInt(4), TypeDef::Bool, TypeDef::Tuple(vec![])]
                .iter()
                .cloned()
                .map(|t| (self.intern(t.get_name()), t))
                .collect();
        self.types = types;
        */
    }

    /*
    /// Returns the symbol naming the given type, or none if it's
    /// not defined.
    pub fn get_typename(&mut self, name: &str) -> Option<TypeSym> {
        let s = self.types.intern(name);
        if self.types.contains_key(&s) {
            Some(s)
        } else {
            None
        }
    }
    */
}
