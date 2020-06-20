//! Garnet compiler guts.
//#![deny(missing_docs)]

use std::borrow::Cow;
use std::rc::Rc;

pub mod ast;
pub mod backend;
pub mod format;
pub mod intern;
pub mod ir;
pub mod parser;
pub mod typeck;

/// The interned name of a type
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct TypeSym(pub usize);

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

pub enum TypeInfo {
    /// Unknown type not inferred yet
    Unknown,
    /// Reference saying "this type is the same as that one",
    /// which may still be unknown.
    /// TODO: Symbol type needs to change.
    Ref(TypeSym),
    /// Known type.
    Known(TypeDef),
}

/// For now this is what we use as a type...
/// This doesn't include the name, just the properties of it.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeDef {
    /// Signed integer with the given number of bytes
    SInt(usize),
    Bool,
    /// We can infer types for tuples
    Tuple(Vec<TypeSym>),
    Lambda(Vec<TypeSym>, Box<TypeSym>),
    /*
    /// TODO: AUGJDKSFLJDSFSLAF
    /// This is basically a type that has been named but we
    /// don't know what type it actually is until after type checking...
    Named(String),
    */
}

impl TypeDef {
    pub fn get_name(&self) -> Cow<'static, str> {
        match self {
            TypeDef::SInt(4) => Cow::Borrowed("I32"),
            TypeDef::SInt(s) => panic!("Undefined integer size {}!", s),
            TypeDef::Bool => Cow::Borrowed("Bool"),
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
///
/// Really this is just an interner for symbols now, and
/// the original plan of bundling it up into a special context
/// type for each step of compilation hasn't really worked out
/// (at least for wasm with `walrus`.  Maybe rename it?
pub struct Cx {
    /// Interned symbols
    syms: intern::Interner<VarSym, String>,
    /// Known types
    types: intern::Interner<TypeSym, TypeDef>,
}

impl Cx {
    pub fn new() -> Self {
        let s = Cx {
            syms: intern::Interner::new(),
            types: intern::Interner::new(),
        };
        s
    }

    /// Shortcut for getting an interned symbol.
    pub fn intern(&self, s: impl AsRef<str>) -> VarSym {
        self.syms.intern(&s.as_ref().to_owned())
    }

    pub fn fetch(&self, s: VarSym) -> Rc<String> {
        self.syms.fetch(s)
    }

    pub fn intern_type(&self, s: &TypeDef) -> TypeSym {
        self.types.intern(&s)
    }

    pub fn fetch_type(&self, s: TypeSym) -> Rc<TypeDef> {
        self.types.fetch(s)
    }

    /// Shortcut for getting the type symbol for I32
    pub fn i32(&self) -> TypeSym {
        //cx.intern_type(&TypeDef::Tuple(vec![]));
        self.intern_type(&TypeDef::SInt(4))
    }

    /// Shortcut for getting the type symbol for Bool
    pub fn bool(&self) -> TypeSym {
        self.intern_type(&TypeDef::Bool)
    }

    /// Shortcut for getting the type symbol for Unit
    pub fn unit(&self) -> TypeSym {
        self.intern_type(&TypeDef::Tuple(vec![]))
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

/// Main driver function.
/// Compile a given source string to wasm.
pub fn compile(src: &str) -> Vec<u8> {
    let cx = &mut Cx::new();
    let ast = {
        let mut parser = parser::Parser::new(cx, src);
        parser.parse()
    };
    let ir = ir::lower(&ast);
    let checked = typeck::typecheck(cx, ir).unwrap();
    let wasm = backend::output(cx, &checked);
    wasm
}
