//! Garnet compiler guts.
//#![deny(missing_docs)]

use std::borrow::Cow;
use std::sync::Arc;

pub mod ast;
pub mod backend;
pub mod format;
pub mod hir;
pub mod intern;
pub mod lir;
pub mod parser;
pub mod passes;
mod scope;
pub mod typeck;

#[cfg(test)]
pub(crate) mod testutil;

use once_cell::sync::Lazy;
/// The interner.  It's the ONLY part we have to actually
/// carry around anywhere, so I'm experimenting with not
/// heckin' bothering.  Seems to work pretty okay.
pub static INT: Lazy<Cx> = Lazy::new(|| Cx::new());

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

/// For now this is what we use as a type...
/// This doesn't include the name, just the properties of it.
///
/// A real type def that has had all the inference stuff done
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeDef {
    /// Signed integer with the given number of bytes
    SInt(u8),
    /// An integer of unknown size, from an integer literal
    UnknownInt,
    Bool,
    /// We can infer types for tuples
    Tuple(Vec<TypeSym>),
    /// Never is a real type, I guess!
    Never,
    Lambda(Vec<TypeSym>, TypeSym),
    // /// Are names parts of structs?  I guess not.
    //Struct(Vec<(VarSym, Sym)>),
    /// This is basically a type that has been named but we
    /// don't know what type it actually is until after type checking...
    ///
    /// ...This is going to get real sticky once we have modules, I think.
    /// Using `VarSym` for this feels wrong, but, we will leave it for now.
    /// I think this actually *has* to be `VarSym` or else we lose the actual
    /// name, which is important.
    Named(VarSym),
}

impl TypeDef {
    /// Get a string for the name of the given type.
    ///
    /// TODO: We need a good way to go the other way around as well,
    /// for built-in types like I32 and Bool.
    pub fn get_name(&self) -> Cow<'static, str> {
        let join_types_with_commas = |lst: &[TypeSym]| {
            let p_strs = lst
                .iter()
                .map(|ptype| {
                    let ptype_def = INT.fetch_type(*ptype);
                    ptype_def.get_name()
                })
                .collect::<Vec<_>>();
            p_strs.join(", ")
        };
        match self {
            TypeDef::SInt(16) => Cow::Borrowed("I128"),
            TypeDef::SInt(8) => Cow::Borrowed("I64"),
            TypeDef::SInt(4) => Cow::Borrowed("I32"),
            TypeDef::SInt(2) => Cow::Borrowed("I16"),
            TypeDef::SInt(1) => Cow::Borrowed("I8"),
            TypeDef::SInt(s) => panic!("Undefined integer size {}!", s),
            TypeDef::UnknownInt => Cow::Borrowed("{number}"),
            TypeDef::Bool => Cow::Borrowed("Bool"),
            TypeDef::Never => Cow::Borrowed("Never"),
            TypeDef::Tuple(v) => {
                if v.len() == 0 {
                    Cow::Borrowed("{}")
                } else {
                    let mut res = String::from("{");
                    res += &join_types_with_commas(v);
                    res += "}";
                    Cow::Owned(res)
                }
            }
            TypeDef::Lambda(params, rettype) => {
                let mut t = String::from("fn(");
                t += &join_types_with_commas(params);

                t += ")";
                let ret_def = INT.fetch_type(*rettype);
                match &*ret_def {
                    TypeDef::Tuple(x) if x.len() == 0 => {
                        // pass, implicitly return unit
                    }
                    x => {
                        let type_str = x.get_name();
                        t += ": ";
                        t += &type_str;
                    }
                }
                Cow::Owned(t)
            }
            TypeDef::Named(s) => Cow::Owned((&*INT.fetch(*s)).clone()),
            /*
            TypeDef::Ptr(t) => {
                let inner_name = cx.fetch_type(**t).get_name();
                let s = format!("{}^", inner_name);
                Cow::Owned(s)
            }
            */
        }
    }

    /// Returns true if this is a `SInt` or `UnknownInt`.
    pub fn is_integer(&self) -> bool {
        match self {
            TypeDef::SInt(_) => true,
            TypeDef::UnknownInt => true,
            _ => false,
        }
    }
}

/// Interner context.
///
/// Really this is just an interner for symbols now, and
/// the original plan of bundling it up into a special context
/// type for each step of compilation hasn't really worked out.
#[derive(Debug)]
pub struct Cx {
    /// Interned symbols
    syms: intern::Interner<VarSym, String>,
    /// Known types
    types: intern::Interner<TypeSym, TypeDef>,
    //files: cs::files::SimpleFiles<String, String>,
}

impl Cx {
    pub fn new() -> Self {
        let s = Cx {
            syms: intern::Interner::new(),
            types: intern::Interner::new(),
            //files: cs::files::SimpleFiles::new(mod_name.to_owned(), source.to_owned()),
        };
        s
    }

    /// Intern the symbol.
    pub fn intern(&self, s: impl AsRef<str>) -> VarSym {
        self.syms.intern(&s.as_ref().to_owned())
    }

    /// Get the string for a variable symbol
    pub fn fetch(&self, s: VarSym) -> Arc<String> {
        self.syms.fetch(s)
    }

    /// Intern a type defintion.
    pub fn intern_type(&self, s: &TypeDef) -> TypeSym {
        self.types.intern(&s)
    }

    /// Intern a "Named" typedef of the given name.
    pub fn named_type(&self, s: impl AsRef<str>) -> TypeSym {
        let sym = self.intern(s);
        self.types.intern(&TypeDef::Named(sym))
    }

    /// Get the TypeDef for a type symbol
    pub fn fetch_type(&self, s: TypeSym) -> Arc<TypeDef> {
        self.types.fetch(s)
    }

    /// Shortcut for getting the type symbol for an unknown int
    pub fn iunknown(&self) -> TypeSym {
        self.intern_type(&TypeDef::UnknownInt)
    }

    /// Shortcut for getting the type symbol for I128
    pub fn i128(&self) -> TypeSym {
        self.intern_type(&TypeDef::SInt(16))
    }

    /// Shortcut for getting the type symbol for I64
    pub fn i64(&self) -> TypeSym {
        self.intern_type(&TypeDef::SInt(8))
    }

    /// Shortcut for getting the type symbol for I32
    pub fn i32(&self) -> TypeSym {
        self.intern_type(&TypeDef::SInt(4))
    }

    /// Shortcut for getting the type symbol for I16
    pub fn i16(&self) -> TypeSym {
        self.intern_type(&TypeDef::SInt(2))
    }

    /// Shortcut for getting the type symbol for I8
    pub fn i8(&self) -> TypeSym {
        self.intern_type(&TypeDef::SInt(1))
    }

    /// Shortcut for getting the type symbol for Bool
    pub fn bool(&self) -> TypeSym {
        self.intern_type(&TypeDef::Bool)
    }

    /// Shortcut for getting the type symbol for Unit
    pub fn unit(&self) -> TypeSym {
        self.intern_type(&TypeDef::Tuple(vec![]))
    }

    /// Shortcut for getting the type symbol for Never
    pub fn never(&self) -> TypeSym {
        self.intern_type(&TypeDef::Never)
    }

    /// Generate a new unique symbol including the given string
    /// Useful for some optimziations and intermediate names and such.
    ///
    /// Starts with a `@` which currently cannot appear in any identifier.
    pub fn gensym(&self, s: &str) -> VarSym {
        let sym = format!("@{}_{}", s, self.syms.count());
        self.intern(sym)
    }
}

/// Main driver function.
/// Compile a given source string to ~~wasm~~ Rust, or return an error.
/// TODO: Parser errors?
///
/// Parse -> lower to IR -> run transformation passes
/// -> typecheck -> compile to wasm
pub fn try_compile(src: &str) -> Result<Vec<u8>, typeck::TypeError> {
    let ast = {
        let mut parser = parser::Parser::new(src);
        parser.parse()
    };
    let hir = hir::lower(&mut |_| (), &ast);
    let hir = passes::run_passes(hir);
    let checked = typeck::typecheck(hir)?;
    Ok(backend::output(backend::Backend::Rust, &checked))
}

/// For when we don't care about catching results
pub fn compile(src: &str) -> Vec<u8> {
    try_compile(src).unwrap_or_else(|e| panic!("Type check error: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Make sure outputting a lambda's name gives us something we expect.
    #[test]
    fn check_name_format() {
        let args = vec![INT.i32(), INT.bool()];
        let def = TypeDef::Lambda(args, INT.i32());
        let gotten_name = def.get_name();
        let desired_name = "fn(I32, Bool): I32";
        assert_eq!(&gotten_name, desired_name);
    }
}
