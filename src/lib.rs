//! Garnet compiler guts.
//#![deny(missing_docs)]

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

pub mod ast;
pub mod backend;
pub mod format;
pub mod hir;
pub mod intern;
pub mod parser;
pub mod passes;
//pub mod typechonk;
pub mod typeck;

#[cfg(test)]
pub(crate) mod testutil;

use once_cell::sync::Lazy;
/// The interner.  It's the ONLY part we have to actually
/// carry around anywhere, so I'm experimenting with making
/// it a global.  Seems to work pretty okay.
pub static INT: Lazy<Cx> = Lazy::new(Cx::new);

/// The interned name of a type
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TypeSym(pub usize);

/// A synthesized type with a number attached to it.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TypeId(pub usize);

/// Required for interner interface.
impl From<usize> for TypeSym {
    fn from(i: usize) -> TypeSym {
        TypeSym(i)
    }
}

/// Required for interner interface.
impl From<TypeSym> for usize {
    fn from(i: TypeSym) -> usize {
        i.0
    }
}

/// The interned name of a variable/value
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VarSym(pub usize);

/// Required for interner interface.
impl From<usize> for VarSym {
    fn from(i: usize) -> VarSym {
        VarSym(i)
    }
}

/// Required for interner interface.
impl From<VarSym> for usize {
    fn from(i: VarSym) -> usize {
        i.0
    }
}

/// A path of modules/structs/whatever
/// `foo.bar.bop`
pub struct Path {
    /// The last part of the path, must always exist.
    pub name: VarSym,
    /// The rest of the (possibly empty) path.
    /// TODO: Whether this must be absolute or could be relative is currently undefined.
    pub path: Vec<VarSym>,
}

/// A complete-ish description of a type.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeDef {
    /// Signed integer with the given number of bytes
    SInt(u8),
    /// An integer of unknown size, from an integer literal
    UnknownInt,
    /// Boolean, obv's
    Bool,
    /// Tuple.  The types inside it may or may not be fully known I guess
    Tuple(Vec<TypeSym>),
    /// Never is a real type, I guess!
    Never,
    /// The type of a lambda is its signature
    Lambda(Vec<TypeSym>, TypeSym),
    /*
    /// This is basically a type that has been named but we
    /// don't know what type it actually is until after type checking...
    ///
    /// ...This is going to get real sticky once we have modules, I think.
    /// Using `VarSym` for this feels wrong, but, we will leave it for now.
    /// I think this actually *has* to be `VarSym` or else we lose the actual
    /// name, which is important.
    Named(VarSym),
    */
    /// A struct.
    Struct {
        fields: BTreeMap<VarSym, TypeSym>,
        typefields: BTreeSet<VarSym>,
    },
    Enum {
        /// TODO: For now the only size of an enum is i32.
        variants: Vec<(VarSym, i32)>,
    },
    //Generic(VarSym),
    /// A type var that might be provided by the user???
    TypeVar(VarSym),
    /// A possibly-unsolved implicit type var???
    ExistentialVar(TypeId),
    /// A generic decl(?????)
    ForAll(TypeId, TypeSym),
}

impl TypeDef {
    /// Takes a string and matches it against the builtin/
    /// primitive types, returning the appropriate `TypeDef`
    ///
    /// If it is not a built-in type, or is a compound such
    /// as a tuple or struct, returns None.
    ///
    /// The effective inverse of `TypeDef::get_name()`.  Compound
    /// types take more parsing to turn from strings to `TypeDef`'s
    /// and are handled in `parser::parse_type()`.
    pub fn get_primitive_type(s: &str) -> Option<TypeDef> {
        match s {
            "I8" => Some(TypeDef::SInt(1)),
            "I16" => Some(TypeDef::SInt(2)),
            "I32" => Some(TypeDef::SInt(4)),
            "I64" => Some(TypeDef::SInt(8)),
            "I128" => Some(TypeDef::SInt(16)),
            "Bool" => Some(TypeDef::Bool),
            "Never" => Some(TypeDef::Never),
            _ => None,
        }
    }

    /// Get a string for the name of the given type.
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
        let join_vars_with_commas = |lst: &BTreeMap<VarSym, TypeSym>| {
            let p_strs = lst
                .iter()
                .map(|(pname, ptype)| {
                    let ptype_def = INT.fetch_type(*ptype);
                    let pname = INT.fetch(*pname);
                    format!("{}: {}", pname, ptype_def.get_name())
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
                if v.is_empty() {
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
                    TypeDef::Tuple(x) if x.is_empty() => {
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
            TypeDef::Struct { fields, .. } => {
                let mut res = String::from("struct {");
                res += &join_vars_with_commas(fields);
                res += "}";
                Cow::Owned(res)
            }
            TypeDef::Enum { variants } => {
                let mut res = String::from("enum {");
                let s = variants
                    .iter()
                    .map(|(nm, vl)| format!("    {} = {},\n", INT.fetch(*nm), vl))
                    .collect::<String>();
                res += &s;
                res += "\n}\n";
                Cow::Owned(res)
            }
            TypeDef::TypeVar(vsym) => Cow::Owned((&*INT.fetch(*vsym)).clone()),
            TypeDef::ExistentialVar(tid) => Cow::Owned(format!("'{}", tid.0)),
            TypeDef::ForAll(tid, tsym) => Cow::from(format!(
                "forall('{} -> {})",
                tid.0,
                INT.fetch_type(*tsym).get_name()
            )),
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

    /// Returns true if this is a tuple of length zero.
    /// Handy sometimes since we don't have a separate unit type.
    pub fn is_unit(&self) -> bool {
        match self {
            TypeDef::Tuple(v) => v.len() == 0,
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
        Cx {
            syms: intern::Interner::new(),
            types: intern::Interner::new(),
            //files: cs::files::SimpleFiles::new(mod_name.to_owned(), source.to_owned()),
        }
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

    /*
    /// Intern a "Named" typedef of the given name.
    pub fn named_type(&self, s: impl AsRef<str>) -> TypeSym {
        let sym = self.intern(s);
        self.types.intern(&TypeDef::Named(sym))
    }
    */

    /// Get the TypeDef for a type symbol
    pub fn fetch_type(&self, s: TypeSym) -> Arc<TypeDef> {
        self.types.fetch(s)
    }

    /// Shortcut for getting the type symbol for an unknown int
    pub fn iunknown(&self) -> TypeSym {
        self.intern_type(&TypeDef::UnknownInt)
    }

    /// Shortcut for getting the type symbol for an int of a particular size
    pub fn isize(&self, size: u8) -> TypeSym {
        self.intern_type(&TypeDef::SInt(size))
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
/// Compile a given source string to Rust source code, or return an error.
/// TODO: Parser errors?
///
/// Parse -> lower to IR -> run transformation passes
/// -> typecheck -> compile to wasm
pub fn try_compile(filename: &str, src: &str) -> Result<Vec<u8>, typeck::TypeError> {
    let ast = {
        let mut parser = parser::Parser::new(filename, src);
        parser.parse()
    };
    let hir = hir::lower(&mut |_| (), &ast);
    let hir = passes::run_passes(hir);
    // TODO: Get rid of this clone, typechecking no longer returns a new AST.
    let tck = typeck::typecheck(hir.clone())?;
    Ok(backend::output(backend::Backend::Rust, &hir, &tck))
}

/// For when we don't care about catching results
pub fn compile(filename: &str, src: &str) -> Vec<u8> {
    try_compile(filename, src).unwrap_or_else(|e| panic!("Type check error: {}", e))
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

    /// Make sure that `TypeDef::get_name()` and `TypeDef::get_primitive_type()`
    /// are inverses for all primitive types.
    ///
    /// TODO: This currently requires that we have a list of all primitive types here,
    /// which is somewhat annoying.
    #[test]
    fn check_primitive_names() {
        let prims = vec![
            TypeDef::SInt(1),
            TypeDef::SInt(2),
            TypeDef::SInt(4),
            TypeDef::SInt(8),
            TypeDef::SInt(16),
            TypeDef::Never,
            TypeDef::Bool,
        ];
        for p in &prims {
            assert_eq!(p, &TypeDef::get_primitive_type(&p.get_name()).unwrap());
        }
    }
}
