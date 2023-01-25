//! Garnet compiler guts.
//#![deny(missing_docs)]

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

pub mod ast;
pub mod backend;
pub mod format;
pub mod hir;
pub mod intern;
pub mod parser;
pub mod passes;
pub mod typeck;

#[cfg(test)]
pub(crate) mod testutil;

use once_cell::sync::Lazy;
/// The interner.  It's the ONLY part we have to actually
/// carry around anywhere, so I'm experimenting with making
/// it a global.  Seems to work pretty okay.
pub static INT: Lazy<Cx> = Lazy::new(Cx::new);

/// A concrete type that has been fully inferred
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Type {
    //Primitive(PrimType),
    /// A C-like enum.
    /// For now we pretend there's no underlying values attached to
    /// the name.
    ///
    /// ... I seem to have ended up with anonymous enums somehow.
    /// Not sure how I feel about this.
    Enum(Vec<(VarSym, i32)>),
    Named(String, Vec<Type>),
    Func(Vec<Type>, Box<Type>),
    /// The vec is the name of any generic type bindings in there
    /// 'cause we need to keep track of those apparently.
    Struct(BTreeMap<VarSym, Type>, Vec<Type>),
    /// Sum type.
    /// I guess this is ok?
    ///
    /// Like structs, contains a list of generic bindings.
    Sum(BTreeMap<VarSym, Type>, Vec<Type>),
    /// Arrays are just a type and a number.
    Array(Box<Type>, usize),
    /// A generic type parameter
    Generic(String),
}

impl Type {
    /// Search through the type and return any generic types in it.
    fn collect_generic_names(&self) -> Vec<String> {
        fn helper(t: &Type, accm: &mut Vec<String>) {
            match t {
                //Type::Primitive(_) => (),
                Type::Enum(_ts) => (),
                Type::Named(_, ts) => {
                    for t in ts {
                        helper(t, accm);
                    }
                }
                Type::Func(args, rettype) => {
                    for t in args {
                        helper(t, accm);
                    }
                    helper(rettype, accm)
                }
                Type::Struct(body, generics) => {
                    for (_, ty) in body {
                        helper(ty, accm);
                    }
                    // This makes me a little uneasy
                    // 'cause I thiiiink the whole point of the names on
                    // the struct is to list *all* the generic names in it...
                    // but we could have nested definitions
                    // like Foo(@T) ( Bar(@G) ( {@T, @G} ) )
                    for ty in generics {
                        helper(ty, accm);
                    }
                }
                // Just like structs
                Type::Sum(body, generics) => {
                    for (_, ty) in body {
                        helper(ty, accm);
                    }
                    for ty in generics {
                        helper(ty, accm);
                    }
                }
                Type::Array(ty, _size) => {
                    helper(ty, accm);
                }
                Type::Generic(s) => {
                    // Deduplicating these things while maintaining ordering
                    // is kinda screwy.
                    // This works, it's just, yanno, also O(n^2)
                    // Could use a set to check membership , but fuckit for now.
                    if !accm.contains(s) {
                        accm.push(s.clone());
                    }
                }
            }
        }
        let mut accm = vec![];
        helper(self, &mut accm);
        accm
    }

    /// Returns the type parameters *specified by the toplevel type*.
    /// Does *not* recurse to all types below it!
    /// ...except for function args apparently.
    fn get_type_params(&self) -> Vec<String> {
        fn helper(t: &Type, accm: &mut Vec<String>) {
            match t {
                //Type::Primitive(_) => (),
                Type::Enum(_ts) => (),
                Type::Named(_, generics) => {
                    for g in generics {
                        helper(g, accm);
                    }
                }
                Type::Func(args, rettype) => {
                    for t in args {
                        helper(t, accm);
                    }
                    helper(rettype, accm)
                }
                Type::Struct(_body, generics) => {
                    for g in generics {
                        helper(g, accm);
                    }
                }
                Type::Sum(_body, generics) => {
                    for g in generics {
                        helper(g, accm);
                    }
                }
                Type::Array(ty, _size) => {
                    helper(ty, accm);
                }
                Type::Generic(s) => {
                    // Deduplicating these things while maintaining ordering
                    // is kinda screwy.
                    // This works, it's just, yanno, also O(n^2)
                    // Could use a set to check membership , but fuckit for now.
                    if !accm.contains(s) {
                        accm.push(s.clone())
                    }
                }
            }
        }
        let mut accm = vec![];
        helper(self, &mut accm);
        println!("Found type params for {:?}: {:?}", self, accm);
        accm
    }

    /// Shortcut for getting the type for an unknown int
    pub fn iunknown() -> Self {
        todo!()
        //Self::UnknownInt
    }

    /// Shortcut for getting the type symbol for an int of a particular size
    pub fn isize(size: u8) -> Self {
        todo!()
        //Self::SInt(size)
    }

    /// Shortcut for getting the type symbol for I128
    pub fn i128() -> Self {
        todo!()
        //Self::SInt(16)
    }

    /// Shortcut for getting the type symbol for I64
    pub fn i64() -> Self {
        todo!()
        //Self::SInt(8)
    }

    /// Shortcut for getting the type symbol for I32
    pub fn i32() -> Self {
        todo!()
        //Self::SInt(4)
    }

    /// Shortcut for getting the type symbol for I16
    pub fn i16() -> Self {
        //Self::SInt(2)
        todo!()
    }

    /// Shortcut for getting the type symbol for I8
    pub fn i8() -> Self {
        todo!()
        //Self::SInt(1)
    }

    /// Shortcut for getting the type symbol for Bool
    pub fn bool() -> Self {
        Self::Named("Bool".to_string(), vec![])
    }

    /// Shortcut for getting the type symbol for Unit
    pub fn unit() -> Self {
        Self::Named("Tuple".to_string(), vec![])
    }

    /// Shortcut for getting the type symbol for Never
    pub fn never() -> Self {
        todo!()
        //Self::Never
    }

    pub fn tuple(args: Vec<Self>) -> Self {
        Self::Named("Tuple".into(), args)
    }

    pub fn named(name: impl Into<String>) -> Self {
        Self::Named(name.into(), vec![])
    }

    pub fn is_integer(&self) -> bool {
        match self {
            Self::Named(nm, _) if nm == "I32" => true,
            _ => false,
        }
    }

    /// Takes a string and matches it against the builtin/
    /// primitive types, returning the appropriate `TypeDef`
    ///
    /// If it is not a built-in type, or is a compound such
    /// as a tuple or struct, returns None.
    ///
    /// The effective inverse of `TypeDef::get_name()`.  Compound
    /// types take more parsing to turn from strings to `TypeDef`'s
    /// and are handled in `parser::parse_type()`.
    pub fn get_primitive_type(s: &str) -> Option<Type> {
        todo!()
        /*
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
            */
    }

    /// Get a string for the name of the given type.
    pub fn get_name(&self) -> Cow<'static, str> {
        todo!()
        /*
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
            let join_generics_with_commas = |lst: &[TypeConstraint]| {
                // slice.join() alone won't work here 'cause we gotta map the thing
                // first, which makes an iterator, not a slice.
                // Apparently we want std::iter::IntersperseWith, but, that's
                // not stable yet.  Odd!
                let strs = lst
                    .iter()
                    .map(|name| (&*INT.fetch(*name)).to_owned())
                    .collect::<Vec<_>>();
                strs.join(", ")
            };
            match self {
                Type::SInt(16) => Cow::Borrowed("I128"),
                Type::SInt(8) => Cow::Borrowed("I64"),
                Type::SInt(4) => Cow::Borrowed("I32"),
                Type::SInt(2) => Cow::Borrowed("I16"),
                Type::SInt(1) => Cow::Borrowed("I8"),
                Type::SInt(s) => panic!("Undefined integer size {}!", s),
                Type::UnknownInt => Cow::Borrowed("{number}"),
                Type::Bool => Cow::Borrowed("Bool"),
                Type::Never => Cow::Borrowed("Never"),
                Type::Tuple(v) => {
                    if v.is_empty() {
                        Cow::Borrowed("{}")
                    } else {
                        let mut res = String::from("{");
                        res += &join_types_with_commas(v);
                        res += "}";
                        Cow::Owned(res)
                    }
                }
                Type::Lambda {
                    generics,
                    params,
                    rettype,
                } => {
                    let mut t = String::from("fn");
                    if generics.len() > 0 {
                        t += "[";
                        t += &join_generics_with_commas(generics);
                        t += "]";
                    }
                    t += "(";
                    t += &join_types_with_commas(params);

                    t += ")";
                    let ret_def = INT.fetch_type(*rettype);
                    match &*ret_def {
                        Type::Tuple(x) if x.is_empty() => {
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
                Type::Struct { fields, .. } => {
                    let mut res = String::from("struct {");
                    res += &join_vars_with_commas(fields);
                    res += "}";
                    Cow::Owned(res)
                }
                Type::Enum { variants } => {
                    let mut res = String::from("enum {");
                    let s = variants
                        .iter()
                        .map(|(nm, vl)| format!("    {} = {},\n", INT.fetch(*nm), vl))
                        .collect::<String>();
                    res += &s;
                    res += "\n}\n";
                    Cow::Owned(res)
                }
                Type::NamedTypeVar(vsym) => Cow::Owned((&*INT.fetch(*vsym)).clone()),
            }
        */
    }
}

/*
/// The interned name of a type
#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TypeSym(pub usize);

impl fmt::Debug for TypeSym {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "TypeSym({}, {:?})", self.0, INT.fetch_type(*self))
    }
}

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

impl TypeSym {
    /// Get the value represented by the interned symbol.
    ///
    /// Can't return a reference, annoyingly, as it
    /// would result in "returning a value to a local"
    fn val(&self) -> Arc<TypeDef> {
        INT.fetch_type(*self)
    }
}
*/

/// The interned name of a variable/value
#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
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

impl fmt::Debug for VarSym {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "VarSym({}, {:?})", self.0, INT.fetch(*self))
    }
}

impl VarSym {
    /// Same as TypeSym::val()
    fn val(&self) -> Arc<String> {
        INT.fetch(*self)
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

/*
/// A complete-ish description of a type.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeDef<T = TypeSym> {
    /// Signed integer with the given number of bytes
    SInt(u8),
    /// An integer of unknown size, from an integer literal
    UnknownInt,
    /// Boolean, obv's
    Bool,
    /// Tuple.  The types inside it may or may not be fully known I guess
    Tuple(Vec<T>),
    /// Never is a real type, I guess!
    Never,
    /// The type of a lambda is its signature.
    Lambda {
        generics: Vec<TypeConstraint>,
        params: Vec<T>,
        rettype: T,
    },
    /// A struct.
    Struct { fields: BTreeMap<VarSym, T> },
    Enum {
        /// TODO: For now the only size of an enum is i32.
        variants: Vec<(VarSym, i32)>,
    },
    /// These are type variables that are explicitly declared by the user.
    NamedTypeVar(VarSym),
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
        let join_generics_with_commas = |lst: &[TypeConstraint]| {
            // slice.join() alone won't work here 'cause we gotta map the thing
            // first, which makes an iterator, not a slice.
            // Apparently we want std::iter::IntersperseWith, but, that's
            // not stable yet.  Odd!
            let strs = lst
                .iter()
                .map(|name| (&*INT.fetch(*name)).to_owned())
                .collect::<Vec<_>>();
            strs.join(", ")
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
            TypeDef::Lambda {
                generics,
                params,
                rettype,
            } => {
                let mut t = String::from("fn");
                if generics.len() > 0 {
                    t += "[";
                    t += &join_generics_with_commas(generics);
                    t += "]";
                }
                t += "(";
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
            TypeDef::NamedTypeVar(vsym) => Cow::Owned((&*INT.fetch(*vsym)).clone()),
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

    /// Returns the field type if this is a struct, None
    /// if this type is not a struct or does not have that field.
    pub fn struct_field(&self, field: VarSym) -> Option<TypeSym> {
        match self {
            TypeDef::Struct { fields } => fields.get(&field).copied(),
            _ => None,
        }
    }

    /// Returns a struct containing numbered fields for the tuple if this is
    /// a tuple, or None otherwise
    pub fn struct_from_tuple(&self) -> Option<TypeSym> {
        match self {
            TypeDef::Tuple(fields) => {
                let body_pairs = fields
                    .iter()
                    .enumerate()
                    .map(|(i, ty)| (INT.intern(format!("_{}", i)), *ty))
                    .collect();
                Some(INT.intern_type(&TypeDef::Struct { fields: body_pairs }))
            }
            _ => None,
        }
    }
}
*/

/// Interner context.
///
/// Really this is just an interner for symbols now, and
/// the original plan of bundling it up into a special context
/// type for each step of compilation hasn't really worked out.
#[derive(Debug)]
pub struct Cx {
    /// Interned var names
    syms: intern::Interner<VarSym, String>,
    //files: cs::files::SimpleFiles<String, String>,
}

impl Cx {
    pub fn new() -> Self {
        Cx {
            syms: intern::Interner::new(),
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

    /*
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
    */

    /// Generate a new unique symbol including the given string
    /// Useful for some optimziations and intermediate names and such.
    ///
    /// Starts with a `!` which currently cannot appear in any identifier.
    pub fn gensym(&self, s: &str) -> VarSym {
        let sym = format!("!{}_{}", s, self.syms.count());
        self.intern(sym)
    }
}

/// Main driver function.
/// Compile a given source string to Rust source code, or return an error.
/// TODO: Parser errors?
///
/// Parse -> lower to IR -> run transformation passes
/// -> typecheck -> compile to wasm
pub fn try_compile(
    filename: &str,
    src: &str,
    backend: backend::Backend,
) -> Result<Vec<u8>, typeck::TypeError> {
    let ast = {
        let mut parser = parser::Parser::new(filename, src);
        parser.parse()
    };
    let hir = hir::lower(&ast);
    let hir = passes::run_passes(hir);
    let tck = typeck::typecheck(&hir)?;
    Ok(backend::output(backend, &hir, &tck))
}

/// For when we don't care about catching results
pub fn compile(filename: &str, src: &str, backend: backend::Backend) -> Vec<u8> {
    try_compile(filename, src, backend).unwrap_or_else(|e| panic!("Type check error: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Make sure outputting a lambda's name gives us something we expect.
    #[test]
    fn check_name_format() {
        let args = vec![Type::i32(), Type::bool()];
        let def = Type::Func(args, Box::new(Type::i32()));
        let gotten_name = def.get_name();
        let desired_name = "fn(I32, Bool): I32";
        assert_eq!(&gotten_name, desired_name);
    }

    /// Make sure that `Type::get_name()` and `TypeDef::get_primitive_type()`
    /// are inverses for all primitive types.
    ///
    /// TODO: This currently requires that we have a list of all primitive types here,
    /// which is somewhat annoying.
    #[test]
    fn check_primitive_names() {
        let prims = vec![
            Type::i8(),
            Type::i16(),
            Type::i32(),
            Type::i64(),
            Type::i128(),
            Type::never(),
            Type::bool(),
        ];
        for p in &prims {
            assert_eq!(p, &Type::get_primitive_type(&p.get_name()).unwrap());
        }
    }
}
