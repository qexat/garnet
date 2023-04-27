//! Garnet compiler guts.
//#![deny(missing_docs)]

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

use log::*;

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
static INT: Lazy<Cx> = Lazy::new(Cx::new);

/// A primitive type.  All primitives are basically atomic
/// and can't contain other types, so.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum PrimType {
    SInt(u8),
    UnknownInt,
    Bool,
    /// erased type
    AnyPtr,
}

impl PrimType {
    fn get_name(&self) -> Cow<'static, str> {
        match self {
            PrimType::SInt(16) => Cow::Borrowed("I128"),
            PrimType::SInt(8) => Cow::Borrowed("I64"),
            PrimType::SInt(4) => Cow::Borrowed("I32"),
            PrimType::SInt(2) => Cow::Borrowed("I16"),
            PrimType::SInt(1) => Cow::Borrowed("I8"),
            PrimType::SInt(s) => panic!("Undefined integer size {}!", s),
            PrimType::UnknownInt => Cow::Borrowed("{number}"),
            PrimType::Bool => Cow::Borrowed("Bool"),
            PrimType::AnyPtr => Cow::Borrowed("AnyPtr"),
        }
    }
}

/// A concrete type that has been fully inferred
///
/// TODO someday: We should make a consistent and very good
/// name-mangling scheme for this, will make some backend stuff
/// simpler.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Type {
    Prim(PrimType),
    /// A C-like enum.
    /// For now we pretend there's no underlying values attached to
    /// the name.
    ///
    /// ... I seem to have ended up with anonymous enums somehow.
    /// Not sure how I feel about this.
    Enum(Vec<(Sym, i32)>),
    Named(Sym, Vec<Type>),
    /// param types, return types, type parameters
    Func(Vec<Type>, Box<Type>, Vec<Type>),
    /// The vec is the name of any generic type bindings in there
    /// 'cause we need to keep track of those apparently.
    Struct(BTreeMap<Sym, Type>, Vec<Type>),
    /// Sum type.
    /// I guess this is ok?
    ///
    /// Like structs, contains a list of generic bindings.
    Sum(BTreeMap<Sym, Type>, Vec<Type>),
    /// Arrays are just a type and a number.
    Array(Box<Type>, usize),
    /// A generic type parameter
    Generic(Sym),
}

impl Type {
    /// Returns the type parameters *specified by the toplevel type*.
    /// Does *not* recurse to all types below it!
    /// ...except for function args apparently.  And some other things.
    /// TODO:
    /// this is still a little cursed 'cause we infer type params in some places
    /// and it's actually really nice.  And also fiddly af so touching it
    /// breaks lots of things that currently work.
    fn get_type_params(&self) -> Vec<Sym> {
        fn helper(t: &Type, accm: &mut Vec<Sym>) {
            match t {
                Type::Prim(_) => (),
                Type::Enum(_ts) => (),
                Type::Named(_, generics) => {
                    for g in generics {
                        helper(g, accm);
                    }
                }
                Type::Func(args, rettype, typeparams) => {
                    for t in args {
                        helper(t, accm);
                    }
                    for t in typeparams {
                        helper(t, accm);
                    }
                    helper(rettype, accm)
                }
                Type::Struct(body, generics) => {
                    for (_, ty) in body {
                        helper(ty, accm);
                    }
                    for g in generics {
                        helper(g, accm);
                    }
                }
                Type::Sum(body, generics) => {
                    for (_, ty) in body {
                        helper(ty, accm);
                    }
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
                    if accm.contains(s) {
                        //todo!("This is probably always wrong 'cause type param names will shadow each other, but, uh...")
                    } else {
                        accm.push(*s)
                    }
                }
            }
        }
        let mut accm = vec![];
        helper(self, &mut accm);
        //trace!("Found type params for {:?}: {:?}", self, accm);
        accm
    }

    /// Shortcut for getting the type for an unknown int
    pub fn iunknown() -> Self {
        Self::Prim(PrimType::UnknownInt)
    }

    /// Shortcut for getting the type symbol for an int of a particular size
    pub fn isize(size: u8) -> Self {
        Self::Prim(PrimType::SInt(size))
    }

    /// Shortcut for getting the type symbol for I128
    pub fn i128() -> Self {
        Self::isize(16)
    }

    /// Shortcut for getting the type symbol for I64
    pub fn i64() -> Self {
        Self::isize(8)
    }

    /// Shortcut for getting the type symbol for I32
    pub fn i32() -> Self {
        Self::isize(4)
    }

    /// Shortcut for getting the type symbol for I16
    pub fn i16() -> Self {
        Self::isize(2)
    }

    /// Shortcut for getting the type symbol for I8
    pub fn i8() -> Self {
        Self::isize(1)
    }

    /// Shortcut for getting the type symbol for Bool
    pub fn bool() -> Self {
        Self::Prim(PrimType::Bool)
    }

    /// Shortcut for getting the type symbol for Unit
    pub fn unit() -> Self {
        Self::Named(Sym::new("Tuple"), vec![])
    }

    /// Shortcut for getting the type symbol for Never
    pub fn never() -> Self {
        Self::Named(Sym::new("Never"), vec![])
    }

    pub fn tuple(args: Vec<Self>) -> Self {
        Self::Named(Sym::new("Tuple"), args)
    }

    pub fn named(name: impl Into<String>) -> Self {
        Self::Named(Sym::new(&name.into()), vec![])
    }

    pub fn array(t: &Type, len: usize) -> Self {
        Self::Array(Box::new(t.clone()), len)
    }

    pub fn is_integer(&self) -> bool {
        match self {
            Self::Prim(PrimType::SInt(_)) => true,
            Self::Prim(PrimType::UnknownInt) => true,
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
        match s {
            "I8" => Some(Type::i8()),
            "I16" => Some(Type::i16()),
            "I32" => Some(Type::i32()),
            "I64" => Some(Type::i64()),
            "I128" => Some(Type::i128()),
            "Bool" => Some(Type::bool()),
            "Never" => Some(Type::never()),
            _ => None,
        }
    }

    /// Get a string for the name of the given type.
    pub fn get_name(&self) -> Cow<'static, str> {
        let join_types_with_commas = |lst: &[Type]| {
            let p_strs = lst.iter().map(Type::get_name).collect::<Vec<_>>();
            p_strs.join(", ")
        };
        let join_vars_with_commas = |lst: &BTreeMap<Sym, Type>| {
            let p_strs = lst
                .iter()
                .map(|(pname, ptype)| {
                    let pname = pname.val();
                    format!("{}: {}", pname, ptype.get_name())
                })
                .collect::<Vec<_>>();
            p_strs.join(", ")
        };
        match self {
            Type::Prim(p) => p.get_name(),
            Type::Enum(variants) => {
                let mut res = String::from("enum {");
                let s = variants
                    .iter()
                    .map(|(nm, vl)| format!("    {} = {},\n", INT.fetch(*nm), vl))
                    .collect::<String>();
                res += &s;
                res += "\n}\n";
                Cow::Owned(res)
            }
            Type::Named(nm, tys) if nm == &Sym::new("Tuple") => {
                if tys.is_empty() {
                    Cow::Borrowed("{}")
                } else {
                    let mut res = String::from("{");
                    res += &join_types_with_commas(tys);
                    res += "}";
                    Cow::Owned(res)
                }
            }
            Type::Named(nm, tys) => {
                let result = if tys.len() == 0 {
                    (&*nm.val()).clone()
                } else {
                    format!("{}({})", nm.val(), &join_types_with_commas(tys))
                };
                Cow::Owned(result)
            }
            Type::Func(params, rettype, typeparams) => {
                let mut t = String::from("fn");
                t += "(";
                t += &join_types_with_commas(params);

                if typeparams.len() > 0 {
                    t += "| ";
                    t += &join_types_with_commas(typeparams);
                }

                t += ")";
                let rettype_str = rettype.get_name();
                t += ": ";
                t += &rettype_str;
                Cow::Owned(t)
            }
            Type::Struct(body, _generics) => {
                let mut res = String::from("struct ");
                res += &join_vars_with_commas(body);
                res += " end";
                Cow::Owned(res)
            }
            Type::Sum(body, _generics) => {
                let mut res = String::from("sum ");
                res += &join_vars_with_commas(body);
                res += " end";
                Cow::Owned(res)
            }
            Type::Array(body, len) => {
                let inner_name = body.get_name();
                Cow::Owned(format!("{}[{}]", inner_name, len))
            }
            Type::Generic(name) => Cow::Owned(format!("@{}", name)),
        }
    }

    /// Takes two types and creates/adds to a map of substitutions
    /// from generics in the first type to the corresponding concrete
    /// types in the second types.
    ///
    /// Panics if non-generic types don't match.
    ///
    /// TODO someday: refactor with passes::type_map()?  Not sure how to make
    /// that walk over two types.
    pub fn find_substs(&self, other: &Type, substitutions: &mut BTreeMap<Sym, Type>) {
        match (self, other) {
            // Types are identical, noop.
            (s, o) if s == o => (),
            (Type::Named(n1, args1), Type::Named(n2, args2)) if n1 == n2 => {
                for (p1, p2) in args1.iter().zip(args2) {
                    p1.find_substs(p2, substitutions);
                }
            }
            (
                Type::Func(params1, rettype1, typeparams1),
                Type::Func(params2, rettype2, typeparams2),
            ) => {
                if params1.len() != params2.len() {
                    panic!("subst for function had incorrect param length")
                }
                if typeparams1.len() != typeparams2.len() {
                    panic!("subst for function had incorrect typeparam length")
                }
                if typeparams1.len() > 0 {
                    todo!()
                }
                for (p1, p2) in params1.iter().zip(params2) {
                    p1.find_substs(p2, substitutions);
                }
                rettype1.find_substs(rettype2, substitutions);
            }
            (Type::Struct(_, _), Type::Struct(_, _)) => {
                unreachable!("Actually can't happen I think, 'cause we tuple-ify everything first?")
            } /*
            (Type::Struct(body1, generics1), Type::Struct(body2, generics2)) => {
            if body1.len() != body2.len() || generics1.len() != generics2.len() {
            panic!("subst for function had incorrect body or generics length")
            }
            if
            }
             */
            (Type::Sum(body1, generics1), Type::Sum(body2, generics2)) => {
                if body1.len() != body2.len() || generics1.len() != generics2.len() {
                    panic!("subst for sum type had non-matching body or generics length")
                }
                if !body1.keys().eq(body2.keys()) {
                    panic!("subst for sum type had non-matching keys")
                }
                for ((_nm1, t1), (_nm2, t2)) in body1.iter().zip(body2) {
                    t1.find_substs(t2, substitutions);
                }

                for (p1, p2) in generics1.iter().zip(generics2) {
                    p1.find_substs(p2, substitutions);
                }
            }
            (Type::Array(t1, len1), Type::Array(t2, len2)) if len1 == len2 => {
                t1.find_substs(t2, substitutions);
            }
            (Type::Generic(nm), p2) => {
                // If we have an existing substitution, does it conflict?
                // Not 100% sure this handles generics right, but should work
                // for now.
                if let Some(other_ty) = substitutions.get(nm) {
                    if other_ty != p2 {
                        panic!("Conflicting subtitution");
                    }
                } else {
                    substitutions.insert(*nm, p2.clone());
                }
            }
            // Types are not identical, panic
            _ => panic!("Cannot substitute {:?} into {:?}", other, self),
        }
    }

    /// Takes a type and a map of substitutions and swaps out any generics
    /// with the substituted types.
    ///
    /// Panics if a generic type has no substitution.
    ///
    /// TODO someday: refactor with passes::type_map()?
    pub fn apply_substs(&self, substs: &BTreeMap<Sym, Type>) -> Type {
        match self {
            Type::Func(params1, rettype1, typeparams1) => {
                let new_params = params1.iter().map(|p1| p1.apply_substs(substs)).collect();
                let new_rettype = rettype1.apply_substs(substs);
                if typeparams1.len() > 0 {
                    todo!("Hsfjkdslfjs");
                }
                Type::Func(new_params, Box::new(new_rettype), vec![])
            }
            Type::Named(n1, args1) => {
                let new_args = args1.iter().map(|p1| p1.apply_substs(substs)).collect();
                Type::Named(*n1, new_args)
            }
            Type::Struct(_, _) => unreachable!("see other unreachable in substitute()"),
            Type::Sum(body, generics) => {
                let new_body = body
                    .iter()
                    .map(|(nm, ty)| (*nm, ty.apply_substs(substs)))
                    .collect();
                let new_generics = generics.iter().map(|p1| p1.apply_substs(substs)).collect();
                Type::Sum(new_body, new_generics)
            }
            Type::Array(body, len) => Type::Array(Box::new(body.apply_substs(substs)), *len),
            Type::Generic(nm) => substs
                .get(&nm)
                .unwrap_or_else(|| panic!("No substitution found for generic named {}!", nm))
                .to_owned(),
            Type::Prim(_) => self.clone(),
            Type::Enum(_) => self.clone(),
        }
    }
}

/// An interned string of some kind, any kind.
#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Sym(pub usize);

/// Required for interner interface.
impl From<usize> for Sym {
    fn from(i: usize) -> Sym {
        Sym(i)
    }
}

/// Required for interner interface.
impl From<Sym> for usize {
    fn from(i: Sym) -> usize {
        i.0
    }
}

impl Sym {
    /// Intern a new string.
    pub fn new(s: impl AsRef<str>) -> Self {
        INT.intern(s)
    }
    /// Get the underlying string
    pub fn val(&self) -> Arc<String> {
        INT.fetch(*self)
    }
}

impl fmt::Debug for Sym {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Sym({}, {:?})", self.0, self.val())
    }
}

impl fmt::Display for Sym {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.val())
    }
}

/*
/// A path of modules/structs/whatever
/// `foo.bar.bop`
pub struct Path {
    /// The last part of the path, must always exist.
    pub name: Sym,
    /// The rest of the (possibly empty) path.
    /// TODO: Whether this must be absolute or could be relative is currently undefined.
    pub path: Vec<Sym>,
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
    syms: intern::Interner<Sym, String>,
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
    fn intern(&self, s: impl AsRef<str>) -> Sym {
        self.syms.intern(&s.as_ref().to_owned())
    }

    /// Get the string for a variable symbol
    fn fetch(&self, s: Sym) -> Arc<String> {
        self.syms.fetch(s)
    }

    /// Generate a new unique symbol including the given string
    /// Useful for some optimziations and intermediate names and such.
    ///
    /// Starts with a `!` which currently cannot appear in any identifier.
    pub fn gensym(&self, s: &str) -> Sym {
        let sym = format!("!{}_{}", s, self.syms.count());
        self.intern(sym)
    }
}

/// Main driver function.
/// Compile a given source string to Rust source code, or return an error.
/// TODO: Better parser errors with locations
///
/// Parse -> lower to IR -> run transformation passes
/// -> typecheck -> more passes -> codegen
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
    info!("HIR from AST lowering:\n{}", &hir);
    let hir = passes::run_passes(hir);
    let tck = &mut typeck::typecheck(&hir)?;
    let hir = passes::run_typechecked_passes(hir, tck);
    info!("HIR after transform passes:\n{}", &hir);
    Ok(backend::output(backend, &hir, tck))
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
        let def = Type::Func(args, Box::new(Type::i32()), vec![]);
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
