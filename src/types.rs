//! Type types, which will tend to need to be everywhere.
//! Not part of ast.rs or hir.rs because they're shared across both,
//! and kinda too big/specialized go to in lib.rs

use std::borrow::Cow;
use std::collections::BTreeMap;

use crate::Sym;

/// A primitive type.  All primitives are basically atomic
/// and can't contain other types, so.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum PrimType {
    /// size in bytes, is-signed
    Int(u8, bool),
    UnknownInt,
    Bool,
    /// erased type, currently unused
    AnyPtr,
}

impl PrimType {
    pub(crate) fn get_name(&self) -> Cow<'static, str> {
        match self {
            PrimType::Int(16, true) => Cow::Borrowed("I128"),
            PrimType::Int(8, true) => Cow::Borrowed("I64"),
            PrimType::Int(4, true) => Cow::Borrowed("I32"),
            PrimType::Int(2, true) => Cow::Borrowed("I16"),
            PrimType::Int(1, true) => Cow::Borrowed("I8"),
            PrimType::Int(16, false) => Cow::Borrowed("U128"),
            PrimType::Int(8, false) => Cow::Borrowed("U64"),
            PrimType::Int(4, false) => Cow::Borrowed("U32"),
            PrimType::Int(2, false) => Cow::Borrowed("U16"),
            PrimType::Int(1, false) => Cow::Borrowed("U8"),
            PrimType::Int(s, is_signed) => {
                let prefix = if *is_signed { "I" } else { "U" };
                panic!("Undefined integer size {}{}!", prefix, s);
            }
            PrimType::UnknownInt => Cow::Borrowed("{number}"),
            PrimType::Bool => Cow::Borrowed("Bool"),
            PrimType::AnyPtr => Cow::Borrowed("AnyPtr"),
        }
    }
}

/// A concrete type that is fully specified but may have type parameters.
///
/// TODO someday: We should make a consistent and very good
/// name-mangling scheme for types, will make some backend stuff
/// simpler.  Also see passes::generate_type_name.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Type {
    /// Primitive type with no subtypes
    Prim(PrimType),
    /// Never type, the type of an infinite loop
    Never,
    /// A C-like enum, and the integer values it corresponds to
    Enum(Vec<(Sym, i32)>),
    /// A nominal type of some kind; may be built-in (like Tuple)
    /// or user-defined.
    Named(Sym, Vec<Type>),
    /// A function pointer.
    ///
    /// The contents are arg types, return types, type parameters
    Func(Vec<Type>, Box<Type>, Vec<Type>),
    /// An anonymous struct.  The vec is type parameters.
    Struct(BTreeMap<Sym, Type>, Vec<Type>),
    /// Sum type.
    ///
    /// Like structs, contains a list of type parameters.
    Sum(BTreeMap<Sym, Type>, Vec<Type>),
    /// Arrays are just a type and a number.
    Array(Box<Type>, usize),
    /// Unique borrow
    Uniq(Box<Type>),
}

impl Type {
    /// Returns the type parameters *specified by the toplevel type*.
    /// Does *not* recurse to all types below it!
    /// ...except for function args apparently.  And some other things.
    /// TODO:
    /// this is still a little cursed 'cause we infer type params in some places
    /// and it's actually really nice.  And also fiddly af so touching it
    /// breaks lots of things that currently work.
    /// plz unfuck this, it's not THAT hard.
    pub(crate) fn get_toplevel_type_params(&self) -> Vec<Sym> {
        fn get_toplevel_names(t: &[Type]) -> Vec<Sym> {
            t.iter()
                .flat_map(|t| match t {
                    Type::Named(nm, generics) => {
                        assert!(generics.len() == 0);
                        Some(*nm)
                    }
                    _ => None,
                })
                .collect()
        }

        let params = match self {
            Type::Prim(_) => vec![],
            Type::Never => vec![],
            Type::Enum(_ts) => vec![],
            Type::Named(_, typeparams) => get_toplevel_names(typeparams),
            Type::Func(_args, _rettype, typeparams) => get_toplevel_names(typeparams),
            Type::Struct(_body, typeparams) => get_toplevel_names(typeparams),
            Type::Sum(_body, typeparams) => get_toplevel_names(typeparams),
            Type::Array(_ty, _size) => {
                // BUGGO: What to do here?????
                // Arrays and ptrs kiiiiinda have type params, but only
                // one???
                vec![]
            }
            Type::Uniq(_ty) => {
                // BUGGO: What to do here?????
                vec![]
            }
        };
        params
    }

    /// Shortcut for getting the type for an unknown int
    pub fn iunknown() -> Self {
        Self::Prim(PrimType::UnknownInt)
    }

    /// Shortcut for getting the Type for a signed int of a particular size
    pub fn isize(size: u8) -> Self {
        Self::Prim(PrimType::Int(size, true))
    }

    /// Shortcut for getting the Type for an unsigned signed int of a particular size
    pub fn usize(size: u8) -> Self {
        Self::Prim(PrimType::Int(size, false))
    }

    /// Shortcut for getting the Type for I128
    pub fn i128() -> Self {
        Self::isize(16)
    }

    /// Shortcut for getting the Type for I64
    pub fn i64() -> Self {
        Self::isize(8)
    }

    /// Shortcut for getting the Type for I32
    pub fn i32() -> Self {
        Self::isize(4)
    }

    /// Shortcut for getting the type for I16
    pub fn i16() -> Self {
        Self::isize(2)
    }

    /// Shortcut for getting the type for I8
    pub fn i8() -> Self {
        Self::isize(1)
    }

    /// Shortcut for getting the Type for U128
    pub fn u128() -> Self {
        Self::usize(16)
    }

    /// Shortcut for getting the Type for U64
    pub fn u64() -> Self {
        Self::usize(8)
    }

    /// Shortcut for getting the Type for U32
    pub fn u32() -> Self {
        Self::usize(4)
    }

    /// Shortcut for getting the type for U16
    pub fn u16() -> Self {
        Self::usize(2)
    }

    /// Shortcut for getting the type for U8
    pub fn u8() -> Self {
        Self::usize(1)
    }

    /// Shortcut for getting the type for Bool
    pub fn bool() -> Self {
        Self::Prim(PrimType::Bool)
    }

    /// Shortcut for getting the type for Unit
    pub fn unit() -> Self {
        Self::Named(Sym::new("Tuple"), vec![])
    }

    /// Shortcut for getting the type for Never
    pub fn never() -> Self {
        Self::Named(Sym::new("Never"), vec![])
    }

    /// Create a Tuple with the given values
    pub fn tuple(args: Vec<Self>) -> Self {
        Self::Named(Sym::new("Tuple"), args)
    }

    /// Used in some tests
    pub fn array(t: &Type, len: usize) -> Self {
        Self::Array(Box::new(t.clone()), len)
    }

    /// Shortcut for a named type with no type params
    // pub fn named0(s: impl AsRef<str>) -> Self {
    pub fn named0(s: Sym) -> Self {
        Type::Named(s, vec![])
    }

    /// Turns a bunch of Named types into a list of symbols.
    /// Panics if it encounters a different type of type.
    pub fn detype_names(ts: &[Type]) -> Vec<Sym> {
        fn f(t: &Type) -> Sym {
            match t {
                Type::Named(nm, generics) => {
                    assert!(generics.len() == 0);
                    *nm
                }
                _ => panic!(
                    "Tried to get a Named type out of a {:?}, should never happen",
                    t
                ),
            }
        }
        ts.iter().map(f).collect()
    }

    pub fn function(params: &[Type], rettype: &Type, generics: &[Type]) -> Self {
        Type::Func(
            Vec::from(params),
            Box::new(rettype.clone()),
            Vec::from(generics),
        )
    }

    fn _is_integer(&self) -> bool {
        match self {
            Self::Prim(PrimType::Int(_, _)) => true,
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
            "U8" => Some(Type::u8()),
            "U16" => Some(Type::u16()),
            "U32" => Some(Type::u32()),
            "U64" => Some(Type::u64()),
            "U128" => Some(Type::u128()),
            "Bool" => Some(Type::bool()),
            "Never" => Some(Type::never()),
            _ => None,
        }
    }

    /// Get a string for the name of the given type in valid Garnet syntax.
    /// Or at least should be.
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
            Type::Never => Cow::Borrowed("!"),
            Type::Enum(variants) => {
                let mut res = String::from("enum ");
                let s = variants
                    .iter()
                    .map(|(nm, vl)| format!("    {} = {},\n", nm.val(), vl))
                    .collect::<String>();
                res += &s;
                res += "\nend\n";
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
                t += " ";
                t += &rettype_str;
                Cow::Owned(t)
            }
            Type::Struct(body, generics) => {
                let mut res = String::from("struct");
                if generics.is_empty() {
                    res += " "
                } else {
                    res += "(";
                    res += &join_types_with_commas(generics);
                    res += ") ";
                }
                res += &join_vars_with_commas(body);
                res += " end";
                Cow::Owned(res)
            }
            Type::Sum(body, generics) => {
                let mut res = String::from("sum");
                if generics.is_empty() {
                    res += " "
                } else {
                    res += "(";
                    res += &join_types_with_commas(generics);
                    res += ") ";
                }
                res += &join_vars_with_commas(body);
                res += " end";
                Cow::Owned(res)
            }
            Type::Array(body, len) => {
                let inner_name = body.get_name();
                Cow::Owned(format!("[{}]{}", len, inner_name))
            }
            Type::Uniq(ty) => {
                let inner = ty.get_name();
                Cow::Owned(format!("&{}", inner))
            }
        }
    }
}

/// A function type signature.
///
/// This is heavily used in parsing and such, but is even more
/// heavily used in type checking and symbol resolution, so it
/// goes here.
#[derive(Debug, Clone, PartialEq)]
pub struct Signature {
    /// Parameters
    pub params: Vec<(Sym, Type)>,
    /// Return type
    pub rettype: Type,
    /// Type parameters
    pub typeparams: Vec<Sym>,
}

impl Signature {
    /// Returns a lambda typedef representing the signature
    pub fn to_type(&self) -> Type {
        let paramtypes = self.params.iter().map(|(_nm, ty)| ty.clone()).collect();
        let typeparams = self.typeparams.iter().map(|nm| Type::named0(*nm)).collect();
        Type::Func(paramtypes, Box::new(self.rettype.clone()), typeparams)
    }

    /// Get all the generic params out of this function sig
    pub fn type_params(&self) -> Vec<Sym> {
        self.to_type().get_toplevel_type_params()
    }

    /// Returns a string containing just the params and rettype bits of the sig
    pub fn to_name(&self) -> String {
        let names: Vec<_> = self
            .params
            .iter()
            .map(|(s, t)| format!("{} {}", &*s.val(), t.get_name()))
            .collect();
        let args = names.join(", ");

        let typenames: Vec<_> = self
            .typeparams
            .iter()
            .map(|t| (t.val().as_str()).to_string())
            .collect();
        let typeargs = typenames.join(", ");
        format!("(|{}| {}) {}", typeargs, args, self.rettype.get_name())
    }
}
