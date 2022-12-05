//! Garnet compiler guts.

use std::collections::{HashMap, HashSet};

pub mod ast;
pub mod parser;
pub mod typeck;

/// A concrete type that has been fully inferred
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Type {
    Named(String, Vec<Type>),
    Func(Vec<Type>, Box<Type>),
    Struct(HashMap<String, Type>),
    /// A generic type parameter
    Generic(String),
}

impl Type {
    fn get_primitive_type(s: &str) -> Option<Type> {
        match s {
            "I32" => Some(Type::Named("I32".to_string(), vec![])),
            "Bool" => Some(Type::Named("Bool".to_string(), vec![])),
            //"Never" => Some(TypeInfo::Never),
            _ => None,
        }
    }

    fn get_generic_names(&self) -> HashSet<String> {
        fn helper(t: &Type, accm: &mut HashSet<String>) {
            match t {
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
                Type::Struct(body) => {
                    for (_, ty) in body {
                        helper(ty, accm);
                    }
                }
                Type::Generic(s) => {
                    accm.insert(s.clone());
                }
            }
        }
        let mut accm = HashSet::new();
        helper(self, &mut accm);
        accm
    }
}

/// A identifier to uniquely refer to our type terms
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct TypeId(usize);

/// Information about a type term
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TypeInfo {
    /// No information about the type of this type term
    Unknown,
    /// This type term is the same as another type term
    Ref(TypeId),
    /// N-ary type constructor.
    /// It could be Int()
    /// or List(Int)
    /// or List(T)
    Named(String, Vec<TypeId>),
    /// This type term is definitely a function
    Func(Vec<TypeId>, TypeId),
    /// This is definitely some kind of struct
    Struct(HashMap<String, TypeId>),
    /// This is some generic type that has a name like @A
    /// AKA a type parameter.
    TypeParam(String),
}

impl TypeInfo {
    fn _get_primitive_type(s: &str) -> Option<TypeInfo> {
        match s {
            "I32" => Some(TypeInfo::Named("I32".to_string(), vec![])),
            "Bool" => Some(TypeInfo::Named("Bool".to_string(), vec![])),
            //"Never" => Some(TypeInfo::Never),
            _ => None,
        }
    }

    fn _generic_name(&self) -> Option<&str> {
        match self {
            TypeInfo::TypeParam(s) => Some(s),
            _ => None,
        }
    }
}

pub fn compile(filename: &str, src: &str) -> Vec<u8> {
    //typecheck2();
    let mut parser = parser::Parser::new(filename, src);
    let ast = parser.parse();
    typeck::typecheck(&ast);
    //let res = format!("AST:\n{:#?}", ast);
    //res.as_bytes().to_owned()
    vec![]
}
