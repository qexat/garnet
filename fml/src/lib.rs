//! Garnet compiler guts.

pub mod ast;
pub mod parser;
pub mod typeck;

/// A concrete type that has been fully inferred
#[derive(Debug)]
pub enum Type {
    Num,
    Bool,
    Func(Vec<Type>, Box<Type>),
}

/// A identifier to uniquely refer to our type terms
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct TypeId(usize);

/// Information about a type term
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum TypeInfo {
    // No information about the type of this type term
    Unknown,
    // This type term is the same as another type term
    Ref(TypeId),
    // This type term is definitely a number
    Num,
    // This type term is definitely a bool
    Bool,
    // This type term is definitely a function
    Func(Vec<TypeId>, TypeId),
    // This is some generic type that has a name like @A
    NamedGeneric(String),
}

impl TypeInfo {
    fn get_primitive_type(s: &str) -> Option<TypeInfo> {
        match s {
            "I32" => Some(TypeInfo::Num),
            "Bool" => Some(TypeInfo::Bool),
            //"Never" => Some(TypeInfo::Never),
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
