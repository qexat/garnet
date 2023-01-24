//! Garnet compiler guts.

use fnv::FnvHashMap;

pub mod ast;
pub mod parser;
pub mod typeck;

/*
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PrimType {
    I32,
    Bool,
}
*/

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
    Enum(Vec<String>),
    Named(String, Vec<Type>),
    Func(Vec<Type>, Box<Type>),
    /// The vec is the name of any generic type bindings in there
    /// 'cause we need to keep track of those apparently.
    Struct(FnvHashMap<String, Type>, Vec<Type>),
    /// Sum type.
    /// I guess this is ok?
    ///
    /// Like structs, contains a list of generic bindings.
    Sum(FnvHashMap<String, Type>, Vec<Type>),
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
    Enum(Vec<String>),
    /// N-ary type constructor.
    /// It could be Int()
    /// or List(Int)
    /// or List(T)
    Named(String, Vec<TypeId>),
    /// This type term is definitely a function
    Func(Vec<TypeId>, TypeId),
    /// This is definitely some kind of struct
    Struct(FnvHashMap<String, TypeId>),
    /// Definitely a sum type
    Sum(FnvHashMap<String, TypeId>),
    /// This is some generic type that has a name like @A
    /// AKA a type parameter.
    TypeParam(String),
}

pub fn compile(filename: &str, src: &str) -> Vec<u8> {
    let mut parser = parser::Parser::new(filename, src);
    let ast = parser.parse();
    let ast2 = ast.lower();
    typeck::typecheck(&ast2);
    vec![]
}
