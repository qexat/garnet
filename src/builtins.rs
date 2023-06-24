//! A place for us to stick compiler builtins and the metadata they need,
//! instead of having them scattered all over.

use std::collections::BTreeMap;

use once_cell::sync::Lazy;

use crate::backend::Backend;
use crate::*;

/// A single built-in value that is just included into the generated program.
pub struct Builtin {
    /// The name of the value
    pub name: Sym,
    /// The value's type, used for typechecking
    pub sig: Type,
    /// The implmentation of the builtin as raw code for each particular
    /// backend.
    pub code: BTreeMap<Backend, &'static str>,
}

pub static BUILTINS: Lazy<Vec<Builtin>> = Lazy::new(Builtin::all);

impl Builtin {
    /// A function that returns all the compiler builtin info.  Just
    /// use the `BUILTINS` global instead, this is basically here
    /// to initialize it.
    fn all() -> Vec<Builtin> {
        let rust_println = r#"fn __println(x: i32) {
    println!("{}", x);
}"#;
        let rust_println_bool = r#"
fn __println_bool(x: bool) {
    println!("{}", x);
}"#;

        let rust_println_i64 = r#"
fn __println_i64(x: i64) {
    println!("{}", x);
}"#;
        let rust_println_i16 = r#" 
fn __println_i16(x: i16) {
    println!("{}", x);
}"#;
        vec![
            Builtin {
                name: Sym::new("__println"),
                sig: Type::Func(vec![Type::i32()], Box::new(Type::unit()), vec![]),
                code: BTreeMap::from([(Backend::Null, ""), (Backend::Rust, rust_println)]),
            },
            Builtin {
                name: Sym::new("__println_bool"),
                sig: Type::Func(vec![Type::bool()], Box::new(Type::unit()), vec![]),
                code: BTreeMap::from([(Backend::Null, ""), (Backend::Rust, rust_println_bool)]),
            },
            Builtin {
                name: Sym::new("__println_i64"),
                sig: Type::Func(vec![Type::i64()], Box::new(Type::unit()), vec![]),
                code: BTreeMap::from([(Backend::Null, ""), (Backend::Rust, rust_println_i64)]),
            },
            Builtin {
                name: Sym::new("__println_i16"),
                sig: Type::Func(vec![Type::i16()], Box::new(Type::unit()), vec![]),
                code: BTreeMap::from([(Backend::Null, ""), (Backend::Rust, rust_println_i16)]),
            },
        ]
    }
}
