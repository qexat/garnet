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

        let band = r#"
fn __band(x: i32, y: i32) -> i32 {
    x & y
}"#;
        let bor = r#"
fn __bor(x: i32, y: i32) -> i32 {
    x | y
}"#;
        let bxor = r#"
fn __bxor(x: i32, y: i32) -> i32 {
    x ^ y
}"#;
        let bnot = r#"
fn __bnot(x: i32) -> i32 {
    !x
}"#;

        let rshift = r#"
fn __rshift(x: i32, i: i32) -> i32 {
    x >> i
}"#; 
        let lshift = r#"
fn __lshift(x: i32, i: i32) -> i32 {
    x << i
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
            // names come from luajit I suppose
            Builtin {
                name: Sym::new("__band"),
                sig: Type::Func(vec![Type::i32(), Type::i32()], Box::new(Type::i32()), vec![]),
                code: BTreeMap::from([(Backend::Null, ""), (Backend::Rust, band)]),
            },
            Builtin {
                name: Sym::new("__bor"),
                sig: Type::Func(vec![Type::i32(), Type::i32()], Box::new(Type::i32()), vec![]),
                code: BTreeMap::from([(Backend::Null, ""), (Backend::Rust, bor)]),
            },
            Builtin {
                name: Sym::new("__bxor"),
                sig: Type::Func(vec![Type::i32(), Type::i32()], Box::new(Type::i32()), vec![]),
                code: BTreeMap::from([(Backend::Null, ""), (Backend::Rust, bxor)]),
            },
            Builtin {
                name: Sym::new("__bnot"),
                sig: Type::Func(vec![Type::i32()], Box::new(Type::i32()), vec![]),
                code: BTreeMap::from([(Backend::Null, ""), (Backend::Rust, bnot)]),
            },
            Builtin {
                name: Sym::new("__rshift"),
                sig: Type::Func(vec![Type::i32(), Type::i32()], Box::new(Type::i32()), vec![]),
                code: BTreeMap::from([(Backend::Null, ""), (Backend::Rust, rshift)]),
            },
            Builtin {
                name: Sym::new("__lshift"),
                sig: Type::Func(vec![Type::i32(), Type::i32()], Box::new(Type::i32()), vec![]),
                code: BTreeMap::from([(Backend::Null, ""), (Backend::Rust, lshift)]),
            },
        ]
    }
}
