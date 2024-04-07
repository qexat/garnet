//! A place for us to stick compiler builtins and the metadata they need,
//! instead of having them scattered all over.

use crate::backend::Backend;
use crate::types::*;
use crate::*;

/// A single built-in value that is just included into the generated program.
pub struct Builtin {
    /// The name of the value
    pub name: Sym,
    /// The value's type, used for typechecking
    pub sig: Type,
    /// The implmentation of the builtin as raw code for each particular
    /// backend.
    pub code: BTreeMap<Backend, String>,
}

pub static BUILTINS: Lazy<Vec<Builtin>> = Lazy::new(Builtin::all);

impl Builtin {
    /// Generate all appropriate methods for the given numeric type.
    /// Right now we just stick em in the toplevel with constructed names,
    /// but eventually they'll have to go into their own module/package/whatever.
    /// That package will still contain these generated strings though, so oh well.
    ///
    /// I guess this is the start of Garnet's macro system, huh?
    ///
    /// TODO: rotates, arithmetic shift, wrapping stuff, other?
    fn generate_numerics_for(name: &str, ty: Type) -> Vec<Builtin> {
        // names come from luajit I suppose
        // I'm not in love wtih 'em but also don't want to think about em.
        let println = format!(
            r#"
fn __println_{name}(x: {name}) {{
    println!("{{}}", x);
}}"#
        );
        let band = format!(
            r#"
fn __band_{name}(x: {name}, y: {name}) -> {name} {{
    x & y
}}"#
        );
        let bor = format!(
            r#"
fn __bor_{name}(x: {name}, y: {name}) -> {name} {{
    x | y
}}"#
        );
        let bxor = format!(
            r#"
fn __bxor_{name}(x: {name}, y: {name}) -> {name} {{
    x ^ y
}}"#
        );
        let bnot = format!(
            r#"
fn __bnot_{name}(x: {name}) -> {name} {{
    !x
}}"#
        );

        let rshift = format!(
            r#"
fn __rshift_{name}(x: {name}, i: {name}) -> {name} {{
    x >> i
}}"#
        );
        let lshift = format!(
            r#"
fn __lshift_{name}(x: {name}, i: {name}) -> {name} {{
    x << i
}}"#
        );
        let rol = format!(
            r#"
fn __rol_{name}(x: {name}, i: i32) -> {name} {{
    x.rotate_left(i as u32)
}}"#
        );
        let ror = format!(
            r#"
fn __ror_{name}(x: {name}, i: i32) -> {name} {{
    x.rotate_right(i as u32)
}}"#
        );

        let cast_i32 = format!(
            r#"
fn __{name}_to_i32(x: {name}) -> i32 {{
    x as i32
}}"#
        );
        let cast_u32 = format!(
            r#"
fn __{name}_to_u32(x: {name}) -> u32 {{
    x as u32
}}"#
        );

        vec![
            Builtin {
                name: Sym::new(format!("__println_{name}")),
                sig: Type::Func(vec![ty.clone()], Box::new(Type::unit()), vec![]),
                code: BTreeMap::from([(Backend::Null, "".into()), (Backend::Rust, println)]),
            },
            Builtin {
                name: Sym::new(format!("__band_{name}")),
                sig: Type::Func(vec![ty.clone(), ty.clone()], Box::new(ty.clone()), vec![]),
                code: BTreeMap::from([(Backend::Null, "".into()), (Backend::Rust, band)]),
            },
            Builtin {
                name: Sym::new(format!("__bor_{name}")),
                sig: Type::Func(vec![ty.clone(), ty.clone()], Box::new(ty.clone()), vec![]),
                code: BTreeMap::from([(Backend::Null, "".into()), (Backend::Rust, bor)]),
            },
            Builtin {
                name: Sym::new(format!("__bxor_{name}")),
                sig: Type::Func(vec![ty.clone(), ty.clone()], Box::new(ty.clone()), vec![]),
                code: BTreeMap::from([(Backend::Null, "".into()), (Backend::Rust, bxor)]),
            },
            Builtin {
                name: Sym::new(format!("__bnot_{name}")),
                sig: Type::Func(vec![ty.clone()], Box::new(ty.clone()), vec![]),
                code: BTreeMap::from([(Backend::Null, "".into()), (Backend::Rust, bnot)]),
            },
            Builtin {
                name: Sym::new(format!("__rshift_{name}")),
                sig: Type::Func(vec![ty.clone(), ty.clone()], Box::new(ty.clone()), vec![]),
                code: BTreeMap::from([(Backend::Null, "".into()), (Backend::Rust, rshift)]),
            },
            Builtin {
                name: Sym::new(format!("__lshift_{name}")),
                sig: Type::Func(vec![ty.clone(), ty.clone()], Box::new(ty.clone()), vec![]),
                code: BTreeMap::from([(Backend::Null, "".into()), (Backend::Rust, lshift)]),
            },
            Builtin {
                name: Sym::new(format!("__rol_{name}")),
                sig: Type::Func(vec![ty.clone(), Type::i32()], Box::new(ty.clone()), vec![]),
                code: BTreeMap::from([(Backend::Null, "".into()), (Backend::Rust, rol)]),
            },
            Builtin {
                name: Sym::new(format!("__ror_{name}")),
                sig: Type::Func(vec![ty.clone(), Type::i32()], Box::new(ty.clone()), vec![]),
                code: BTreeMap::from([(Backend::Null, "".into()), (Backend::Rust, ror)]),
            },
            Builtin {
                name: Sym::new(format!("__{name}_to_i32")),
                sig: Type::Func(vec![ty.clone()], Box::new(Type::i32()), vec![]),
                code: BTreeMap::from([(Backend::Null, "".into()), (Backend::Rust, cast_i32)]),
            },
            Builtin {
                name: Sym::new(format!("__{name}_to_u32")),
                sig: Type::Func(vec![ty.clone()], Box::new(Type::u32()), vec![]),
                code: BTreeMap::from([(Backend::Null, "".into()), (Backend::Rust, cast_u32)]),
            },
        ]
    }
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

        let mut funcs = vec![
            Builtin {
                name: Sym::new("__println"),
                sig: Type::Func(vec![Type::i32()], Box::new(Type::unit()), vec![]),
                code: BTreeMap::from([
                    (Backend::Null, "".into()),
                    (Backend::Rust, rust_println.into()),
                ]),
            },
            Builtin {
                name: Sym::new("__println_bool"),
                sig: Type::Func(vec![Type::bool()], Box::new(Type::unit()), vec![]),
                code: BTreeMap::from([
                    (Backend::Null, "".into()),
                    (Backend::Rust, rust_println_bool.into()),
                ]),
            },
        ];
        funcs.extend(Self::generate_numerics_for("i8", Type::i8()));
        funcs.extend(Self::generate_numerics_for("i16", Type::i16()));
        funcs.extend(Self::generate_numerics_for("i32", Type::i32()));
        funcs.extend(Self::generate_numerics_for("i64", Type::i64()));
        funcs.extend(Self::generate_numerics_for("u8", Type::u8()));
        funcs.extend(Self::generate_numerics_for("u16", Type::u16()));
        funcs.extend(Self::generate_numerics_for("u32", Type::u32()));
        funcs.extend(Self::generate_numerics_for("u64", Type::u64()));
        funcs
    }
}
