//! Test garnet as a crate: compile programs from AST
//! and ensure that they give the output we want.
//!
//! TODO: Cleanup unused code.  Do we want to keep any of this?  The
//! name mangling had to chnage so we could just call main() `main`
/*
use std::fmt::Display;
use std::fs;
use std::io::Write;

use garnet::{self, INT};

/// We gotta create a temporary file, write our code to it, call rustc on it, and return the
/// result as a dynamic library already loaded via libloading
fn eval_rs_dylib(bytes: &[u8]) -> libloading::Library {
    // Write program to a temp file
    let dir = tempfile::TempDir::new().unwrap();
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::path::PathBuf;

    // Won't make a collision since TempDir() is always unique, right?  Totally.  Honest.
    let mut h = DefaultHasher::new();
    bytes.hash(&mut h);
    let hashnum = h.finish();
    let filename = format!("{}.rs", hashnum);
    let mut filepath = PathBuf::from(dir.path());
    filepath.push(filename);

    // Output file
    {
        let mut f = fs::File::create(&filepath).unwrap();
        f.write(bytes).unwrap();
    }

    eprintln!("Source file: {:?}", &filepath);
    eprintln!(
        "{}",
        String::from_utf8(fs::read(&filepath).unwrap()).unwrap()
    );

    // Execute rustc
    let mut output_file = filepath.clone();
    output_file.set_extension("dlib");

    eprintln!("Output file: {:?}", &output_file);

    use std::process::Command;
    let output = Command::new("rustc")
        .arg("-o")
        .arg(&output_file)
        .arg("--crate-type")
        .arg("cdylib")
        .arg(filepath)
        .output()
        .expect("Failed to execute rustc");
    println!("rustc output:");
    println!("stdout: {}", String::from_utf8(output.stdout).unwrap());
    println!("stderr: {}", String::from_utf8(output.stderr).unwrap());
    // Load and return resulting program as DLL
    let lib = unsafe { libloading::Library::new(output_file).unwrap() };

    //clean up dir
    // TODO: Make it leave the file behind if compilation failed somehow.
    dir.close().unwrap();

    lib
}

/// Takes a program defining a function named `test` with 1 arg,
/// returns its result
fn eval_program0(src: &str) -> i32 {
    let out = garnet::compile(src);
    let lib = eval_rs_dylib(&out);
    unsafe {
        let sym: libloading::Symbol<unsafe extern "C" fn() -> i32> = lib.get(b"test").unwrap();
        sym()
    }
}

/// Same as eval_program0 but `test` takes 1 args.
fn eval_program1<T1>(src: &str, input: T1) -> i32
where
    T1: Display,
{
    let out = garnet::compile(src);
    let lib = eval_rs_dylib(&out);
    unsafe {
        let sym: libloading::Symbol<unsafe extern "C" fn(T1) -> i32> = lib.get(b"test").unwrap();
        sym(input)
    }
}

/// Same as eval_program0 but `test` takes 2 args.
fn eval_program2<T1, T2>(src: &str, i1: T1, i2: T2) -> i32
where
    T1: Display,
    T2: Display,
{
    let out = garnet::compile(src);
    let lib = eval_rs_dylib(&out);
    unsafe {
        let sym: libloading::Symbol<unsafe extern "C" fn(T1, T2) -> i32> =
            lib.get(b"test").unwrap();
        sym(i1, i2)
    }
}

use garnet::ast;
#[test]
fn var_lookup() {
    let mainsym = INT.intern("test");
    let varsym = INT.intern("a");
    let i32_t = INT.intern_type(&garnet::TypeDef::SInt(4));
    let ast = ast::Ast {
        decls: vec![ast::Decl::Function {
            name: mainsym,
            signature: ast::Signature {
                params: vec![(varsym, i32_t)],
                rettype: i32_t,
            },
            body: vec![ast::Expr::Var { name: varsym }],
            doc_comment: vec![],
        }],
    };
    let ir = garnet::hir::lower(&mut |_| (), &ast);
    let checked = garnet::typeck::typecheck(ir).unwrap();
    let src = garnet::backend::output(garnet::backend::Backend::Rust, &checked);

    let lib = eval_rs_dylib(&src);
    let res = unsafe {
        let sym: libloading::Symbol<unsafe extern "C" fn(i32) -> i32> = lib.get(b"test").unwrap();
        sym(5)
    };
    assert_eq!(res, 5);
}

#[test]
fn subtraction() {
    let mainsym = INT.intern("test");
    let i32_t = INT.intern_type(&garnet::TypeDef::SInt(4));
    let ast = ast::Ast {
        decls: vec![ast::Decl::Function {
            name: mainsym,
            signature: ast::Signature {
                params: vec![],
                rettype: i32_t,
            },
            body: vec![ast::Expr::BinOp {
                op: ast::BOp::Sub,
                lhs: Box::new(ast::Expr::int(9)),
                rhs: Box::new(ast::Expr::int(-3)),
            }],
            doc_comment: vec![],
        }],
    };
    let ir = garnet::hir::lower(&mut |_| (), &ast);
    let checked = garnet::typeck::typecheck(ir).unwrap();
    let src = garnet::backend::output(garnet::backend::Backend::Rust, &checked);
    let lib = eval_rs_dylib(&src);
    let res = unsafe {
        let sym: libloading::Symbol<unsafe extern "C" fn() -> i32> = lib.get(b"test").unwrap();
        sym()
    };
    assert_eq!(res, 12);
}

#[test]
fn maths() {
    let mainsym = INT.intern("test");
    let i32_t = INT.intern_type(&garnet::TypeDef::SInt(4));
    let ast = ast::Ast {
        decls: vec![ast::Decl::Function {
            name: mainsym,
            signature: ast::Signature {
                params: vec![],
                rettype: i32_t,
            },
            body: vec![ast::Expr::BinOp {
                op: ast::BOp::Div,
                lhs: Box::new(ast::Expr::int(9)),
                rhs: Box::new(ast::Expr::int(3)),
            }],
            doc_comment: vec![],
        }],
    };
    let ir = garnet::hir::lower(&mut |_| (), &ast);
    let checked = garnet::typeck::typecheck(ir).unwrap();
    let src = garnet::backend::output(garnet::backend::Backend::Rust, &checked);

    let lib = eval_rs_dylib(&src);
    let res = unsafe {
        let sym: libloading::Symbol<unsafe extern "C" fn() -> i32> = lib.get(b"test").unwrap();
        sym()
    };
    assert_eq!(res, 3);
}

#[test]
fn block() {
    let mainsym = INT.intern("test");
    let bool_t = INT.intern_type(&garnet::TypeDef::Bool);
    let ast = ast::Ast {
        decls: vec![ast::Decl::Function {
            name: mainsym,
            signature: ast::Signature {
                params: vec![],
                rettype: bool_t,
            },
            body: vec![
                ast::Expr::BinOp {
                    op: ast::BOp::Div,
                    lhs: Box::new(ast::Expr::int(3)),
                    rhs: Box::new(ast::Expr::int(1)),
                },
                ast::Expr::Block {
                    body: vec![
                        ast::Expr::BinOp {
                            op: ast::BOp::Div,
                            lhs: Box::new(ast::Expr::int(3)),
                            rhs: Box::new(ast::Expr::int(1)),
                        },
                        ast::Expr::bool(false),
                    ],
                },
            ],
            doc_comment: vec![],
        }],
    };
    let ir = garnet::hir::lower(&mut |_| (), &ast);
    let checked = garnet::typeck::typecheck(ir).unwrap();
    let src = garnet::backend::output(garnet::backend::Backend::Rust, &checked);

    let lib = eval_rs_dylib(&src);
    let res = unsafe {
        let sym: libloading::Symbol<unsafe extern "C" fn() -> i32> = lib.get(b"test").unwrap();
        sym()
    };
    assert_eq!(res, 0);
}
*/
