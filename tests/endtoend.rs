/// End-to-end test rig: compile programs from source,
/// or currently from AST,
/// and ensure that they give the output we want.
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

#[test]
fn parse_and_compile() {
    let src = r#"fn test(): I32 = 12 end"#;
    let res = eval_program0(src);
    assert_eq!(res, 12);
}

#[test]
fn parse_and_compile2() {
    let src = r#"fn test(x: I32): I32 = x end"#;
    let res = eval_program1(src, 3);
    assert_eq!(res, 3);
}

#[test]
fn parse_and_compile_comments() {
    let src = r#"
-- foo
        -- bar
fn test(): I32 =
    -- baz
    12 -- bop
end  -- bleg
    -- blar
"#;
    let res = eval_program0(src);
    assert_eq!(res, 12);
}

#[test]
fn parse_and_compile_doc_comments() {
    let src = r#"
--- Doc comments go here
fn test(): I32 =
    12
end
"#;
    let res = eval_program0(src);
    assert_eq!(res, 12);
}

#[should_panic]
#[test]
fn parse_and_compile_doc_comments2() {
    let src = r#"
fn test(): I32 =
    --- Doc comments don't go here
    12
end
"#;
    let res = eval_program0(src);
    assert_eq!(res, 12);
}

#[test]
fn parse_and_compile_expr() {
    let src = r#"fn test(x: I32): I32 = x end"#;
    assert_eq!(eval_program1(src, 3), 3);

    let src = r#"fn test(x: I32): I32 = 3 * 3 + 2 end"#;
    assert_eq!(eval_program1(src, 0), 11);

    let src = r#"fn test(x: I32): I32 = 3 * x + 2 end"#;
    assert_eq!(eval_program1(src, 3), 11);

    let src = r#"fn test(x: I32): I32 = if true then x else 3 * 2 end end"#;
    assert_eq!(eval_program1(src, 3), 3);
    let src = r#"fn test(x: I32): I32 = if true then x elseif false then 5 else 3 * 2 end end"#;
    assert_eq!(eval_program1(src, 3), 3);
}

#[test]
fn parse_and_compile_fn2() {
    let src = r#"fn test(x: I32, y: I32): I32 = x + y end"#;
    assert_eq!(eval_program2(src, 3, 4), 7);
}

#[test]
fn parse_and_compile_fn3() {
    let src = r#"fn test(x: Bool, y: I32): I32 = if x then y+1 else y+2 end end"#;
    assert_eq!(eval_program2(src, true, 4), 5);
    assert_eq!(eval_program2(src, false, 4), 6);
}

#[should_panic]
#[test]
fn omg_typechecking_works() {
    let src = r#"fn test(x: Bool, y: I32): Bool = if x then true else y+2 end end"#;
    assert_eq!(eval_program2(src, 1, 4), 5);
}

#[test]
fn omg_funcalls_work() {
    let src = r#"
fn foo(x: I32): I32 = x + 1 end
fn test(x: I32): I32 = foo(x+90) end
"#;
    assert_eq!(eval_program1(src, 1), 92);
}

#[test]
fn fib1() {
    let src = r#"
fn fib(x: I32): I32 =
    if x < 2 then 1
    else fib(x-1) + fib(x - 2)
    end
end

fn test(): I32 =
    fib(7)
end
"#;
    assert_eq!(eval_program0(src), 21);
}

#[test]
fn fib2() {
    // Test function name resolution works without
    // forward decl's too
    let src = r#"
fn test(): I32 =
    fib(7)
end

fn fib(x: I32): I32 =
    if x < 2 then 1
    else fib(x-1) + fib(x - 2)
    end
end
"#;
    assert_eq!(eval_program0(src), 21);
}

/// Test truth tables of logical operators.
#[test]
fn truth() {
    let tests = vec![
        // and
        ("fn test(): Bool = true and true end", 1),
        ("fn test(): Bool = true and false end", 0),
        ("fn test(): Bool = false and true end", 0),
        ("fn test(): Bool = false and false end", 0),
        // or
        ("fn test(): Bool = true or true end", 1),
        ("fn test(): Bool = true or false end", 1),
        ("fn test(): Bool = false or true end", 1),
        ("fn test(): Bool = false or false end", 0),
        // xor
        ("fn test(): Bool = true xor true end", 0),
        ("fn test(): Bool = true xor false end", 1),
        ("fn test(): Bool = false xor true end", 1),
        ("fn test(): Bool = false xor false end", 0),
        // not
        ("fn test(): Bool = not true end", 0),
        ("fn test(): Bool = not false end", 1),
    ];
    for (inp, outp) in &tests {
        assert_eq!(eval_program0(*inp), *outp);
    }
}

/// Test lambda shenanigans
#[test]
fn lambda1() {
    let src = r#"
fn apply(f: fn(I32): I32, arg: I32): I32 =
    f(arg)
end

fn test(): I32 =
    let f: fn(I32): I32 = fn(x: I32): I32 = x * 9 end
    let result: I32 = apply(f, 10)
    result
end
"#;
    assert_eq!(eval_program0(src), 90);
}
