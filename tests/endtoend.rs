/// End-to-end test rig: compile programs from source,
/// or currently from AST,
/// and ensure that they give the output we want.
use std::fmt::Display;
use std::fs;
use std::io::Write;

use garnet;

/// We gotta create a temporary file, write our code to it, call rustc on it, and execute the
/// result.
fn eval_rs(bytes: &[u8], entry_point: &str) -> i32 {
    // Write program to a temp file
    let dir = tempfile::TempDir::new().unwrap();
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::path::PathBuf;

    // Won't make a collision since TempDir() is always unique, right?  Totally.
    let mut h = DefaultHasher::new();
    bytes.hash(&mut h);
    let hashnum = h.finish();
    let filename = format!("{}.rs", hashnum);
    let mut filepath = PathBuf::from(dir.path());
    filepath.push(filename);

    // Output file with test harness entry point.
    {
        let mut f = fs::File::create(&filepath).unwrap();
        f.write(bytes).unwrap();
        f.write(entry_point.as_bytes()).unwrap();
    }

    eprintln!("Source file: {:?}", &filepath);
    eprintln!(
        "{}",
        String::from_utf8(fs::read(&filepath).unwrap()).unwrap()
    );

    // Execute rustc
    let mut output_file = filepath.clone();
    output_file.set_extension("exe");

    eprintln!("Output file: {:?}", &output_file);

    use std::process::Command;
    let output = Command::new("rustc")
        .arg(filepath)
        .arg("-o")
        .arg(&output_file)
        .output()
        .expect("Failed to execute rustc");
    println!("rustc output:");
    println!("stdout: {}", String::from_utf8(output.stdout).unwrap());
    println!("stderr: {}", String::from_utf8(output.stderr).unwrap());
    // Execute resulting program
    let output = Command::new(output_file)
        .output()
        .expect("Failed to execute program");

    // get return code
    let out = output.status.code().unwrap();

    //clean up dir
    // TODO: Make it leave the file behind if compilation failed somehow.
    dir.close().unwrap();

    out
}

/// Entry point for a func with 0 args.
/// USed in a couple different places.
const ENTRY0: &str = r#"
fn main() {
    let res = test();
    std::process::exit(res as i32);
}
"#;

/// Takes a program defining a function named `test` with 1 arg,
/// returns its result
fn eval_program0(src: &str) -> i32 {
    let out = garnet::compile(src);
    eval_rs(&out, ENTRY0)
}

/// Same as eval_program0 but `test` takes 1 args.
fn eval_program1(src: &str, input: i32) -> i32 {
    let out = garnet::compile(src);
    // Add entry point
    let entry = format!(
        r#"
fn main() {{
    let res = test({});
    std::process::exit(res as i32);
}}
"#,
        input
    );
    eval_rs(&out, &entry)
}

/// Same as eval_program0 but `test` takes 2 args.
fn eval_program2<T1, T2>(src: &str, i1: T1, i2: T2) -> i32
where
    T1: Display,
    T2: Display,
{
    let out = garnet::compile(src);
    // Add entry point
    let entry = format!(
        r#"
fn main() {{
    let res = test({}, {});
    std::process::exit(res as i32);
}}
"#,
        i1, i2
    );
    eval_rs(&out, &entry)
}

/* TODO: Since we have to add an entry point explicitly to a Rust program,
 * these AST-based tests can't be executed correctly...
 * Ponder how to fix it.  Either we rewrite them to start from source code,
 * or we figure out a way to staple the Rust entry point onto them afterwards,
 * or we make the Rust entry point redundant and/or more flexible.

#[test]
fn var_lookup() {
    let mut cx = garnet::Cx::new();
    let mainsym = cx.intern("test");
    let varsym = cx.intern("a");
    let i32_t = cx.intern_type(&garnet::TypeDef::SInt(4));
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
    let ir = garnet::hir::lower(&ast);
    let checked = garnet::typeck::typecheck(&mut cx, ir).unwrap();
    let wasm = garnet::backend::output(garnet::backend::Backend::Rust, &mut cx, &checked);
    let res = eval_rs(&wasm, ENTRY0);
    assert_eq!(res, 5);
}

#[test]
fn subtraction() {
    let mut cx = garnet::Cx::new();
    let mainsym = cx.intern("test");
    let i32_t = cx.intern_type(&garnet::TypeDef::SInt(4));
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
    let ir = garnet::hir::lower(&ast);
    let checked = garnet::typeck::typecheck(&mut cx, ir).unwrap();
    let wasm = garnet::backend::output(garnet::backend::Backend::Rust, &mut cx, &checked);
    let res = eval_program0(&String::from_utf8(wasm).unwrap());
    assert_eq!(res, 12);
}

#[test]
fn maths() {
    let mut cx = garnet::Cx::new();
    let mainsym = cx.intern("test");
    let i32_t = cx.intern_type(&garnet::TypeDef::SInt(4));
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
    let ir = garnet::hir::lower(&ast);
    let checked = garnet::typeck::typecheck(&mut cx, ir).unwrap();
    let wasm = garnet::backend::output(garnet::backend::Backend::Rust, &mut cx, &checked);
    // Compiling a function gets us a dynamically-typed thing.
    // Assert what its type is and run it.
    let res = eval_program0(&String::from_utf8(wasm).unwrap());
    assert_eq!(res, 3);
}

#[test]
fn block() {
    let mut cx = garnet::Cx::new();
    let mainsym = cx.intern("test");
    let bool_t = cx.intern_type(&garnet::TypeDef::Bool);
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
    let ir = garnet::hir::lower(&ast);
    let checked = garnet::typeck::typecheck(&mut cx, ir).unwrap();
    let wasm = garnet::backend::output(garnet::backend::Backend::Rust, &mut cx, &checked);
    let res = eval_program0(&String::from_utf8(wasm).unwrap());
    assert_eq!(res, 0);
}
*/

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

#[test]
fn lambda2() {
    let src = r#"
fn test(): I32 =
    if true then
        fn(): I32 = 1 end
    else
        fn(): I32 = 10 end
    end
    ()
end
"#;

    assert_eq!(eval_program0(src), 1);
}

/// Test tuple constructors
#[test]
fn tuples1() {
    let src = r#"
fn test(): I32 =
    let t: {I32, I32, Bool} = {3, 5, false}
    t.0
end
"#;
    assert_eq!(eval_program0(src), 3);
}

#[test]
fn tuples2() {
    let src = r#"

fn test(): I32 =
    let x: {I32, I32} = {10, 11}
    x.0 + x.1
end
"#;
    assert_eq!(eval_program0(src), 21);
}

#[test]
fn assignment() {
    let src = r#"
fn test(): I32 =
    let mut x: I32 = 10
    x = 20
    x
end
"#;
    assert_eq!(eval_program0(src), 20);
}

#[test]
fn assign_tuples() {
    let src = r#"

fn test(): I32 =
    let x: {I32, I32} = {10, 11}
    let mut y: {I32, I32} = {20, 21}
    y = x
    y.0
end
"#;
    assert_eq!(eval_program0(src), 10);
}

#[test]
fn assign_tuples_lvalue() {
    let src = r#"

fn test(): I32 =
    let x: {I32, I32} = {10, 11}
    let mut y: {I32, I32} = {20, 21}
    y.1 = x.0
    y.1
end
"#;
    assert_eq!(eval_program0(src), 10);
}

/// This is a little weird... apparently Rust tuples
/// automatically implement Copy if all their contents
/// do?
#[test]
fn move_tuples() {
    let src = r#"

fn test(): I32 =
    let x: {I32, I32} = {10, 11}
    let mut y: {I32, I32} = {20, 21}
    let mut z: {I32, I32} = {30, 31}
    y = x
    z = x
    z.0
end
"#;
    assert_eq!(eval_program0(src), 10);
}

#[test]
fn return_tuples() {
    let src = r#"

fn foo(): {I32, I32} =
    {10, 20}
end

fn test(): I32 =
    let x: {I32, I32} = foo()
    x.1
end
"#;
    assert_eq!(eval_program0(src), 20);
}

/*  This needs to wait for generic-ish binops
#[test]
fn add_numbers() {
    let src = r#"

fn foo(x: I64): I64 =
    x + 3i64
end

fn test(): I64 =
    let x: I64 = foo(9)
    x
end
"#;
    assert_eq!(eval_program0(src), 12);
}
*/
