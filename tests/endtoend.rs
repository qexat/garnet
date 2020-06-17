/// End-to-end test rig: compile programs from source,
/// or currently from AST,
/// and ensure that they give the output we want.
use garnet::{self, ast};
use wasmtime as w;

/* TODO: Hmmm, this is awkward 'cause of all the interning.
 * How do we do this nicely without parsing?
fn compile_garnet(ast: &ir::Ast) -> Vec<u8> {
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
        }],
    };
    let ir = garnet::ir::lower(&ast);
    garnet::backend::output(&mut cx, &ir)
}
*/

fn compile_wasm(bytes: &[u8]) -> w::Func {
    // Set up settings and compile bytes
    let store = w::Store::default();
    let module = w::Module::new(&store, bytes).expect("Unvalid module");
    // Create runtime env
    let instance = w::Instance::new(&module, &[]).expect("Could not instantiate module");
    // extract function
    instance
        .get_func("test")
        .expect("Test function needs to be called 'test'")
}

/// Takes a program defining a function with 1 arg named 'test',
/// that returns 1 arg, and makes sure they match.
fn eval_program1(src: &str, input: i32) -> i32 {
    let wasm = garnet::compile(src);
    let f = compile_wasm(&wasm).get1::<i32, i32>().unwrap();
    f(input).unwrap()
}

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
        }],
    };
    let ir = garnet::ir::lower(&ast);
    let checked = garnet::typeck::typecheck(&mut cx, ir).unwrap();
    let wasm = garnet::backend::output(&mut cx, &checked);
    // Compiling a function gets us a dynamically-typed thing.
    // Assert what its type is and run it.
    let f = compile_wasm(&wasm).get1::<i32, i32>().unwrap();
    let res: i32 = f(5).unwrap();
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
        }],
    };
    let ir = garnet::ir::lower(&ast);
    let checked = garnet::typeck::typecheck(&mut cx, ir).unwrap();
    let wasm = garnet::backend::output(&mut cx, &checked);
    let f = compile_wasm(&wasm).get1::<(), i32>().unwrap();
    let res: i32 = f(()).unwrap();
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
        }],
    };
    let ir = garnet::ir::lower(&ast);
    let checked = garnet::typeck::typecheck(&mut cx, ir).unwrap();
    let wasm = garnet::backend::output(&mut cx, &checked);
    // Compiling a function gets us a dynamically-typed thing.
    // Assert what its type is and run it.
    let f = compile_wasm(&wasm).get1::<(), i32>().unwrap();
    let res: i32 = f(()).unwrap();
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
        }],
    };
    let ir = garnet::ir::lower(&ast);
    let checked = garnet::typeck::typecheck(&mut cx, ir).unwrap();
    let wasm = garnet::backend::output(&mut cx, &checked);
    let f = compile_wasm(&wasm).get1::<(), i32>().unwrap();
    let res: i32 = f(()).unwrap();
    assert_eq!(res, 0);
}

#[test]
fn parse_and_compile() {
    let src = r#"fn test(): I32 = 12 end"#;
    let wasm = garnet::compile(src);
    let f = compile_wasm(&wasm).get1::<(), i32>().unwrap();
    let res: i32 = f(()).unwrap();
    assert_eq!(res, 12);
}

#[test]
fn parse_and_compile2() {
    let src = r#"fn test(x: I32): I32 = x end"#;
    let wasm = garnet::compile(src);
    let f = compile_wasm(&wasm).get1::<i32, i32>().unwrap();
    let res: i32 = f(3).unwrap();
    assert_eq!(res, 3);
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
