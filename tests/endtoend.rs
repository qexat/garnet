/// End-to-end test rig: compile programs from source,
/// or currently from AST,
/// and ensure that they give the output we want.
use garnet::{self, ast};
use wasmtime as w;

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
    let wasm = garnet::backend::output(&mut cx, &ir);
    // Compiling a function gets us a dynamically-typed thing.
    // Assert what its type is and run it.
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
    let wasm = garnet::backend::output(&mut cx, &ir);
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
    //let i32_t = cx.intern_type(&garnet::TypeDef::SInt(4));
    let bool_t = cx.intern_type(&garnet::TypeDef::Bool);
    let ast = ast::Ast {
        decls: vec![ast::Decl::Function {
            name: mainsym,
            signature: ast::Signature {
                params: vec![],
                rettype: bool_t,
            },
            //body: vec![ast::Expr::int(42)],
            body: vec![
                ast::Expr::BinOp {
                    op: ast::BOp::Div,
                    lhs: Box::new(ast::Expr::int(3)),
                    rhs: Box::new(ast::Expr::int(1)),
                },
                ast::Expr::Block {
                    body: vec![
                        /* TODO: Fix this test, blocks don't end
                         * with the right number of things on the stack.
                         */
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
    let wasm = garnet::backend::output(&mut cx, &ir);
    // Compiling a function gets us a dynamically-typed thing.
    // Assert what its type is and run it.
    let f = compile_wasm(&wasm).get1::<(), i32>().unwrap();
    let res: i32 = f(()).unwrap();
    assert_eq!(res, 0);
}
