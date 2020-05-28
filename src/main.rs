use garnet;
use garnet::ast;

fn main() {
    let mut cx = garnet::Cx::new();
    //let mainsym = cx.intern("main");
    let mainsym = cx.intern("_start");
    let varsym = cx.intern("foo");
    let i32_t = cx.intern_type(&garnet::TypeDef::SInt(4));
    let ast = ast::Ast {
        decls: vec![ast::Decl::Function {
            name: mainsym,
            signature: ast::Signature {
                params: vec![],
                rettype: i32_t,
            },
            //body: vec![ast::Expr::int(42)],
            body: vec![
                ast::Expr::Let {
                    varname: varsym,
                    typename: i32_t,
                    init: Box::new(ast::Expr::int(42)),
                },
                ast::Expr::BinOp {
                    op: ast::BOp::Add,
                    lhs: Box::new(ast::Expr::Var { name: varsym }),
                    rhs: Box::new(ast::Expr::Var { name: varsym }),
                },
            ],
        }],
    };
    println!("AST: {:#?}", ast);
    let ir = garnet::ir::lower(&ast);
    //println!("That becomes: {:#?}", ir);
    let wasm = garnet::backend::output(&mut cx, &ir);
    //println!("WASM is: {:?}", wasm);

    // Output to file
    std::fs::write("out.wasm", &wasm).unwrap();
}
