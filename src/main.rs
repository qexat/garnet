use garnet;
use garnet::ast;

fn main() {
    let mut cx = garnet::Cx::new();
    //let mainsym = cx.intern("main");
    let mainsym = cx.intern("_start");
    let ast = ast::Ast {
        decls: vec![ast::Decl::Function {
            name: mainsym,
            signature: ast::Signature {
                params: vec![],
                rettype: cx.intern_type(&garnet::TypeDef::SInt(4)),
            },
            body: vec![ast::Expr::int(42)],
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
