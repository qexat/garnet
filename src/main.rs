use garnet;
use garnet::ast;

fn main() {
    let mut cx = garnet::Cx::new();
    let mainsym = cx.intern("main");
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
    println!("Hello, garnet: {:#?}", ast);
    let ir = garnet::ir::lower(&ast);
    println!("That becomes: {:#?}", ir);
}
