use garnet;
use garnet::ast;

fn main() {
    let mut cx = garnet::Cx::new();
    let mainsym = cx.intern("main");
    let i32sym = cx.intern("i32");
    let ast = ast::Ast {
        decls: vec![ast::Decl::Function {
            name: ast::Symbol(mainsym),
            signature: ast::Signature {
                params: vec![],
                rettype: ast::Type {
                    name: ast::Symbol(i32sym),
                },
            },
            body: vec![ast::Expr::int(42)],
        }],
    };
    println!("Hello, garnet: {:#?}", ast);
    let ir = garnet::ir::lower(&ast);
    println!("That becomes: {:#?}", ir);
}
