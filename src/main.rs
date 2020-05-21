use garnet;
use garnet::ast;

fn main() {
    let mut cx = garnet::Cx::new();
    let mainsym = cx.intern("main");
    let i32sym = cx.get_typename("i32").unwrap();
    let ast = ast::Ast {
        decls: vec![ast::Decl::Function {
            name: garnet::VarSym(mainsym),
            signature: ast::Signature {
                params: vec![],
                rettype: i32sym,
            },
            body: vec![ast::Expr::int(42)],
        }],
    };
    println!("Hello, garnet: {:#?}", ast);
    let ir = garnet::ir::lower(&ast);
    println!("That becomes: {:#?}", ir);
}
