use garnet;
use garnet::ast;

fn main() {
    let mut cx = garnet::Cx::new();
    let mainsym = cx.intern("main");
    let i32sym = cx.intern("i32");
    let ast = ast::Decl::Function {
        name: ast::Symbol(mainsym),
        signature: ast::Signature {
            params: vec![],
            rettype: ast::Type::Name(ast::Symbol(i32sym)),
        },
        body: vec![ast::Expr::Lit {
            val: ast::Literal::Integer(42),
        }],
    };
    println!("Hello, garnet: {:#?}", ast);
}
