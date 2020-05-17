use garnet::ast;

fn main() {
    let ast = ast::Decl::Function {
        name: ast::Symbol("main".into()),
        signature: ast::Signature {
            params: vec![],
            rettype: ast::Type::Name(ast::Symbol("i32".into())),
        },
        body: vec![ast::Expr::Lit {
            val: ast::Literal::Integer(42),
        }],
    };
    println!("Hello, garnet: {:#?}", ast);
}
