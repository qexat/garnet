//! Basic formatter and pretty-printer.
//! Takes an AST and outputs to a Write object.

use std::io;

use crate::ast::*;
use crate::Cx;

const INDENT_SIZE: usize = 4;

fn unparse_decl(cx: &Cx, d: &Decl, out: &mut dyn io::Write) -> io::Result<()> {
    match d {
        Decl::Function {
            name,
            signature,
            body,
        } => {
            let name = cx.fetch(*name);
            write!(out, "fn {}", name)?;
            unparse_sig(cx, signature, out)?;
            writeln!(out, " =")?;
            for expr in body {
                unparse_expr(cx, expr, 1, out)?;
            }

            writeln!(out, "\nend")
        }
        Decl::Const {
            name,
            typename,
            init,
        } => {
            let name = cx.fetch(*name);
            let tname = cx.fetch_type(*typename).get_name();
            write!(out, "const {}: {} = ", name, tname)?;
            unparse_expr(cx, init, 0, out)?;
            writeln!(out)
        }
    }
}

fn unparse_sig(cx: &Cx, sig: &Signature, out: &mut dyn io::Write) -> io::Result<()> {
    write!(out, "(")?;
    for (name, typename) in sig.params.iter() {
        let name = cx.fetch(*name);
        let tname = cx.fetch_type(*typename).get_name();
        write!(out, "{}: {}", name, tname)?;
    }
    write!(out, "): ")?;
    let rettype = cx.fetch_type(sig.rettype).get_name();
    write!(out, "{}", rettype)
}
fn unparse_expr(cx: &Cx, e: &Expr, indent: usize, out: &mut dyn io::Write) -> io::Result<()> {
    use Expr as E;
    for _ in 0..(indent * INDENT_SIZE) {
        write!(out, " ")?;
    }
    match e {
        E::Lit { val } => match val {
            Literal::Integer(i) => write!(out, "{}", i),
            Literal::Bool(b) => write!(out, "{}", b),
            Literal::Unit => write!(out, "{{}}"),
        },
        E::Var { name } => {
            let name = cx.fetch(*name);
            write!(out, "{}", name)
        }
        E::UniOp { op, rhs } => todo!(),
        E::BinOp { op, lhs, rhs } => {
            let opstr = match op {
                BOp::Add => "+",
                BOp::Sub => "-",
                BOp::Mul => "*",
                BOp::Div => "/",
                BOp::Mod => "%",
            };
            unparse_expr(cx, lhs, 0, out)?;
            write!(out, " {} ", opstr)?;
            unparse_expr(cx, rhs, 0, out)
        }
        E::Block { body } => {
            writeln!(out, "do")?;
            for e in body {
                unparse_expr(cx, e, indent + 1, out)?;
            }
            writeln!(out, "end")
        }
        E::Let {
            varname,
            typename,
            init,
        } => {
            let name = cx.fetch(*varname);
            let tname = cx.fetch_type(*typename).get_name();
            write!(out, "let {}: {} = ", name, tname)?;
            unparse_expr(cx, init, 0, out)?;
            writeln!(out)
        }
        E::If { cases, falseblock } => todo!(),
        E::Loop { body } => todo!(),
        E::Lambda { signature, body } => todo!(),
        E::Funcall { func, params } => {
            unparse_expr(cx, func, 0, out)?;
            write!(out, "(")?;
            for e in params {
                unparse_expr(cx, e, 0, out)?;
                write!(out, ", ")?;
            }
            write!(out, ")")
        }
        E::Break => writeln!(out, "break"),
        E::Return { retval: Some(e) } => {
            write!(out, "return ")?;
            unparse_expr(cx, e, 0, out)?;
            writeln!(out)
        }
        E::Return { retval: None } => writeln!(out, "return"),
    }
}

/// Take the AST and produce a formatted string of source code.
pub fn unparse(cx: &Cx, ast: &Ast, out: &mut dyn io::Write) -> io::Result<()> {
    for decl in ast.decls.iter() {
        unparse_decl(cx, decl, out)?;
        writeln!(out)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::format::unparse;
    use crate::Cx;
    use std::io::Cursor;
    #[test]
    fn test_reparse() {
        let src = r#"fn test(x: I32): I32 =
    3 * x + 2
end

"#;
        let cx = &mut Cx::new();
        let ast = {
            let mut parser = crate::parser::Parser::new(cx, src);
            parser.parse()
        };
        let mut output = Cursor::new(vec![]);
        unparse(cx, &ast, &mut output).unwrap();
        let output_str = String::from_utf8(output.into_inner()).unwrap();
        assert_eq!(src, output_str);
    }
}