//! Basic formatter and pretty-printer.
//! Takes an AST and outputs to a Write object.

use std::io;

use crate::ast::*;
use crate::INT;

const INDENT_SIZE: usize = 4;

fn unparse_decl(d: &Decl, out: &mut dyn io::Write) -> io::Result<()> {
    match d {
        Decl::Function {
            name,
            signature,
            body,
            doc_comment,
        } => {
            for line in doc_comment.iter() {
                // No writeln, doc comment strings already end in \n
                write!(out, "--- {}", line)?;
            }
            let name = INT.fetch(*name);
            write!(out, "fn {}", name)?;
            unparse_sig(signature, out)?;
            writeln!(out, " =")?;
            for expr in body {
                unparse_expr(expr, 1, out)?;
            }

            writeln!(out, "\nend")
        }
        Decl::Const {
            name,
            typename,
            init,
            doc_comment,
        } => {
            for line in doc_comment.iter() {
                write!(out, "--- {}", line)?;
            }
            let name = INT.fetch(*name);
            let tname = INT.fetch_type(*typename).get_name();
            write!(out, "const {}: {} = ", name, tname)?;
            unparse_expr(init, 0, out)?;
            writeln!(out)
        }
        Decl::TypeDef { .. } => todo!(),
        Decl::StructDef { .. } => todo!(),
    }
}

fn unparse_sig(sig: &Signature, out: &mut dyn io::Write) -> io::Result<()> {
    write!(out, "(")?;
    for (name, typename) in sig.params.iter() {
        let name = INT.fetch(*name);
        let tname = INT.fetch_type(*typename).get_name();
        write!(out, "{}: {}", name, tname)?;
    }
    write!(out, "): ")?;
    let rettype = INT.fetch_type(sig.rettype).get_name();
    write!(out, "{}", rettype)
}

fn unparse_exprs(exprs: &[Expr], indent: usize, out: &mut dyn io::Write) -> io::Result<()> {
    for expr in exprs {
        unparse_expr(expr, indent, out)?;
    }
    Ok(())
}
fn unparse_expr(e: &Expr, indent: usize, out: &mut dyn io::Write) -> io::Result<()> {
    use Expr as E;
    for _ in 0..(indent * INDENT_SIZE) {
        write!(out, " ")?;
    }
    match e {
        E::Lit { val } => match val {
            Literal::Integer(i) => write!(out, "{}", i),
            Literal::SizedInteger { vl, bytes } => {
                let size = bytes * 8;
                write!(out, "{}i{}", vl, size)
            }
            Literal::Bool(b) => write!(out, "{}", b),
        },
        E::Var { name } => {
            let name = INT.fetch(*name);
            write!(out, "{}", name)
        }
        E::UniOp { op, rhs } => {
            let opstr = match op {
                UOp::Neg => "-",
                UOp::Not => "not ",
                UOp::Ref => "&",
                UOp::Deref => "^",
            };
            write!(out, "{}", opstr)?;
            unparse_expr(rhs, 0, out)
        }
        E::BinOp { op, lhs, rhs } => {
            let opstr = match op {
                BOp::Add => "+",
                BOp::Sub => "-",
                BOp::Mul => "*",
                BOp::Div => "/",
                BOp::Mod => "%",

                BOp::Eq => "==",
                BOp::Neq => "/=",
                BOp::Gt => ">",
                BOp::Lt => "<",
                BOp::Gte => ">=",
                BOp::Lte => "<=",

                BOp::And => "and",
                BOp::Or => "or",
                BOp::Xor => "xor",
            };
            unparse_expr(lhs, 0, out)?;
            write!(out, " {} ", opstr)?;
            unparse_expr(rhs, 0, out)
        }
        E::Block { body } => {
            writeln!(out, "do")?;
            for e in body {
                unparse_expr(e, indent + 1, out)?;
            }
            writeln!(out, "end")
        }
        E::Let {
            varname,
            typename,
            init,
            mutable,
        } => {
            let name = INT.fetch(*varname);
            write!(out, "let ")?;
            if *mutable {
                write!(out, "mut ")?;
            }
            write!(out, "{} ", name)?;
            let tname = INT.fetch_type(*typename).get_name();
            write!(out, ": {} ", tname)?;
            write!(out, "= ")?;

            unparse_expr(init, 0, out)?;
            writeln!(out)
        }
        E::If { cases, falseblock } => {
            assert!(cases.len() > 1);
            let first_case = &cases[0];
            write!(out, "if ")?;
            unparse_expr(&*first_case.condition, 0, out)?;
            writeln!(out, "then")?;
            unparse_exprs(&first_case.body, indent + 1, out)?;

            for case in &cases[1..] {
                write!(out, "elseif ")?;
                unparse_expr(&case.condition, 0, out)?;
                writeln!(out, "then")?;
                unparse_exprs(&case.body, indent + 1, out)?;
            }
            if !falseblock.is_empty() {
                writeln!(out, "else")?;
                unparse_exprs(falseblock, indent + 1, out)?;
            }
            writeln!(out, "end")
        }
        E::Loop { body } => {
            writeln!(out, "loop")?;
            unparse_exprs(body, indent + 1, out)?;
            writeln!(out)?;
            writeln!(out, "end")
        }
        E::Lambda { signature, body } => {
            write!(out, "fn(")?;
            unparse_sig(signature, out)?;
            writeln!(out, " =")?;
            unparse_exprs(body, indent + 1, out)?;
            writeln!(out)?;
            writeln!(out, "end")
        }
        E::Funcall { func, params } => {
            unparse_expr(func, 0, out)?;
            write!(out, "(")?;
            for e in params {
                unparse_expr(e, 0, out)?;
                write!(out, ", ")?;
            }
            write!(out, ")")
        }
        E::Break => writeln!(out, "break"),
        E::Return { retval: e } => {
            write!(out, "return ")?;
            unparse_expr(e, 0, out)?;
            writeln!(out)
        }
        E::TupleCtor { body } => {
            write!(out, "{{")?;
            for e in body {
                unparse_expr(e, 0, out)?;
                write!(out, ", ")?;
            }
            write!(out, "}}")
        }
        E::StructCtor { name, body, types } => {
            writeln!(out, "{} {{", INT.fetch(*name))?;
            for (nm, ty) in types {
                let tname = INT.fetch_type(*ty).get_name();
                write!(out, "type {} = {}", INT.fetch(*nm), tname)?;
                writeln!(out, ",")?;
            }
            for (nm, expr) in body {
                write!(out, "{} = ", INT.fetch(*nm))?;
                unparse_expr(expr, 0, out)?;
                writeln!(out, ",")?;
            }
            write!(out, "}}")
        }
        E::TupleRef { expr, elt } => {
            unparse_expr(&*expr, indent, out)?;
            write!(out, ".{}", elt)
        }
        E::StructRef { expr, elt } => {
            unparse_expr(&*expr, indent, out)?;
            write!(out, ".{}", INT.fetch(*elt))
        }
        E::Ref { expr } => {
            unparse_expr(&*expr, indent, out)?;
            write!(out, "&")
        }
        E::Deref { expr } => {
            unparse_expr(&*expr, indent, out)?;
            write!(out, "^")
        }
        E::Assign { lhs, rhs } => {
            unparse_expr(&*lhs, indent, out)?;
            write!(out, " = ")?;
            unparse_expr(&*rhs, indent, out)
        }
    }
}

/// Take the AST and produce a formatted string of source code.
pub fn unparse(ast: &Ast, out: &mut dyn io::Write) -> io::Result<()> {
    for decl in ast.decls.iter() {
        unparse_decl(decl, out)?;
        writeln!(out)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::format::unparse;
    use std::io::Cursor;
    #[test]
    fn test_reparse() {
        let src = r#"fn test(x: I32): I32 =
    3 * x + 2
end

"#;
        let ast = {
            let mut parser = crate::parser::Parser::new("unittest", src);
            parser.parse()
        };
        let mut output = Cursor::new(vec![]);
        unparse(&ast, &mut output).unwrap();
        let output_str = String::from_utf8(output.into_inner()).unwrap();
        assert_eq!(src, output_str);
    }

    // TODO: moar tests
}
