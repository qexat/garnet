//! Basic formatter and pretty-printer.
//! Takes an AST and outputs to a Write object.

use std::io;

use crate::ast::*;

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
                write!(out, "---{}", line)?;
            }
            let name = name.val();
            write!(out, "fn {}", name)?;
            unparse_sig(signature, out)?;
            writeln!(out, " =")?;
            unparse_exprs(body, 1, out)?;
            write!(out, "end")
        }
        Decl::Const {
            name,
            typename,
            init,
            doc_comment,
        } => {
            for line in doc_comment.iter() {
                write!(out, "---{}", line)?;
            }
            let name = name.val();
            let tname = typename.get_name();
            write!(out, "const {} {} = ", name, tname)?;
            unparse_expr(init, 0, out)
        }
        Decl::TypeDef {
            name,
            params,
            typedecl,
            doc_comment,
        } => {
            for line in doc_comment.iter() {
                write!(out, "---{}", line)?;
            }
            let name = name.val();
            let tname = typedecl.get_name();
            if params.is_empty() {
                writeln!(out, "type {} = {}", name, tname)?;
            } else {
                let mut paramstr = String::from("");
                let mut first = true;
                for t in params {
                    if !first {
                        paramstr += ", ";
                    } else {
                        first = false;
                    }
                    paramstr += &*t.val();
                }
                writeln!(out, "type {}({}) = {}", name, paramstr, tname)?;
            }
            writeln!(out)
        }
        Decl::Import { name, rename } => {
            if let Some(re) = rename {
                write!(out, "import {} as {}", name.val(), re.val())
            } else {
                write!(out, "import {}", name.val())
            }
        }
    }
}

fn unparse_sig(sig: &Signature, out: &mut dyn io::Write) -> io::Result<()> {
    write!(out, "(")?;
    // Type parameters
    if !sig.typeparams.is_empty() {
        write!(out, "|")?;
        let mut first = true;
        for name in sig.typeparams.iter() {
            if !first {
                write!(out, ", ")?;
            } else {
                first = false;
            }
            write!(out, "{}", name)?;
        }
        write!(out, "| ")?;
    }

    // Write (foo I32, bar I16)
    // not (foo I32, bar I16, )
    let mut first = true;
    for (name, typename) in sig.params.iter() {
        if !first {
            write!(out, ", ")?;
        } else {
            first = false;
        }
        let name = name.val();
        let tname = typename.get_name();
        write!(out, "{} {}", name, tname)?;
    }
    write!(out, ") ")?;
    let rettype = sig.rettype.get_name();
    write!(out, "{}", rettype)
}

fn unparse_exprs(exprs: &[Expr], indent: usize, out: &mut dyn io::Write) -> io::Result<()> {
    for expr in exprs {
        unparse_expr(expr, indent, out)?;
        writeln!(out)?;
    }
    Ok(())
}

fn write_indent(indent: usize, out: &mut dyn io::Write) -> io::Result<()> {
    for _ in 0..(indent * INDENT_SIZE) {
        write!(out, " ")?;
    }
    Ok(())
}

fn unparse_expr(e: &Expr, indent: usize, out: &mut dyn io::Write) -> io::Result<()> {
    use Expr as E;
    write_indent(indent, out)?;
    match e {
        E::Lit { val } => write!(out, "{}", val),
        E::Var { name } => {
            let name = name.val();
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
            write_indent(indent, out)?;
            writeln!(out, "end")
        }
        E::Let {
            varname,
            typename,
            init,
            mutable,
        } => {
            let name = varname.val();
            write!(out, "let ")?;
            if *mutable {
                write!(out, "mut ")?;
            }
            write!(out, "{} ", name)?;
            if let Some(t) = typename {
                let tname = t.get_name();
                write!(out, "{} ", tname)?;
            }
            write!(out, "= ")?;

            unparse_expr(init, 0, out)?;
            writeln!(out)
        }
        E::If { cases, falseblock } => {
            assert!(!cases.is_empty());
            let first_case = &cases[0];
            write!(out, "if ")?;
            unparse_expr(&first_case.condition, 0, out)?;
            writeln!(out, " then")?;
            unparse_exprs(&first_case.body, indent + 1, out)?;

            for case in &cases[1..] {
                write_indent(indent, out)?;
                write!(out, "elseif ")?;
                unparse_expr(&case.condition, 0, out)?;
                writeln!(out, " then")?;
                unparse_exprs(&case.body, indent + 1, out)?;
            }
            if !falseblock.is_empty() {
                write_indent(indent, out)?;
                writeln!(out, "else")?;
                unparse_exprs(falseblock, indent + 1, out)?;
            }
            write_indent(indent, out)?;
            write!(out, "end")
        }
        E::Loop { body } => {
            writeln!(out, "loop")?;
            unparse_exprs(body, indent + 1, out)?;
            writeln!(out, "end")
        }
        E::While { cond, body } => {
            write!(out, "while ")?;
            unparse_expr(cond, 0, out)?;
            writeln!(out, " do")?;
            unparse_exprs(body, indent + 1, out)?;
            writeln!(out, "end")
        }
        E::Lambda { signature, body } => {
            write!(out, "fn")?;
            unparse_sig(signature, out)?;
            writeln!(out, " =")?;
            // not sure these indent numbers are a good idea but works for now...
            unparse_exprs(body, indent + 2, out)?;
            write_indent(indent + 1, out)?;
            write!(out, "end")
        }
        E::Funcall {
            func,
            params,
            typeparams,
        } => {
            unparse_expr(func, 0, out)?;
            write!(out, "(")?;
            if !typeparams.is_empty() {
                write!(out, "|")?;
                let mut first = true;
                for t in typeparams {
                    if !first {
                        write!(out, ", ")?;
                    } else {
                        first = false;
                    }
                    let tname = t.get_name();
                    write!(out, "{}", tname)?;
                }
                write!(out, "| ")?;
            }
            let mut first = true;
            for e in params {
                if !first {
                    write!(out, ", ")?;
                } else {
                    first = false;
                }
                unparse_expr(e, 0, out)?;
            }
            write!(out, ")")
        }
        E::Break => {
            write_indent(indent, out)?;
            write!(out, "break")
        }
        E::Return { retval: e } => {
            write_indent(indent, out)?;
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
        E::StructCtor { body } => {
            writeln!(out, "{{")?;
            for (nm, expr) in body {
                write!(out, ".{} = ", nm.val())?;
                unparse_expr(expr, 0, out)?;
                writeln!(out, ",")?;
            }
            write!(out, "}}")
        }
        E::ArrayRef { expr, idx } => {
            unparse_expr(expr, indent, out)?;
            write!(out, "[")?;
            unparse_expr(idx, 0, out)?;
            write!(out, "]")
        }
        E::TupleRef { expr, elt } => {
            unparse_expr(expr, indent, out)?;
            write!(out, ".{}", elt)
        }
        E::StructRef { expr, elt } => {
            unparse_expr(expr, indent, out)?;
            write!(out, ".{}", elt.val())
        }
        E::TypeUnwrap { expr } => {
            unparse_expr(expr, indent, out)?;
            write!(out, "$")
        }
        E::Ref { expr } => {
            unparse_expr(expr, indent, out)?;
            write!(out, "&")
        }
        E::Deref { expr } => {
            unparse_expr(expr, indent, out)?;
            write!(out, "^")
        }
        E::Assign { lhs, rhs } => {
            unparse_expr(lhs, 0, out)?;
            write!(out, " = ")?;
            unparse_expr(rhs, 0, out)
        }
        E::ArrayCtor { body } => {
            writeln!(out, "[")?;
            for expr in body {
                unparse_expr(expr, indent + 1, out)?;
                writeln!(out, ",")?;
            }
            write!(out, "]")
        }
    }
}

/// Take the AST and produce a formatted string of source code.
pub fn unparse(ast: &Ast, out: &mut dyn io::Write) -> io::Result<()> {
    for decl in ast.decls.iter() {
        if !ast.module_docstring.is_empty() {
            writeln!(out, "--- {}", &ast.module_docstring)?;
        }
        writeln!(out)?;
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
        let src = r#"
fn test(x I32) I32 =
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
