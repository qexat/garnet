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
            doc_comment,
        } => {
            for line in doc_comment.iter() {
                // No writeln, doc comment strings already end in \n
                write!(out, "--- {}", line)?;
            }
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
            doc_comment,
        } => {
            for line in doc_comment.iter() {
                write!(out, "--- {}", line)?;
            }
            let name = cx.fetch(*name);
            let tname = cx.fetch_type(*typename).get_name(cx);
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
        let tname = cx.fetch_type(*typename).get_name(cx);
        write!(out, "{}: {}", name, tname)?;
    }
    write!(out, "): ")?;
    let rettype = cx.fetch_type(sig.rettype).get_name(cx);
    write!(out, "{}", rettype)
}

fn unparse_exprs(
    cx: &Cx,
    exprs: &[Expr],
    indent: usize,
    out: &mut dyn io::Write,
) -> io::Result<()> {
    for expr in exprs {
        unparse_expr(cx, expr, indent, out)?;
    }
    Ok(())
}
fn unparse_expr(cx: &Cx, e: &Expr, indent: usize, out: &mut dyn io::Write) -> io::Result<()> {
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
            let name = cx.fetch(*name);
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
            unparse_expr(cx, rhs, 0, out)
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
            mutable,
        } => {
            let name = cx.fetch(*varname);
            let tname = cx.fetch_type(*typename).get_name(cx);
            write!(out, "let ")?;
            if *mutable {
                write!(out, "mut ")?;
            }
            write!(out, "{}: {} = ", name, tname)?;
            unparse_expr(cx, init, 0, out)?;
            writeln!(out)
        }
        E::If { cases, falseblock } => {
            assert!(cases.len() > 1);
            let first_case = &cases[0];
            write!(out, "if ")?;
            unparse_expr(cx, &*first_case.condition, 0, out)?;
            writeln!(out, "then")?;
            unparse_exprs(cx, &first_case.body, indent + 1, out)?;

            for case in &cases[1..] {
                write!(out, "elseif ")?;
                unparse_expr(cx, &case.condition, 0, out)?;
                writeln!(out, "then")?;
                unparse_exprs(cx, &case.body, indent + 1, out)?;
            }
            if falseblock.len() > 0 {
                writeln!(out, "else")?;
                unparse_exprs(cx, falseblock, indent + 1, out)?;
            }
            writeln!(out, "end")
        }
        E::Loop { body } => {
            writeln!(out, "loop")?;
            unparse_exprs(cx, body, indent + 1, out)?;
            writeln!(out, "")?;
            writeln!(out, "end")
        }
        E::Lambda { signature, body } => {
            write!(out, "fn(")?;
            unparse_sig(cx, signature, out)?;
            writeln!(out, " =")?;
            unparse_exprs(cx, body, indent + 1, out)?;
            writeln!(out, "")?;
            writeln!(out, "end")
        }
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
        E::Return { retval: e } => {
            write!(out, "return ")?;
            unparse_expr(cx, e, 0, out)?;
            writeln!(out)
        }
        E::TupleCtor { body } => {
            write!(out, "{{")?;
            for e in body {
                unparse_expr(cx, e, 0, out)?;
                write!(out, ", ")?;
            }
            write!(out, "}}")
        }
        E::TupleRef { expr, elt } => {
            unparse_expr(cx, &*expr, indent, out)?;
            write!(out, ".{}", elt)
        }
        E::Ref { expr } => {
            unparse_expr(cx, &*expr, indent, out)?;
            write!(out, "&")
        }
        E::Deref { expr } => {
            unparse_expr(cx, &*expr, indent, out)?;
            write!(out, "^")
        }
        E::Assign { lhs, rhs } => {
            unparse_expr(cx, &*lhs, indent, out)?;
            write!(out, " = ")?;
            unparse_expr(cx, &*rhs, indent, out)
        }
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

// //////////////// LIR formatting functions //////////////////////////////
fn display_instr(cx: &Cx, f: &Instr, out: &mut dyn io::Write) -> io::Result<()> {
    let display_op = |op: &Op, out: &mut dyn io::Write| -> io::Result<()> {
        match op {
            Op::ValI32(x) => write!(out, "const {}", x)?,
            Op::ValUnit => write!(out, "const unit")?,

            Op::BinOpI32(bop, v1, v2) => write!(out, "binop.i32 {:?}, {:?}, {:?}", bop, v1, v2)?,
            Op::UniOpI32(uop, v) => write!(out, "uniop.i32 {:?}, {:?}", uop, v)?,

            Op::GetLocal(name) => write!(out, "getlocal {}", cx.fetch(*name))?,
            Op::SetLocal(name, v) => write!(out, "setlocal {}, {:?}", cx.fetch(*name), v)?,
            Op::AddrOf(v) => write!(out, "addrof {:?}", v)?,
            Op::LoadI32(v) => write!(out, "load.i32 {:?}", v)?,
            Op::LoadOffsetI32(v1, v2) => write!(out, "loadoffset.i32 {:?} + {:?}", v1, v2)?,
            Op::StoreI32(v1, v2) => write!(out, "store.i32 {:?} {:?}", v1, v2)?,
            Op::StoreOffsetI32(v1, v2, v3) => {
                write!(out, "storeoffset.i32 {:?} {:?} + {:?}", v1, v2, v3)?
            }

            Op::Phi(bbs) => write!(out, "PHI {:?}", bbs)?,
            Op::Call(name, args) => write!(out, "call {} {:?}", cx.fetch(*name), args)?,
            Op::CallIndirect(v, args) => write!(out, "call indirect {:?} {:?}", v, args)?,
        }
        Ok(())
    };
    // Indentation is fixed here, makes life easy.
    match f {
        Instr::Assign(var, typesym, op) => {
            write!(
                out,
                "    {:?}: {} = ",
                var,
                cx.fetch_type(*typesym).get_name(cx)
            )?;
            display_op(op, out)?;
            writeln!(out)?;
        }
    }
    Ok(())
}

use crate::lir::*;
fn display_func(cx: &Cx, f: &Func, out: &mut dyn io::Write) -> io::Result<()> {
    /*
    writeln!(
        out,
        "Function {}, signature {:?}, params {:?}, returns {}",
        cx.fetch(f.name),
        f.signature,
        f.params,
        cx.fetch_type(f.returns).get_name(cx),
    )?;
    */

    write!(out, "Function {}", cx.fetch(f.name),)?;
    unparse_sig(cx, &f.signature, out)?;
    writeln!(out)?;
    writeln!(out, "Locals: {:?}", f.locals)?;
    writeln!(out, "Frame layout: {:?}", f.frame_layout)?;
    writeln!(out, "Entry point: {:?}", f.entry)?;
    for (id, bb) in &f.body {
        writeln!(out, "  {:?}", id)?;
        for instr in &bb.body {
            display_instr(cx, instr, out)?;
        }
        match &bb.terminator {
            Branch::Jump(v, bb) => writeln!(out, "  jump {:?}, {:?}", v, bb)?,
            Branch::Branch(v, bb1, bb2) => writeln!(out, "  branch {:?}, {:?}, {:?}", v, bb1, bb2)?,
            Branch::Return(v) => writeln!(out, "  return {:?}", v)?,
            Branch::Unreachable => writeln!(out, "  unreachable")?,
        }
    }
    Ok(())
}

/// dump LIR to a string for debugging
/// Easier than
pub fn display_lir(cx: &Cx, lir: &Lir, out: &mut dyn io::Write) -> io::Result<()> {
    for f in &lir.funcs {
        display_func(cx, f, out)?;
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

    // TODO: moar tests
}
