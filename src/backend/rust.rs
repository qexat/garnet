//! Compile our HIR to Rust.
//! This is kinda silly, but hey why not.
//!
//!
//! Potential improvements:
//!  * Use something smarter than strings to collect output -- could just
//!    output to a stream like the formatter does
//!  * Use `syn` or something to generate tokens for output rather than strings

use std::borrow::Cow;

//use crate::lir;
use crate::hir;
use crate::*;

/// Whatever prefix stuff we want in the Rust program.
fn prelude() -> &'static str {
    r#"
// ahahahahaha wow
"#
}

fn compile_typedef(cx: &Cx, td: &TypeDef) -> Cow<'static, str> {
    use crate::TypeDef::*;
    match td {
        SInt(4) => "i32".to_owned().into(),
        SInt(_) => unimplemented!(),
        Bool => "bool".into(),
        Tuple(types) => {
            let mut accm = String::from("(");
            for typ in types {
                accm += &compile_typedef(cx, &*cx.fetch_type(*typ));
                accm += ", ";
            }
            accm += ")";
            accm.into()
        }
        Lambda(_params, _ret) => unimplemented!(),
        Ptr(_) => unimplemented!(),
    }
}

pub(super) fn output(cx: &Cx, lir: &hir::Ir<TypeSym>) -> Vec<u8> {
    let mut strings = vec![prelude().to_owned()];
    strings.extend(lir.decls.iter().map(|d| compile_decl(cx, d)));
    let s = strings.join("\n");
    s.into_bytes()
}

fn compile_decl(cx: &Cx, decl: &hir::Decl<TypeSym>) -> String {
    match decl {
        hir::Decl::Function {
            name,
            signature,
            body,
        } => {
            let nstr = &*cx.fetch(*name);
            let sstr = compile_fn_signature(cx, signature);
            let bstr = compile_exprs(cx, body, ";\n");
            format!("fn {}{} {{\n{}\n}}\n", nstr, sstr, bstr)
        }
        hir::Decl::Const {
            name,
            typename,
            init,
        } => {
            let nstr = &*cx.fetch(*name);
            let tstr = compile_typedef(cx, &*cx.fetch_type(*typename));
            let istr = compile_expr(cx, init);
            format!("const {}: {} = {};", nstr, tstr, istr)
        }
    }
}

fn compile_fn_signature(cx: &Cx, sig: &ast::Signature) -> String {
    let mut accm = String::from("(");
    for (varsym, typesym) in sig.params.iter() {
        accm += &*cx.fetch(*varsym);
        accm += ": ";
        accm += &compile_typedef(cx, &*cx.fetch_type(*typesym));
    }
    accm += ") -> ";
    accm += &compile_typedef(cx, &*cx.fetch_type(sig.rettype));
    accm
}

fn compile_exprs(cx: &Cx, exprs: &[hir::TypedExpr<TypeSym>], separator: &str) -> String {
    let ss: Vec<String> = exprs.iter().map(|e| compile_expr(cx, e)).collect();
    ss.join(separator)
}

fn compile_bop(op: hir::BOp) -> &'static str {
    use hir::BOp;
    match op {
        BOp::Add => "+",
        BOp::Sub => "-",
        BOp::Mul => "*",
        BOp::Div => "/",
        BOp::Mod => "%",

        BOp::Eq => "==",
        BOp::Neq => "!=",
        BOp::Gt => ">",
        BOp::Lt => "<",
        BOp::Gte => ">=",
        BOp::Lte => "<=",

        BOp::And => "&&",
        BOp::Or => "||",
        BOp::Xor => "^",
    }
}
fn compile_uop(op: hir::UOp) -> &'static str {
    use hir::UOp;
    match op {
        UOp::Neg => "-",
        UOp::Not => "! ",
        UOp::Ref => "&",
        UOp::Deref => "*",
    }
}

fn compile_expr(cx: &Cx, expr: &hir::TypedExpr<TypeSym>) -> String {
    use hir::Expr as E;
    match &expr.e {
        E::Lit {
            val: ast::Literal::Integer(i),
        } => format!("{}", i),
        E::Lit {
            val: ast::Literal::Bool(b),
        } => format!("{}", b),
        E::Var { name } => cx.fetch(*name).as_ref().to_owned(),
        E::BinOp { op, lhs, rhs } => format!(
            "({} {} {})",
            compile_expr(cx, lhs),
            compile_bop(*op),
            compile_expr(cx, rhs)
        ),
        E::UniOp { op, rhs } => {
            format!("({}{})", compile_uop(*op), compile_expr(cx, rhs))
        }
        E::Block { body } => format!("{{\n{}\n}}", compile_exprs(cx, body, ";\n")),
        E::Let {
            varname,
            typename,
            init,
            mutable,
        } => {
            let vstr = cx.fetch(*varname);
            let tstr = cx.fetch_type(*typename).get_name(cx);
            let istr = compile_expr(cx, init);
            if *mutable {
                format!("let mut {}: {} = {};", vstr, tstr, istr)
            } else {
                format!("let {}: {} = {};", vstr, tstr, istr)
            }
        }
        E::If { cases, falseblock } => {
            let mut accm = String::new();
            for (cond, body) in cases {
                accm += "if ";
                accm += &compile_expr(cx, cond);
                accm += " {\n";
                accm += &compile_exprs(cx, &body, ";\n");
                accm += "} \n";
            }
            accm += "else {\n";
            accm += &compile_exprs(cx, falseblock, ";\n");
            accm += "}\n";
            accm
        }
        E::Loop { body } => {
            format!("loop {{\n{}\n}}\n", compile_exprs(cx, body, ";\n"))
        }
        // TODO: We don't have closures, lambda are just functions, so...
        E::Lambda { signature, body } => {
            format!(
                "fn {} {{ {} }}",
                compile_fn_signature(cx, signature),
                compile_exprs(cx, body, ";\n")
            )
        }
        E::Funcall { func, params } => {
            let nstr = compile_expr(cx, func);
            let pstr = compile_exprs(cx, params, ", ");
            format!("{}({})", nstr, pstr)
        }
        E::Break => {
            format!("break;")
        }
        E::Return { retval } => {
            format!("return {};", compile_expr(cx, retval))
        }
        E::TupleCtor { body } => {
            // We *don't* want join() here, we want to append comma's so that
            // `(foo,)` works properly.
            let mut accm = String::from("(");
            for expr in body {
                accm += &compile_expr(cx, expr);
                accm += ", ";
            }
            accm += ")";
            accm
        }
        E::TupleRef { expr, elt } => {
            format!("{}.{}", compile_expr(cx, expr), elt)
        }
        E::Assign { lhs, rhs } => {
            format!("{} = {};", compile_expr(cx, lhs), compile_expr(cx, rhs))
        }
        E::Deref { expr } => format!("*{}", compile_expr(cx, expr)),
        E::Ref { expr } => format!("&{}", compile_expr(cx, expr)),
    }
}
