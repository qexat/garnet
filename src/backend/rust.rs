//! Compile our HIR to Rust.
//! This is kinda silly, but hey why not.
//!
//!
//! Potential improvements:
//!  * Use something smarter than strings to collect output -- could just
//!    output to a stream like the formatter does.  This is a little weird though 'cause
//!    we often want to build pieces of code, and then combine them together at the end.
//!  * Use `syn` or something to generate tokens for output rather than strings

use std::borrow::Cow;
use std::io::{self, Write};

use crate::hir;
use crate::*;

/// Whatever prefix stuff we want in the Rust program.
fn prelude() -> &'static str {
    r#"
fn __println(x: i32) {
    println!("{}", x);
}

fn __println_bool(x: bool) {
    println!("{}", x);
}

fn __println_i64(x: i64) {
    println!("{}", x);
}

fn __println_i16(x: i16) {
    println!("{}", x);
}
"#
}

/// Compiles a `TypeDef` into a declaration statement.
fn compile_typedef(td: &TypeDef) -> Cow<'static, str> {
    use crate::TypeDef::*;
    match td {
        Struct { name, fields, .. } => {
            let mut accm = String::from("struct ");
            let name = &*INT.fetch(*name);
            accm += name;
            accm += "{ \n";
            for (nm, ty) in fields {
                let nm_str = &*INT.fetch(*nm);
                let vl_str = INT.fetch_type(*ty);
                accm += nm_str;
                accm += ": ";
                accm += &vl_str.get_name();
                accm += ",\n";
            }
            accm += "}\n";
            accm.into()
        }
        _other => compile_typename(td),
    }
}

/// Similar to `compile_typedef` but only gets names, not full definitions.
///
/// Needed for when we do `let x: Foo = ...` rather than `struct Foo { ... }`
fn compile_typename(td: &TypeDef) -> Cow<'static, str> {
    use crate::TypeDef::*;
    match td {
        SInt(16) => "i128".into(),
        SInt(8) => "i64".into(),
        SInt(4) => "i32".into(),
        SInt(2) => "i16".into(),
        SInt(1) => "i8".into(),
        SInt(e) => {
            unreachable!("Invalid integer size: {}", e)
        }
        UnknownInt => unreachable!("Backend got an integer of unknown size, should never happen!"),
        Bool => "bool".into(),
        Never => panic!("Stable rust can't quite do this yet..."),
        Tuple(types) => {
            let mut accm = String::from("(");
            for typ in types {
                accm += &compile_typename(&*INT.fetch_type(*typ));
                accm += ", ";
            }
            accm += ")";
            accm.into()
        }
        Lambda(params, ret) => {
            let mut accm = String::from("fn (");
            for p in params {
                accm += &compile_typename(&*INT.fetch_type(*p));
                accm += ", ";
            }
            accm += ") -> ";
            accm += &compile_typename(&*INT.fetch_type(*ret));
            accm.into()
        }
        Named(sym) => (&*INT.fetch(*sym)).clone().into(),
        Struct { name, .. } => (&*INT.fetch(*name)).clone().into(),
    }
}

pub(super) fn output(lir: &hir::Ir<TypeSym>) -> Vec<u8> {
    let mut output = Vec::new();
    output.extend(prelude().as_bytes());
    for decl in lir.decls.iter() {
        compile_decl(&mut output, decl).expect("IO error writing output code.  Out of memory???");
    }
    output
}

/// Mangle/unmangle a name for a function.
/// We need this for lambda's, migth need it for other things, we'll see.
/// TODO: There might be a better way to make lambda's un-nameable.
/// Probably, really.
fn mangle_name(s: &str) -> String {
    s.replace("@", "__")
}

fn compile_decl(w: &mut impl Write, decl: &hir::Decl<TypeSym>) -> io::Result<()> {
    match decl {
        hir::Decl::Function {
            name,
            signature,
            body,
        } => {
            let nstr = mangle_name(&*INT.fetch(*name));
            let sstr = compile_fn_signature(signature);
            let bstr = compile_exprs(body, ";\n");
            writeln!(w, "pub fn {}{} {{\n{}\n}}\n", nstr, sstr, bstr)
        }
        hir::Decl::Const {
            name,
            typename,
            init,
        } => {
            let nstr = mangle_name(&INT.fetch(*name));
            let tstr = compile_typedef(&*INT.fetch_type(*typename));
            let istr = compile_expr(init);
            writeln!(w, "const {}: {} = {};", nstr, tstr, istr)
        }
        // Typedefs compile into newtype structs.
        hir::Decl::TypeDef { name, typedecl } => {
            let nstr = mangle_name(&INT.fetch(*name));
            let tstr = compile_typedef(&*INT.fetch_type(*typedecl));
            writeln!(w, "pub struct {}({});", nstr, tstr)
        }
        hir::Decl::StructDef { name, fields } => {
            let nstr = mangle_name(&INT.fetch(*name));
            let mut accm = String::new();
            for (n, t) in fields {
                accm += &format!(
                    "{}: {},\n",
                    &INT.fetch(*n),
                    compile_typedef(&*INT.fetch_type(*t))
                );
            }
            writeln!(w, "pub struct {} {{\n {} }}\n", nstr, accm)
        }
        // For these we have to look at the signature and make a
        // function that constructs a struct or tuple or whatever
        // out of it.
        hir::Decl::Constructor { name, signature } => {
            // Rust's newtype struct constructors are already functions,
            // so we don't need to actually output a separate constructor
            // function.  Huh.
            let typename = &*INT.fetch(*name);
            let nstr = mangle_name(typename);
            // We do need to make the destructure function I guess.
            let destructure_signature = hir::Signature {
                params: vec![(INT.intern("input"), signature.rettype)],
                rettype: signature.params[0].1,
            };
            let sig_str = compile_fn_signature(&destructure_signature);
            writeln!(w, "fn {}_unwrap{} {{ input.0 }}\n", nstr, sig_str,)
        }
    }
}

fn compile_fn_signature(sig: &ast::Signature) -> String {
    let mut accm = String::from("(");
    for (varsym, typesym) in sig.params.iter() {
        accm += &*INT.fetch(*varsym);
        accm += ": ";
        accm += &compile_typename(&*INT.fetch_type(*typesym));
        accm += ", ";
    }
    accm += ") -> ";
    accm += &compile_typename(&*INT.fetch_type(sig.rettype));
    accm
}

fn compile_exprs(exprs: &[hir::TypedExpr<TypeSym>], separator: &str) -> String {
    let ss: Vec<String> = exprs.iter().map(|e| compile_expr(e)).collect();
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

fn compile_expr(expr: &hir::TypedExpr<TypeSym>) -> String {
    use hir::Expr as E;
    match &expr.e {
        E::Lit {
            val: ast::Literal::Integer(i),
        } => format!("{}", i),
        E::Lit {
            val: ast::Literal::Bool(b),
        } => format!("{}", b),
        E::Lit {
            val: ast::Literal::SizedInteger { vl, bytes },
        } => {
            let bits = bytes * 8;
            format!("{}i{}", vl, bits)
        }
        E::Var { name } => mangle_name(&*INT.fetch(*name)),
        E::BinOp { op, lhs, rhs } => format!(
            "({} {} {})",
            compile_expr(lhs),
            compile_bop(*op),
            compile_expr(rhs)
        ),
        E::UniOp { op, rhs } => {
            format!("({}{})", compile_uop(*op), compile_expr(rhs))
        }
        E::Block { body } => format!("{{\n{}\n}}", compile_exprs(body, ";\n")),
        E::Let {
            varname,
            typename,
            init,
            mutable,
        } => {
            let vstr = mangle_name(&*INT.fetch(*varname));
            let tstr = compile_typename(&*INT.fetch_type(*typename));
            let istr = compile_expr(init);
            if *mutable {
                format!("let mut {}: {} = {}", vstr, tstr, istr)
            } else {
                format!("let {}: {} = {}", vstr, tstr, istr)
            }
        }
        E::If { cases } => {
            let mut accm = String::new();
            let falseblock = cases.last().unwrap().1.clone();
            let n = cases.len();
            for (i, (cond, body)) in cases[..(n - 1)].iter().enumerate() {
                if i == 0 {
                    accm += "if ";
                } else {
                    accm += " else if "
                }
                accm += &compile_expr(cond);
                accm += " {\n";
                accm += &compile_exprs(&body, ";\n");
                accm += "} \n";
            }
            accm += "else {\n";
            accm += &compile_exprs(&falseblock, ";\n");
            accm += "}\n";
            accm
        }
        E::Loop { body } => {
            format!("loop {{\n{}\n}}\n", compile_exprs(body, ";\n"))
        }
        // TODO: We don't have closures, lambda are just functions, so...
        E::Lambda { signature, body } => {
            format!(
                "fn {} {{ {} }}",
                compile_fn_signature(signature),
                compile_exprs(body, ";\n")
            )
        }
        E::Funcall { func, params } => {
            // We have to store an intermediate value for the func, 'cause
            // Rust has problems with things like this:
            // fn f1() -> i32 { 1 }
            // fn f2() -> i32 { 10 }
            // fn f() -> i32 { if true { f1 } else { f2 }() }
            //
            // Should this happen in an IR lowering step?  idk.
            let nstr = compile_expr(func);
            let pstr = compile_exprs(params, ", ");
            format!("{{ let __dummy = {}; __dummy({}) }}", nstr, pstr)
        }
        E::Break => "break;".to_string(),
        E::Return { retval } => {
            format!("return {};", compile_expr(retval))
        }
        E::TupleCtor { body } => {
            // We *don't* want join() here, we want to append comma's so that
            // `(foo,)` works properly.
            let mut accm = String::from("(");
            for expr in body {
                accm += &compile_expr(expr);
                accm += ", ";
            }
            accm += ")";
            accm
        }
        E::StructCtor { name, body } => {
            let mut accm = String::from(&*INT.fetch(*name));
            accm += " {\n";
            for (nm, vl) in body {
                accm += &*INT.fetch(*nm);
                accm += ": ";
                accm += &compile_expr(vl);
                accm += ",\n";
            }
            accm += "}\n";
            accm
        }
        E::TupleRef { expr, elt } => {
            format!("{}.{}", compile_expr(expr), elt)
        }
        E::StructRef { expr, elt } => {
            format!("{}.{}", compile_expr(expr), INT.fetch(*elt))
        }
        E::Assign { lhs, rhs } => {
            format!("{} = {}", compile_expr(lhs), compile_expr(rhs))
        }
        E::Deref { expr } => format!("*{}", compile_expr(expr)),
        E::Ref { expr } => format!("&{}", compile_expr(expr)),
    }
}
