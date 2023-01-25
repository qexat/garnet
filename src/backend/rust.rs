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
use crate::typeck::Tck;
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
fn compile_typedef(td: &Type) -> Cow<'static, str> {
    compile_typename(td)
}

/// Similar to `compile_typedef` but only gets names, not full definitions.
///
/// Needed for when we do `let x: Foo = ...` rather than `struct Foo { ... }`
fn compile_typename(_td: &Type) -> Cow<'static, str> {
    todo!()
    /*
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
                accm += &compile_typename(&*typ);
                accm += ", ";
            }
            accm += ")";
            accm.into()
        }
        Lambda {
            generics,
            params,
            rettype,
        } => {
            let mut accm = String::from("fn ");
            // TODO: ...make sure this actually works.
            if generics.len() > 0 {
                accm += "<";
                for g in generics {
                    accm += &*INT.fetch(*g);
                    accm += ", ";
                }
                accm += ">";
            }
            accm += "(";
            for p in params {
                accm += &compile_typename(&*p);
                accm += ", ";
            }
            accm += ") -> ";
            accm += &compile_typename(&*rettype);
            accm.into()
        }
        //Named(sym) => (&*INT.fetch(*sym)).clone().into(),
        Struct { fields } => {
            // We compile our structs into Rust tuples.
            // Our fields are always ordered via BTreeMap etc,
            // so it's okay
            /*
            if typefields.len() > 0 {
                todo!("Figure out type thingies");
            }
            */
            let mut accm = String::from("(");
            for (nm, typ) in fields.iter() {
                accm += &format!("/* {} */
    {}, \n", INT.fetch(*nm), compile_typename(&*typ));
            }
            accm += ")";
            accm.into()
        }
        Enum { variants: _ } => {
            todo!("Enums probably should be lowered to numbers?")
        }
        NamedTypeVar(vsym) => {
            let s = &*INT.fetch(*vsym);
            mangle_name(s).into()
        }
    }
*/
}

pub(super) fn output(lir: &hir::Ir, tck: &Tck) -> Vec<u8> {
    let mut output = Vec::new();
    output.extend(prelude().as_bytes());
    for decl in lir.decls.iter() {
        compile_decl(&mut output, decl, tck)
            .expect("IO error writing output code.  Out of memory???");
    }
    output
}

/// Mangle/unmangle a name for a function.
/// We need this for lambda's, migth need it for other things, we'll see.
/// TODO: There might be a better way to make lambda's un-nameable.
/// Probably, really.
fn mangle_name(s: &str) -> String {
    s.replace("!", "__")
}

fn compile_decl(w: &mut impl Write, decl: &hir::Decl, tck: &Tck) -> io::Result<()> {
    match decl {
        hir::Decl::Function {
            name,
            signature,
            body,
        } => {
            let nstr = mangle_name(&*INT.fetch(*name));
            let sstr = compile_fn_signature(signature);
            let bstr = compile_exprs(body, ";\n", tck);
            writeln!(w, "pub fn {}{} {{\n{}\n}}\n", nstr, sstr, bstr)
        }
        hir::Decl::Const {
            name,
            typename,
            init,
        } => {
            let nstr = mangle_name(&INT.fetch(*name));
            let tstr = compile_typedef(&*typename);
            let istr = compile_expr(init, tck);
            writeln!(w, "const {}: {} = {};", nstr, tstr, istr)
        }
        // Typedefs compile into newtype structs.
        hir::Decl::TypeDef { name, typedecl } => {
            let nstr = mangle_name(&INT.fetch(*name));
            let tstr = compile_typedef(&*typedecl);
            writeln!(w, "pub struct {}({});", nstr, tstr)
        }
        /*
        hir::Decl::StructDef {
            name,
            fields,
            typefields: _,
        } => {
            let nstr = mangle_name(&INT.fetch(*name));
            let mut accm = String::new();
            for (n, t) in fields {
                accm += &format!(
                    "{}: {},\n",
                    &INT.fetch(*n),
                    compile_typedef(&*t)
                );
            }
            writeln!(w, "pub struct {} {{\n {} }}\n", nstr, accm)
        }
        */
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
                params: vec![(INT.intern("input"), signature.rettype.clone())],
                rettype: signature.params[0].1.clone(),
            };
            let sig_str = compile_fn_signature(&destructure_signature);
            writeln!(w, "fn {}_unwrap{} {{ input.0 }}\n", nstr, sig_str,)
        }
    }
}

fn compile_fn_signature(sig: &ast::Signature) -> String {
    let mut accm = String::from("");
    todo!();
    /*
    if sig.generics.len() > 0 {
        accm += "<";
        for generic in sig.generics.iter() {
            accm += &mangle_name(&*INT.fetch(*generic));
            accm += ", ";
        }
        accm += ">";
    }
    accm += "(";
    for (varsym, typesym) in sig.params.iter() {
        accm += &*INT.fetch(*varsym);
        accm += ": ";
        accm += &compile_typename(&*typesym);
        accm += ", ";
    }
    accm += ") -> ";
    accm += &compile_typename(&*INT.fetch_type(sig.rettype));
    accm
    */
}

fn compile_exprs(exprs: &[hir::ExprNode], separator: &str, tck: &Tck) -> String {
    let ss: Vec<String> = exprs.iter().map(|e| compile_expr(e, tck)).collect();
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

fn compile_expr(expr: &hir::ExprNode, tck: &Tck) -> String {
    use hir::Expr as E;
    match &expr.e {
        E::Lit {
            val: ast::Literal::Integer(i),
        } => format!("{}", i),
        E::EnumLit { val: _val, ty: _ty } => todo!(),
        E::Lit {
            val: ast::Literal::Bool(b),
        } => format!("{}", b),
        E::Lit {
            val: ast::Literal::SizedInteger { vl, bytes },
        } => {
            let bits = bytes * 8;
            format!("{}i{}", vl, bits)
        }
        E::Var { name, .. } => mangle_name(&*INT.fetch(*name)),
        E::BinOp { op, lhs, rhs } => format!(
            "({} {} {})",
            compile_expr(lhs, tck),
            compile_bop(*op),
            compile_expr(rhs, tck)
        ),
        E::UniOp { op, rhs } => {
            format!("({}{})", compile_uop(*op), compile_expr(rhs, tck))
        }
        E::Block { body } => format!("{{\n{}\n}}", compile_exprs(body, ";\n", tck)),
        E::Let {
            varname,
            typename,
            init,
            mutable,
        } => {
            let vstr = mangle_name(&*INT.fetch(*varname));
            let tstr = compile_typename(&*typename);
            let istr = compile_expr(init, tck);
            if *mutable {
                format!("let mut {}: {} = {};", vstr, tstr, istr)
            } else {
                format!("let {}: {} = {};", vstr, tstr, istr)
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
                accm += &compile_expr(cond, tck);
                accm += " {\n";
                accm += &compile_exprs(&body, ";\n", tck);
                accm += "} \n";
            }
            accm += "else {\n";
            accm += &compile_exprs(&falseblock, ";\n", tck);
            accm += "}\n";
            accm
        }
        E::Loop { body } => {
            format!("loop {{\n{}\n}}\n", compile_exprs(body, ";\n", tck))
        }
        // TODO: We don't have closures, lambda are just functions, so...
        E::Lambda { signature, body } => {
            format!(
                "fn {} {{ {} }}",
                compile_fn_signature(signature),
                compile_exprs(body, ";\n", tck)
            )
        }
        E::Funcall {
            func,
            params,
            generic_types,
        } => {
            // We have to store an intermediate value for the func, 'cause
            // Rust has problems with things like this:
            // fn f1() -> i32 { 1 }
            // fn f2() -> i32 { 10 }
            // fn f() -> i32 { if true { f1 } else { f2 }() }
            //
            // Should this happen in an IR lowering step?  idk.
            let nstr = compile_expr(func, tck);
            let pstr = compile_exprs(params, ", ", tck);
            format!("{{ let __dummy = {}; __dummy({}) }}", nstr, pstr)
        }
        E::Break => "break;".to_string(),
        E::Return { retval } => {
            format!("return {};", compile_expr(retval, tck))
        }
        /*
        E::TupleCtor { body } => {
            // We *don't* want join() here, we want to append comma's so that
            // `(foo,)` works properly.
            let mut accm = String::from("(");
            for expr in body {
                accm += &compile_expr(expr, tck);
                accm += ", ";
            }
            accm += ")";
            accm
        }
        */
        // Unit type
        E::StructCtor { body } if body.len() == 0 => String::from(" ()\n"),
        E::StructCtor { body } => {
            /*
            if types.len() > 0 {
                todo!("not implemented")
            }
            */
            let mut accm = String::from("(\n");
            for (nm, expr) in body {
                accm += &format!("/* {} */ {}, \n", INT.fetch(*nm), compile_expr(expr, tck));
            }
            accm += ")\n";
            accm
        }
        E::StructRef { expr, elt } => {
            // We turn our structs into Rust tuples, so we need to
            // to turn our field names into indices
            todo!()
            /*
            let tv = tck.get_typevar_for_expression(expr).unwrap();
            let tdef = tck.follow_typevar(tv).unwrap();
            if let Type::Struct(body, _generics) = tdef {
                let mut nth = 9999_9999;
                for (i, (nm, _ty)) in body.iter().enumerate() {
                    if nm == elt {
                        nth = i;
                        break;
                    }
                }
                format!("{}.{}", compile_expr(expr, tck), nth)
            } else {
                panic!(
                    "Struct wasn't actually a struct in backend, was {}.  should never happen",
                    compile_typename(tdef)
                )
            }
            */
        }
        E::Assign { lhs, rhs } => {
            format!("{} = {}", compile_expr(lhs, tck), compile_expr(rhs, tck))
        }
        E::Deref { expr } => format!("*{}", compile_expr(expr, tck)),
        E::Ref { expr } => format!("&{}", compile_expr(expr, tck)),
    }
}
