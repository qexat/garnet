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

use log::*;

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

/// Compiles a `Type` into a a valid Rust type.
///
/// Needed for when we do `let x: Foo = ...` rather than `struct Foo { ... }`
fn compile_typename(t: &Type) -> Cow<'static, str> {
    use crate::Type::*;
    match t {
        Prim(PrimType::SInt(16)) => "i128".into(),
        Prim(PrimType::SInt(8)) => "i64".into(),
        Prim(PrimType::SInt(4)) => "i32".into(),
        Prim(PrimType::SInt(2)) => "i16".into(),
        Prim(PrimType::SInt(1)) => "i8".into(),
        Prim(PrimType::SInt(e)) => {
            unreachable!("Invalid integer size: {}", e)
        }
        Prim(PrimType::UnknownInt) => {
            unreachable!("Backend got an integer of unknown size, should never happen!")
        }
        Prim(PrimType::Bool) => "bool".into(),
        Prim(PrimType::AnyPtr) => format!("*const u8").into(),
        Named(s, types) if s == &Sym::new("Tuple") => {
            trace!("Compiling tuple {:?}...", t);
            let mut accm = String::from("(");
            for typ in types {
                accm += &compile_typename(&*typ);
                accm += ", ";
            }
            accm += ")";
            accm.into()
        }
        Func(params, rettype, typeparams) => {
            if typeparams.len() > 0 {
                todo!();
            }
            let mut accm = String::from("fn ");
            // TODO: ...make sure this actually works.
            /*
            if generics.len() > 0 {
                accm += "<";
                for g in generics {
                    accm += &*INT.fetch(*g);
                    accm += ", ";
                }
                accm += ">";
            }
            */
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
        Struct(_fields, _generics) => {
            // We compile our structs into Rust tuples.
            // Our fields are always ordered via BTreeMap etc,
            // so it's okay
            /*
            if typefields.len() > 0 {
                todo!("Figure out type thingies");
            }
            */
            //         let mut accm = String::from("(");
            //         for (nm, typ) in fields.iter() {
            //             accm += &format!(
            //                 "/* {} */
            // {}, \n",
            //                 INT.fetch(*nm),
            //                 compile_typename(&*typ)
            //             );
            //         }
            //         accm += ")";
            //         accm.into()
            passes::generate_type_name(t).into()
        }
        Enum(_things) => {
            // Construct names for anonymous enums by concat'ing the member
            // names together.
            // TODO: is this just silly?
            // Lowering enums to numbers first would make this easier tbh
            // We don't know the name of the
            // let mut accm = String::from("Enum");
            // for (nm, _vl) in things {
            //     accm += &*nm.val();
            // }
            // accm.into()
            passes::generate_type_name(t).into()
        }
        Generic(s) => mangle_name(&*s.val()).into(),
        Array(t, len) => format!("[{};{}]", compile_typename(t), len).into(),
        Named(sym, generics) => {
            if generics.len() == 0 {
                format!("{}", sym).into()
            } else {
                let generic_strings: Vec<_> =
                    generics.iter().map(|t| compile_typename(t)).collect();
                let args = generic_strings.join(", ");
                format!("{}<{}>", sym, args).into()
            }
        }
        Sum(_body, _generics) => {
            /*
            let mut accm = String::from("(");
            for (nm, typ) in fields.iter() {
                accm += &format!(
                    "/* {} */
    {}, \n",
                    INT.fetch(*nm),
                    compile_typename(&*typ)
                );
            }
            accm += ")";
            accm.into()
            */
            format!("SomeSum").into()
        }
    }
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
            if body.iter().all(|expr| expr.is_const) {
                trace!("Function is const: {:?}", nstr);
                writeln!(w, "pub const fn {}{} {{\n{}\n}}\n", nstr, sstr, bstr)
            } else {
                trace!("Function is NOT const: {:?}\nBody:{:?}", nstr, body);
                writeln!(w, "pub fn {}{} {{\n{}\n}}\n", nstr, sstr, bstr)
            }
        }
        hir::Decl::Const {
            name,
            typ: typename,
            init,
        } => {
            let nstr = mangle_name(&INT.fetch(*name));
            let tstr = compile_typename(&*typename);
            let istr = compile_expr(init, tck);
            writeln!(w, "const {}: {} = {};", nstr, tstr, istr)
        }
        // Typedefs compile into aliases.
        // I wanted them to compile into newtype structs,
        // but we already define a constructor function and that
        // name clashes with the constructor function that rustc
        // creates with newtype structs.
        // So I guess we have some type erasure here.
        hir::Decl::TypeDef {
            name,
            typedecl,
            params,
        } => {
            match typedecl {
                Type::Enum(_) => {
                    // Enums just become literal integers.
                    assert!(
                        params.is_empty(),
                        "Bruh you have generic params on an enum type"
                    );
                    writeln!(w, "pub type {} = i32;", mangle_name(&*name.val()))?;
                    Ok(())
                }
                Type::Sum(body, _generics) => {
                    let nstr = mangle_name(&*name.val());
                    let param_strings: Vec<_> =
                        params.iter().map(|sym| (&*sym.val()).clone()).collect();
                    let args = param_strings.join(", ");
                    writeln!(w, "pub enum {}<{}> {{ ", nstr, args)?;
                    for (nm, ty) in body {
                        writeln!(w, "    {} ({}),", nm, compile_typename(ty))?;
                    }
                    writeln!(w, "}}")?;
                    Ok(())
                }
                // For everything else we just make a fairly literal alias to the existing type.
                _other => {
                    let nstr = mangle_name(&*name.val());
                    let tstr = compile_typename(typedecl);
                    //writeln!(w, "pub struct {}({});", nstr, tstr)
                    // TODO: <> is a valid generic param decl in Rust
                    if params.len() == 0 {
                        writeln!(w, "pub type {} = {};", nstr, tstr)
                    } else {
                        let param_strings: Vec<_> =
                            params.iter().map(|sym| (&*sym.val()).clone()).collect();
                        let args = param_strings.join(", ");
                        writeln!(w, "pub type {}<{}> = {};", nstr, args, tstr).into()
                    }
                }
            }
        }
        hir::Decl::Import { .. } => todo!(),
    }
}

fn compile_fn_signature(sig: &ast::Signature) -> String {
    let mut accm = String::from("");
    let generics = sig.generic_type_names();

    if generics.len() > 0 {
        accm += "<";
        for generic in generics.iter() {
            accm += &mangle_name(&*generic.val());
            accm += ", ";
        }
        accm += ">";
    }
    accm += "(";
    for (varsym, typesym) in sig.params.iter() {
        accm += &*INT.fetch(*varsym);
        accm += ": ";
        accm += &compile_typename(&typesym);
        accm += ", ";
    }
    accm += ") -> ";
    accm += &compile_typename(&sig.rettype);
    if generics.len() > 0 {
        accm += " where ";
        for generic in generics.iter() {
            accm += &mangle_name(&*generic.val());
            accm += ": Copy,";
        }
    }
    accm
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

/*
fn contains_anyptr(t: &Type) -> bool {
    use Type::*;
    fn list_contains_anyptr<'a>(ts: &mut impl Iterator<Item = &'a Type>) -> bool {
        ts.any(contains_anyptr)
    }
    match t {
        Prim(PrimType::AnyPtr) => true,
        Enum(_body) => todo!(),
        Named(_, body) => list_contains_anyptr(&mut body.iter()),
        Func(_args, rettype, _generics) => {
            trace!("Finding anyptrs in {:?}", &t);
            contains_anyptr(&*rettype)
        }
        Struct(_body, _generics) => todo!(),
        Sum(_body, _generics) => todo!(),
        Array(t, _) => contains_anyptr(t),
        Generic(_) => unreachable!(),
        _ => false,
    }
}
    */

fn compile_expr(expr: &hir::ExprNode, tck: &Tck) -> String {
    use hir::Expr as E;
    let expr_str = match &*expr.e {
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
            typename: _,
            init,
            mutable,
        } => {
            let vstr = mangle_name(&*INT.fetch(*varname));
            // typename may be elided, so we get the real type from the tck
            // TODO: Someday this should just be filled in by a lowering pass
            let type_id = tck.get_expr_type(init);
            let typ = tck.reconstruct(type_id).expect("Passed typechecking but failed to reconstruct during codegen, should never happen!");
            trace!("Type for let statement is {:?}", typ);
            let tstr = compile_typename(&typ);
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
            type_params: _,
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
            //format!("{{ let __dummy = {}; __dummy({}) }}", nstr, pstr)
            format!("{}({})", nstr, pstr)
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
        E::TupleCtor { body } if body.len() == 0 => String::from(" ()\n"),
        E::TupleCtor { body } => {
            let contents = compile_exprs(body, ",", tck);
            format!("({},)", contents)
        }
        // Just becomes a function call.
        // TODO someday: Make an explicit turbofish if necessary?
        E::TypeCtor {
            name: _,
            body,
            type_params: _,
        } => {
            let contents = compile_expr(body, tck);
            //format!("{}({})", &*name.val(), contents)
            format!("{}", contents)
        }
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
        E::ArrayRef { expr, idx } => {
            // TODO: We don't actually have usize types yet, so...
            format!(
                "{}[{} as usize]",
                compile_expr(expr, tck),
                compile_expr(idx, tck)
            )
        }
        E::StructRef { expr, elt } => {
            // panic!("Should never happen, structs should always be tuples by now!");
            format!("{}.{}", compile_expr(expr, tck), elt)
        }
        E::TupleRef { expr, elt } => {
            // We turn our structs into Rust tuples, so we need to
            // to turn our field names into indices
            let typevar = tck.get_expr_type(expr);
            let ty = tck.reconstruct(typevar).unwrap();
            if let Type::Named(_thing, _generics) = ty {
                format!("{}.{}", compile_expr(expr, tck), elt)
            } else {
                panic!(
                    "Tuple ref wasn't actually a tuple in backend, was {:?}.  should never happen",
                    ty
                )
            }
        }
        E::Assign { lhs, rhs } => {
            format!("{} = {}", compile_expr(lhs, tck), compile_expr(rhs, tck))
        }
        E::ArrayCtor { body } => {
            let mut accm = String::from("[");
            for expr in body {
                accm += &compile_expr(expr, tck);
                accm += ", ";
            }
            accm += "]";
            accm
        }
        /*
        E::Deref { expr } => format!("*{}", compile_expr(expr, tck)),
        E::Ref { expr } => format!("&{}", compile_expr(expr, tck)),
        */
        E::TypeUnwrap { expr } => {
            // Since our typedefs compile to Rust type aliases, we don't
            // have to do anything to unwrap them
            compile_expr(expr, tck)
        }
        E::SumCtor {
            name,
            variant,
            body,
        } => {
            // This should just be a function call, right?
            // hahahahha NO this is what has to exist INSIDE the function call.
            format!(
                "{}::{}({})",
                &*name.val(),
                &*variant.val(),
                compile_expr(body, tck)
            )
        }
        E::EnumCtor {
            name: _,
            variant: _,
            value,
        } => {
            format!("{}", value)
        }
        other => todo!("{:?}", other),
    };
    expr_str
}
