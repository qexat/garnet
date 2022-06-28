//! Compile our HIR to QBE.  Or attempt to at least.
//! Just to see how it goes.

use std::borrow::Cow;

use qbe as q;

use crate::hir;
use crate::typeck::Tck;
use crate::*;

/// Whatever prefix stuff we want in the program
/// We just use C printf for it.
///
/// TODO: Narrow i16's properly???  Need to reread that bit
fn prelude() -> &'static str {
    r#"
data $__fmt16 = { b "%d\n", b 0 }
function $__println_i16(w %i) {
@start
    call $printf(l $__fmt16, ..., w %i)
    ret
}

data $__fmt32 = { b "%d\n", b 0 }
function $__println_i32(w %i) {
@start
    call $printf(l $__fmt32, ..., w %i)
    ret
}

data $__fmt64 = { b "%ld\n", b 0 }
function $__println_i64(l %i) {
@start
    call $printf(l $__fmt64, ..., l %i)
    ret
}

# We don't have a printf specifier for bool,
# of course.
# So we just heckin' do it.
data $__fmt_true  = { b "true\n", b 0 }
data $__fmt_false = { b "false\n", b 0 }
function $__println_bool(w %i) {
@start
    jnz %i, @printtrue, @printfalse
@printtrue
    call $puts(l $__fmt_true)
    jmp @end
@printfalse
    call $puts(l $__fmt_false)
    jmp @end
@end
    ret
}

"#
}

fn compile_typedef(td: &TypeDef) -> q::Type {
    use crate::TypeDef::*;
    match td {
        SInt(16) => panic!("Who needs i128 anyway, amirite?"),
        SInt(8) => q::Type::Long,
        SInt(4) => q::Type::Word,
        // Halfword and Byte types can't be in function args.
        // For now we just compile everything to words.
        // Which will probably not quite work right when we
        // get to compiling structs, but hey.
        SInt(2) => q::Type::Word,
        SInt(1) => q::Type::Word,
        SInt(e) => {
            unreachable!("Invalid integer size: {}", e)
        }
        UnknownInt => unreachable!("Backend got an integer of unknown size, should never happen!"),
        Bool => q::Type::Word,
        Never => panic!("Stable rust can't quite do this yet..."),
        Tuple(types) => {
            q::Type::Aggregate("TODO".into())
            //todo!("Tuple")
            /*
            let mut accm = String::from("(");
            for typ in types {
                accm += &compile_typename(&*INT.fetch_type(*typ));
                accm += ", ";
            }
            accm += ")";
            accm.into()
                */
        }
        Lambda {
            generics,
            params,
            rettype,
        } => {
            todo!("Lambda")
            /*
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
                accm += &compile_typename(&*INT.fetch_type(*p));
                accm += ", ";
            }
            accm += ") -> ";
            accm += &compile_typename(&*INT.fetch_type(*rettype));
            accm.into()
                */
        }
        //Named(sym) => (&*INT.fetch(*sym)).clone().into(),
        Struct { fields } => {
            todo!("Struct")
            // We compile our structs into Rust tuples.
            // Our fields are always ordered via BTreeMap etc,
            // so it's okay
            /*
            if typefields.len() > 0 {
                todo!("Figure out type thingies");
            }
            */
            /*
            let mut accm = String::from("(");
            for (nm, typ) in fields.iter() {
                accm += &format!(
                    "/* {} */
            {}, \n",
                    INT.fetch(*nm),
                    compile_typename(&*INT.fetch_type(*typ))
                );
            }
            accm += ")";
            accm.into()
                */
        }
        Enum { variants: _ } => {
            todo!("Enum")
        }
        NamedType(_vsym) => todo!("Output typevar"),
    }
}

/// Handy shortcut
fn compile_typesym(ts: TypeSym) -> q::Type {
    compile_typedef(&*INT.fetch_type(ts))
}

pub(super) fn output(lir: &hir::Ir, tck: &Tck) -> Vec<u8> {
    let mut output = vec![];
    output.extend(prelude().as_bytes());
    let mut module = q::Module::new();
    for decl in lir.decls.iter() {
        compile_decl(&mut module, decl, tck);
    }
    let qbe_text = format!("{}", module);
    output.extend(qbe_text.as_bytes());
    output
}

/// Mangle/unmangle a name for a function.
/// We need this for lambda's, migth need it for other things, we'll see.
/// TODO: There might be a better way to make lambda's un-nameable.
/// Probably, really.
fn mangle_name(s: &str) -> String {
    s.replace("@", "__")
}

/// Gensym a unique name appropriate for a temporary or such
fn tmp_name(s: &str) -> String {
    let name = &*INT.gensym("lit").val();
    mangle_name(name)
}

fn compile_decl(module: &mut q::Module, decl: &hir::Decl, tck: &Tck) {
    match decl {
        hir::Decl::Function {
            name,
            signature,
            body,
        } => {
            let linkage = q::Linkage::public();
            let nstr = mangle_name(&*INT.fetch(*name));
            let (args, rettype) = compile_fn_signature(signature);
            let mut f = q::Function::new(linkage, nstr, args, rettype);

            compile_exprs(&mut f, body, tck);

            module.add_function(f);
        }
        /*
        hir::Decl::Const {
            name,
            typename,
            init,
        } => {
            let nstr = mangle_name(&INT.fetch(*name));
            let tstr = compile_typedef(&*INT.fetch_type(*typename));
            let istr = compile_expr(init, tck);
            writeln!(w, "const {}: {} = {};", nstr, tstr, istr)
        }
        // Typedefs compile into newtype structs.
        hir::Decl::TypeDef { name, typedecl } => {
            let nstr = mangle_name(&INT.fetch(*name));
            let tstr = compile_typedef(&*INT.fetch_type(*typedecl));
            writeln!(w, "pub struct {}({});", nstr, tstr)
        }

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
                generics: vec![],
                params: vec![(INT.intern("input"), signature.rettype)],
                rettype: signature.params[0].1,
            };
            let sig_str = compile_fn_signature(&destructure_signature);
            writeln!(w, "fn {}_unwrap{} {{ input.0 }}\n", nstr, sig_str,)
        }
        */
        _ => todo!(),
    }
}

/// Returns (args, rettype)
fn compile_fn_signature(sig: &ast::Signature) -> (Vec<(q::Type, q::Value)>, Option<q::Type>) {
    if sig.generics.len() > 0 {
        todo!("Instantiate generics");
    }
    let mut args = vec![];
    for (varsym, typesym) in sig.params.iter() {
        let typedef = &*INT.fetch_type(*typesym);
        let name = &*INT.fetch(*varsym);
        let t = compile_typedef(typedef);
        let n = q::Value::Temporary(name.into());
        args.push((t, n));
    }
    let rettype = if sig.rettype == INT.unit() {
        None
    } else {
        Some(compile_typedef(&*INT.fetch_type(sig.rettype)))
    };
    (args, rettype)
}

fn compile_exprs(function: &mut q::Function, exprs: &[hir::TypedExpr], tck: &Tck) {
    /*
    let mut current_block = q::Block {
        label: "start".into(),
        statements: vec![],
    };
    */
    function.add_block("start".into());
    for expr in exprs {
        use hir::Expr as E;
        // Get the checked type for the expression
        let typevar = tck.get_typevar_for_expression(expr).unwrap();
        let typesym = crate::typeck::try_solve_type(tck, typevar).unwrap();
        let ty = compile_typesym(typesym);
        // Compile the expression
        match &expr.e {
            E::Lit {
                val: ast::Literal::Integer(i),
            } => {
                let name = tmp_name("lit");
                let val = q::Value::Temporary(name);
                let instr = q::Instr::Copy(q::Value::Const(*i as u64));

                let stm = q::Statement::Assign(val, ty, instr);
                let current_block = function.blocks.last_mut().unwrap();
                current_block.statements.push(stm);
            }
            E::Let { varname, init, .. } => {
                // TODO:
                // OK so we need to be able to compile some
                // expressions and know what the name of the
                // last value is
                let init_exprs = vec![(**init).clone()];
                let _ = compile_exprs(function, &init_exprs, tck);

                let name = tmp_name(&format!("var_{}", &*varname.val()));
                let val = q::Value::Temporary(name);
                let instr = q::Instr::Copy(q::Value::Temporary("foo".into()));

                let stm = q::Statement::Assign(val, ty, instr);
                let current_block = function.blocks.last_mut().unwrap();
                current_block.statements.push(stm);
            }
            other => todo!("Compile {:?}", other), /*
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
                                                                     let tstr = compile_typename(&*INT.fetch_type(*typename));
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
                                                                 E::Funcall { func, params } => {
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
                                                                         accm += &format!("/* {} */
                                                                 {
                                                   */
        }
    }
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
