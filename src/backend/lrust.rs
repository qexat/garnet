//! Compile our HIR to Rust.
//! This is kinda silly, but hey why not.
//!
//!
//! Potential improvements:
//!  * Use something smarter than strings to collect output -- could just
//!    output to a stream like the formatter does
//!  * Use `syn` or something to generate tokens for output rather than strings

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};

use crate::lir::{self, BB};
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
        Lambda(params, ret) => {
            let mut accm = String::from("fn (");
            for p in params {
                accm += &compile_typedef(cx, &*cx.fetch_type(*p));
                accm += ", ";
            }
            accm += ") -> ";
            accm += &compile_typedef(cx, &*cx.fetch_type(*ret));
            accm.into()
        }
        Ptr(_) => unimplemented!(),
    }
}

pub(super) fn output(cx: &Cx, lir: &lir::Lir) -> Vec<u8> {
    let mut strings = vec![prelude().to_owned()];
    strings.extend(lir.funcs.iter().map(|d| compile_func(cx, d)));
    let s = strings.join("\n");
    s.into_bytes()
}

/// Mangle/unmangle a name for a function.
/// We need this for lambda's, migth need it for other things, we'll see.
/// TODO: There might be a better way to make lambda's un-nameable.
/// Probably, really.
fn mangle_name(s: &str) -> String {
    s.replace("@", "__")
}

/*
/// Control flow graph.  Pretty simple currently.
enum CFGNode {
    IfElse(BB, BB),
    Loop(BB),
    Fallthrough(BB),
    Return,
}

/// Okay so first we need to make a control flow graph
/// and figure out what order we're doing things in, THEN
/// we can compile basic blocks just by walking down the
/// ordered CFG from start to end
fn mongle_cfg(f: &lir::Func) -> Vec<CFGNode> {
    let mut done_bbs = BTreeSet::new();
    let mut current_bb = f.entry;
    let mut res = vec![];
    while !done_bbs.contains(&current_bb) {
        let current_block = f.body.get(&current_bb).expect("Invalid BB, aieeeee");
        match current_block.terminator {
            lir::Branch::Jump(_, b) => {
                if done_bbs.contains(&b) {
                    // This is a loop
                    todo!();
                } else {
                    // Not a loop
                    res.push(CFGNode::Fallthrough(b));
                    done_bbs.insert(current_bb);
                    current_bb = b;
                }
            },
            lir::Branch::Branch(_, b1, b2) => {
                res.push(CFGNode::IfElse(b1, b2));
            }
            lir::Branch::Return(v) => todo!(),
            lir::Branch::Unreachable => todo!("Can't happen???"),
        }

    }

    res
}
*/

fn compile_func(cx: &Cx, func: &lir::Func) -> String {
    let nstr = mangle_name(&*cx.fetch(func.name));
    let sstr = compile_fn_signature(cx, &func.signature);
    //let bstr = compile_exprs(cx, &func.body, ";\n");

    // We start from the first BB and follow its branches.
    let mut out: Vec<String> = vec![];
    // We store all the basic blocks we've done so we don't
    // get caught in loops
    let mut done_bbs = BTreeSet::new();
    let mut bbs_to_visit = vec![func.entry];
    // We do a depth-first walk through all BB's, compiling each of them.
    loop {
        if let Some(current_bb) = bbs_to_visit.pop() {
            let current_block = func.body.get(&current_bb).expect("Invalid BB, aieeeee");
            let res = compile_block(cx, &func, current_bb);
            out.push(res);
            done_bbs.insert(current_bb);

            match current_block.terminator {
                lir::Branch::Jump(v, b) => {
                    if done_bbs.contains(&b) {
                        todo!("Loops can't happen yet I guess!");
                    } else {
                        // Just fall through to the next thing
                        bbs_to_visit.push(b);
                    }
                }
                lir::Branch::Branch(v, b1, b2) => {
                    bbs_to_visit.push(b1);
                    bbs_to_visit.push(b2);
                }
                lir::Branch::Return(v) => todo!(),
                lir::Branch::Unreachable => todo!("Can't happen???"),
            }
        } else {
            // Stack is empty, we are done!
            break;
        }
    }

    //format!("fn {}{} {{\n{}\n}}\n", nstr, sstr, bstr)
    todo!("Output")
}

/// We do a depth-first walk through all BB's, compiling each of them.
/// Recursively, like God intended.
fn heckin_block(
    cx: &Cx,
    func: &lir::Func,
    current_bb: BB,
    done_bbs: &mut BTreeSet<BB>,
    out: &mut Vec<String>,
) {
    let current_block = func.body.get(&current_bb).expect("Invalid BB, aieeeee");
    let res = compile_block(cx, &func, current_bb);
    out.push(res);
    done_bbs.insert(current_bb);

    match current_block.terminator {
        lir::Branch::Jump(v, b) => {
            if done_bbs.contains(&b) {
                todo!("Loops can't happen yet I guess!");
            } else {
                heckin_block(cx, func, b, done_bbs, out)
            }
        }
        lir::Branch::Branch(v, b1, b2) => {
            heckin_block(cx, func, b1, done_bbs, out);
            heckin_block(cx, func, b2, done_bbs, out);
        }
        lir::Branch::Return(v) => todo!(),
        lir::Branch::Unreachable => todo!("Can't happen???"),
    }
}

fn compile_fn_signature(cx: &Cx, sig: &ast::Signature) -> String {
    let mut accm = String::from("(");
    for (varsym, typesym) in sig.params.iter() {
        accm += &*cx.fetch(*varsym);
        accm += ": ";
        accm += &compile_typedef(cx, &*cx.fetch_type(*typesym));
        accm += ", ";
    }
    accm += ") -> ";
    accm += &compile_typedef(cx, &*cx.fetch_type(sig.rettype));
    accm
}

fn compile_block(cx: &Cx, func: &lir::Func, bb: lir::BB) -> String {
    //let ss: Vec<String> = exprs.iter().map(|e| compile_expr(cx, e)).collect();
    //ss.join(separator)
    todo!()
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
fn compile_expr(cx: &Cx, expr: &hir::TypedExpr<TypeSym>) -> String {
    use hir::Expr as E;
    match &expr.e {
        E::Lit {
            val: ast::Literal::Integer(i),
        } => format!("{}", i),
        E::Lit {
            val: ast::Literal::Bool(b),
        } => format!("{}", b),
        E::Var { name } => mangle_name(&*cx.fetch(*name)),
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
            let vstr = mangle_name(&*cx.fetch(*varname));
            let tstr = compile_typedef(cx, &*cx.fetch_type(*typename));
            let istr = compile_expr(cx, init);
            if *mutable {
                format!("let mut {}: {} = {}", vstr, tstr, istr)
            } else {
                format!("let {}: {} = {}", vstr, tstr, istr)
            }
        }
        E::If { cases, falseblock } => {
            let mut accm = String::new();
            for (i, (cond, body)) in cases.iter().enumerate() {
                if i == 0 {
                    accm += "if ";
                } else {
                    accm += " else if "
                }
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
            // We have to store an intermediate value for the func, 'cause
            // Rust has problems with things like this:
            // fn f1() -> i32 { 1 }
            // fn f2() -> i32 { 10 }
            // fn f() -> i32 { if true { f1 } else { f2 }() }
            //
            // Should this happen in an IR lowering step?  idk.
            let nstr = compile_expr(cx, func);
            let pstr = compile_exprs(cx, params, ", ");
            format!("{{ let __dummy = {}; __dummy({}) }}", nstr, pstr)
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
            format!("{} = {}", compile_expr(cx, lhs), compile_expr(cx, rhs))
        }
        E::Deref { expr } => format!("*{}", compile_expr(cx, expr)),
        E::Ref { expr } => format!("&{}", compile_expr(cx, expr)),
    }
}
*/
