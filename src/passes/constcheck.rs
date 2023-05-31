//! Ok this is the skeleton of a Pass that will mark expressions/decls
//! as const, basically inferring them down the chain.
//! Shoooooould be pretty easy: some terminal expressions such as
//! assignment, loops, and such are non const, and if something contains
//! any non-const expressions it is not const.  We start off by assuming
//! nothing is const and annotating it downward from leaves to roots.
//!
//! We might need to do things like make sure we mark main() and
//! println functions as non-const somehow, we will see.
//!
//! Is this a Pass or a TckPass?  We need to be able to do name lookups
//! to see whether a function we are calling is const or not... hmmm.
//! We really do need a symbol table somewhere, don't we.
//! We'll make this a TckPass so we don't have to change it later
//! once we figure that shit out.

use crate::passes::*;

/// Hmmm.  expr_map() does a pre-traversal; this needs to be
/// a post-traversal.  ie, we need to go from the tip of the
/// leaves down up to the root.
fn constcheck_expr(expr: ExprNode) -> ExprNode {
    let is_const = match &*expr.e {
        E::Lit { .. } => true,
        E::Var { .. } => true,                   // I think?
        E::Break => false,                       // probably?
        E::Return { retval } => retval.is_const, // maybe?
        E::EnumCtor { .. } => true,
        E::TupleCtor { body } => body.iter().all(|e| e.is_const),
        E::StructCtor { body } => body.iter().all(|(_k, v)| v.is_const),
        E::ArrayCtor { body } => body.iter().all(|e| e.is_const),
        E::SumCtor { body, .. } => body.is_const,
        E::TypeCtor { body, .. } => body.is_const,
        E::TypeUnwrap { expr, .. } => expr.is_const,
        E::TupleRef { expr, .. } => expr.is_const,
        E::StructRef { expr, .. } => expr.is_const,
        E::ArrayRef { expr, idx } => expr.is_const && idx.is_const,
        E::Assign { .. } => false, // For now
        E::Typecast { .. } => unreachable!(),
        // FOR NOW, all our binops are const.
        // Eventually this may not be true for floating point ops.
        E::BinOp { .. } => true,
        E::UniOp {
            op: hir::UOp::Neg,
            rhs,
        } => rhs.is_const,
        E::UniOp {
            op: hir::UOp::Not,
            rhs,
        } => rhs.is_const,
        E::UniOp {
            op: hir::UOp::Ref, ..
        } => false,
        E::UniOp {
            op: hir::UOp::Deref,
            ..
        } => false,
        E::Block { body } => body.iter().all(|e| e.is_const),
        E::Let { init, mutable, .. } => !mutable && init.is_const,
        E::If { .. } => false, // hmmm
        E::Loop { .. } => false,
        E::Lambda { .. } => true,
        E::Funcall { .. } => false, // TODO: True if function being called is also const
    };
    let mut expr = expr;
    expr.is_const = is_const;
    expr
}

pub fn constcheck(ir: Ir, _tck: &mut typeck::Tck) -> Ir {
    let type_map = &mut |t| t;
    let new_decls = ir
        .decls
        .into_iter()
        .map(|d| decl_map_pre(d, &mut constcheck_expr, type_map))
        .collect();
    Ir {
        decls: new_decls,
        ..ir
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hir::*;

    // hum bum grum something is wrong, let's find it
    #[test]
    fn constcheck_var() {
        let inp = ExprNode::new(Expr::Var {
            name: Sym::new("foo"),
        });
        assert!(!inp.is_const);
        let outp = constcheck_expr(inp);
        assert!(outp.is_const);
    }

    #[test]
    fn constcheck_nested_var() {
        let inp = ExprNode::new(Expr::Block {
            body: vec![ExprNode::new(Expr::Var {
                name: Sym::new("foo"),
            })],
        });
        assert!(!inp.is_const);
        let outp = constcheck_expr(inp);
        assert!(outp.is_const);
    }
}
