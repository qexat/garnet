//! Utilities for unit tests.

use crate::hir;

/// For shortcutting HIR type annotations.
///
/// High on the "I can't believe this is working" scale.
pub fn rly(expr: &hir::Expr<()>) -> () {
    ()
}
