use crate::hir;
use crate::typeck;

pub fn borrowck(_ir: &hir::Ir, _tck: &typeck::Tck) -> Result<(), String> {
    // Everything's fine, trust me.
    Ok(())
}
