//! Garnet compiler guts.
//#![deny(missing_docs)]

pub mod ast;
pub mod intern;
pub mod ir;

/// Compilation context.  Contains things like symbol tables.
/// Probably keeps most stages of compilation around to do
/// things like output error messages or whatever?
pub struct Cx {
    /// Interned symbols
    syms: intern::Interner,
}

impl Cx {
    pub fn new() -> Self {
        Cx {
            syms: intern::Interner::default(),
        }
    }

    pub fn intern(&mut self, s: &str) -> intern::Sym {
        self.syms.intern(s)
    }
}
