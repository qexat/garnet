//! Garnet compiler guts.
//#![deny(missing_docs)]

use std::collections::HashMap;

pub mod ast;
pub mod intern;
pub mod ir;
pub mod typeck;

/// For now this is what we use as a type...
/// This doesn't include the name, just the properties of it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeDef {
    /// Signed integer with the given number of bytes
    SInt(usize),
    Bool,
    Tuple(Vec<TypeDef>),
}

/// Compilation context.  Contains things like symbol tables.
/// Probably keeps most stages of compilation around to do
/// things like output error messages or whatever?
pub struct Cx {
    /// Interned symbols
    syms: intern::Interner,
    /// Known types
    types: HashMap<ir::TypeSym, TypeDef>,
}

impl Cx {
    pub fn new() -> Self {
        let mut s = Cx {
            syms: intern::Interner::default(),
            types: HashMap::new(),
        };
        s.populate_builtin_types();
        s
    }

    /// Shortcut for getting an interned symbol.
    pub fn intern(&mut self, s: &str) -> intern::Sym {
        self.syms.intern(s)
    }

    pub fn unintern(&self, s: intern::Sym) -> &str {
        self.syms.get(s)
    }

    /// Fill type table with whatever builtin types we have.
    fn populate_builtin_types(&mut self) {
        let types: HashMap<ir::TypeSym, TypeDef> = [
            (self.intern("i32"), TypeDef::SInt(4)),
            (self.intern("bool"), TypeDef::Bool),
            // TODO: Not sure I like naming tuples like this.
            (self.intern("()"), TypeDef::Tuple(vec![])),
        ]
        .iter()
        .cloned()
        .map(|(n, t)| (ir::TypeSym(n), t))
        .collect();
        self.types = types;
    }

    /// Returns the symbol naming the given type, or none if it's
    /// not defined.
    pub fn get_typename(&mut self, name: &str) -> Option<ir::TypeSym> {
        let s = ir::TypeSym(self.syms.intern(name));
        if self.types.contains_key(&s) {
            Some(s)
        } else {
            None
        }
    }
}
