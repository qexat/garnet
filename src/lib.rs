//! Garnet compiler driver functions and utility funcs.

//#![deny(missing_docs)]

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use log::*;
use once_cell::sync::Lazy;

mod ast;
pub mod backend;
pub mod borrowck;
mod builtins;
pub mod format;
pub mod hir;
mod intern;
pub mod parser;
pub mod passes;
pub mod symtbl;
pub mod typeck;
pub mod types;

/// The interner.  It's the ONLY part we have to actually
/// carry around anywhere, so I'm experimenting with making
/// it a global.  Seems to work pretty okay.
static INT: Lazy<Cx> = Lazy::new(Cx::new);

/// An interned string of some kind, any kind.
#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Sym(pub usize);

/// Required for interner interface.
impl From<usize> for Sym {
    fn from(i: usize) -> Sym {
        Sym(i)
    }
}

/// Required for interner interface.
impl From<Sym> for usize {
    fn from(i: Sym) -> usize {
        i.0
    }
}

impl Sym {
    /// Intern a new string.
    pub fn new(s: impl AsRef<str>) -> Self {
        INT.intern(s)
    }

    /// Get the underlying string
    pub fn val(&self) -> Arc<String> {
        INT.fetch(*self)
    }
}

impl fmt::Debug for Sym {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Sym({}, {:?})", self.0, self.val())
    }
}

impl fmt::Display for Sym {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.val())
    }
}

/// Global interner context for Sym's.
///
/// Could be used to store other things someday if we need to.
#[derive(Debug)]
struct Cx {
    /// Interned var names
    syms: intern::Interner<Sym, String>,
    //files: cs::files::SimpleFiles<String, String>,
}

impl Cx {
    pub fn new() -> Self {
        Cx {
            syms: intern::Interner::new(),
            //files: cs::files::SimpleFiles::new(mod_name.to_owned(), source.to_owned()),
        }
    }

    /// Intern the symbol.
    fn intern(&self, s: impl AsRef<str>) -> Sym {
        self.syms.intern(&s.as_ref().to_owned())
    }

    /// Get the string for a variable symbol
    fn fetch(&self, s: Sym) -> Arc<String> {
        self.syms.fetch(s)
    }

    /// Generate a new unique symbol including the given string
    /// Useful for some optimziations and intermediate names and such.
    ///
    /// Starts with a `!` which currently cannot appear in any identifier.
    pub fn gensym(&self, s: &str) -> Sym {
        let sym = format!("!{}_{}", s, self.syms.count());
        self.intern(sym)
    }
}

/// Main driver function.
/// Compile a given source string to Rust source code, or return an error.
/// TODO: Better errors with locations
///
/// Parse -> lower to IR -> run transformation passes
/// -> typecheck -> more passes -> codegen
pub fn try_compile(
    filename: &str,
    src: &str,
    backend: backend::Backend,
) -> Result<Vec<u8>, typeck::TypeError> {
    let ast = {
        let mut parser = parser::Parser::new(filename, src);
        parser.parse()
    };
    let hir = hir::lower(&ast);
    info!("HIR from AST lowering:\n{}", &hir);
    // let (hir, _symtbl) = symtbl::make_symbols_unique(hir);
    // info!("HIR from symtbl renaming 1:\n{}", &hir);
    let hir = passes::run_passes(hir);
    info!("HIR from first passes:\n{}", &hir);
    // Symbol resolution has to happen (again???) after passes 'cause
    // the passes may generate new code.
    let (hir, mut symtbl) = symtbl::make_symbols_unique(hir);
    info!("HIR from symtbl renaming:\n{}", &hir);
    // info!("Symtbl from AST:\n{:#?}", &symtbl);
    let tck = &mut typeck::typecheck(&hir, &mut symtbl)?;
    borrowck::borrowck(&hir, tck).unwrap();
    let hir = passes::run_typechecked_passes(hir, &mut symtbl, tck);
    info!("HIR after transform passes:\n{}", &hir);
    Ok(backend::output(backend, &hir, tck))
}

/// For when we don't care about catching results
pub fn compile(filename: &str, src: &str, backend: backend::Backend) -> Vec<u8> {
    try_compile(filename, src, backend).unwrap_or_else(|e| panic!("Type check error: {}", e))
}

/// Turns source code into HIR, panicking on any error.
/// Useful for unit tests.
#[cfg(test)]
fn _compile_to_hir_expr(src: &str) -> hir::ExprNode {
    let ast = {
        let mut parser = parser::Parser::new("__None__", src);
        let res = parser.parse_expr(0);
        res.expect("input to compile_to_hir_expr had a syntax error!")
    };
    hir::lower_expr(&ast)
}

#[cfg(test)]
fn _compile_to_hir_exprs(src: &str) -> Vec<hir::ExprNode> {
    let ast = {
        let mut parser = parser::Parser::new("__None__", src);
        parser.parse_exprs()
    };
    hir::lower_exprs(&ast)
}

#[cfg(test)]
mod tests {
    use crate::types::*;
    /// Make sure outputting a lambda's name gives us something we expect.
    #[test]
    fn check_name_format() {
        let args = vec![Type::i32(), Type::bool()];
        let def = Type::function(&args, &Type::i32(), &vec![]);
        let gotten_name = def.get_name();
        let desired_name = "fn(I32, Bool) I32";
        assert_eq!(&gotten_name, desired_name);
    }

    /// Make sure that `Type::get_name()` and `TypeDef::get_primitive_type()`
    /// are inverses for all primitive types.
    ///
    /// TODO: This currently requires that we have a list of all primitive types here,
    /// which is somewhat annoying.
    #[test]
    fn check_primitive_names() {
        let prims = vec![
            Type::i8(),
            Type::i16(),
            Type::i32(),
            Type::i64(),
            Type::i128(),
            Type::never(),
            Type::bool(),
        ];
        for p in &prims {
            assert_eq!(p, &Type::get_primitive_type(&p.get_name()).unwrap());
        }
    }
}
