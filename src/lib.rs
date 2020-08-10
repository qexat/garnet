//! Garnet compiler guts.
//#![deny(missing_docs)]

use std::borrow::Cow;
use std::rc::Rc;

use codespan_reporting as cs;

pub mod ast;
pub mod backend;
pub mod format;
pub mod intern;
pub mod ir;
pub mod lir;
pub mod parser;
pub mod passes;
mod scope;
pub mod typeck;

/// The interned name of a type
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct TypeSym(pub usize);

impl From<usize> for TypeSym {
    fn from(i: usize) -> TypeSym {
        TypeSym(i)
    }
}

impl From<TypeSym> for usize {
    fn from(i: TypeSym) -> usize {
        i.0
    }
}

/// The interned name of a variable/value
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct VarSym(pub usize);

impl From<usize> for VarSym {
    fn from(i: usize) -> VarSym {
        VarSym(i)
    }
}

impl From<VarSym> for usize {
    fn from(i: VarSym) -> usize {
        i.0
    }
}

/// For now this is what we use as a type...
/// This doesn't include the name, just the properties of it.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeDef {
    /// Signed integer with the given number of bytes
    SInt(usize),
    Bool,
    /// We can infer types for tuples
    Tuple(Vec<TypeSym>),
    Lambda(Vec<TypeSym>, TypeSym),
    Ptr(Box<TypeSym>),
    /*
    /// TODO: AUGJDKSFLJDSFSLAF
    /// This is basically a type that has been named but we
    /// don't know what type it actually is until after type checking...
    Named(String),
    */
}

impl TypeDef {
    /// Get a string for the name of the given type.
    ///
    /// TODO: We need a good way to go the other way around as well,
    /// for built-in types like I32 and Bool.
    pub fn get_name(&self, cx: &Cx) -> Cow<'static, str> {
        let join_types_with_commas = |lst: &[TypeSym]| {
            let p_strs = lst
                .iter()
                .map(|ptype| {
                    let ptype_def = cx.fetch_type(*ptype);
                    ptype_def.get_name(cx)
                })
                .collect::<Vec<_>>();
            p_strs.join(", ")
        };
        match self {
            TypeDef::SInt(4) => Cow::Borrowed("I32"),
            TypeDef::SInt(s) => panic!("Undefined integer size {}!", s),
            TypeDef::Bool => Cow::Borrowed("Bool"),
            TypeDef::Tuple(v) => {
                if v.len() == 0 {
                    Cow::Borrowed("{}")
                } else {
                    let mut res = String::from("{");
                    res += &join_types_with_commas(v);
                    res += "}";
                    Cow::Owned(res)
                }
            }
            TypeDef::Lambda(params, rettype) => {
                let mut t = String::from("fn(");
                t += &join_types_with_commas(params);

                t += ")";
                let ret_def = cx.fetch_type(*rettype);
                match &*ret_def {
                    TypeDef::Tuple(x) if x.len() == 0 => {
                        // pass, implicitly return unit
                    }
                    x => {
                        let type_str = x.get_name(cx);
                        t += ": ";
                        t += &type_str;
                    }
                }
                Cow::Owned(t)
            }
            TypeDef::Ptr(t) => {
                let inner_name = cx.fetch_type(**t).get_name(cx);
                let s = format!("{}^", inner_name);
                Cow::Owned(s)
            }
        }
    }
}

/// Compilation context.
///
/// Really this is just an interner for symbols now, and
/// the original plan of bundling it up into a special context
/// type for each step of compilation hasn't really worked out
/// (at least for wasm with `walrus`.  TODO: Maybe rename it?
pub struct Cx {
    /// Interned symbols
    syms: intern::Interner<VarSym, String>,
    /// Known types
    types: intern::Interner<TypeSym, TypeDef>,
    files: cs::files::SimpleFiles<String, String>,
}

impl Cx {
    pub fn new(mod_name: &str, source: &str) -> Self {
        let s = Cx {
            syms: intern::Interner::new(),
            types: intern::Interner::new(),
            files: cs::files::SimpleFiles::new(mod_name.to_owned(), source.to_owned()),
        };
        s
    }

    /// Intern the symbol.
    pub fn intern(&self, s: impl AsRef<str>) -> VarSym {
        self.syms.intern(&s.as_ref().to_owned())
    }

    /// Get the string for a variable symbol
    pub fn fetch(&self, s: VarSym) -> Rc<String> {
        self.syms.fetch(s)
    }

    /// Intern a type defintion.
    pub fn intern_type(&self, s: &TypeDef) -> TypeSym {
        self.types.intern(&s)
    }

    /// Get the TypeDef for a type symbol
    pub fn fetch_type(&self, s: TypeSym) -> Rc<TypeDef> {
        self.types.fetch(s)
    }

    /// Shortcut for getting the type symbol for I32
    pub fn i32(&self) -> TypeSym {
        self.intern_type(&TypeDef::SInt(4))
    }

    /// Shortcut for getting the type symbol for Bool
    pub fn bool(&self) -> TypeSym {
        self.intern_type(&TypeDef::Bool)
    }

    /// Shortcut for getting the type symbol for Unit
    pub fn unit(&self) -> TypeSym {
        self.intern_type(&TypeDef::Tuple(vec![]))
    }

    /// Generate a new unique symbol including the given string
    /// Useful for some optimziations and intermediate names and such.
    ///
    /// Starts with a `@` which currently cannot appear in any identifier.
    pub fn gensym(&self, s: &str) -> VarSym {
        let sym = format!("@{}_{}", s, self.syms.count());
        self.intern(sym)
    }
}

/// Main driver function.
/// Compile a given source string to wasm.
///
/// Parse -> lower to IR -> run transformation passes
/// -> typecheck -> compile to wasm
pub fn compile(src: &str) -> Vec<u8> {
    let cx = &mut Cx::new();
    let ast = {
        let mut parser = parser::Parser::new(cx, src);
        parser.parse()
    };
    let ir = ir::lower(&ast);
    let ir = passes::run_passes(cx, ir);
    let checked =
        typeck::typecheck(cx, ir).unwrap_or_else(|e| panic!("Type check error: {}", e.format(cx)));
    let wasm = backend::output(backend::Backend::LirWasm32, cx, &checked);
    wasm
}

/// Compiles using the LirWasm32 backend.  Will go away someday.
pub fn compile_lir(src: &str) -> Vec<u8> {
    let cx = &mut Cx::new();
    let ast = {
        let mut parser = parser::Parser::new(cx, src);
        parser.parse()
    };
    let ir = ir::lower(&ast);
    let ir = passes::run_passes(cx, ir);
    let checked =
        typeck::typecheck(cx, ir).unwrap_or_else(|e| panic!("Type check error: {}", e.format(cx)));
    let wasm = backend::output(backend::Backend::LirWasm32, cx, &checked);
    wasm
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Make sure outputting a lambda's name gives us something we expect.
    #[test]
    fn check_name_format() {
        let cx = Cx::new();
        let args = vec![cx.i32(), cx.bool()];
        let def = TypeDef::Lambda(args, cx.i32());
        let gotten_name = def.get_name(&cx);
        let desired_name = "fn(I32, Bool): I32";
        assert_eq!(&gotten_name, desired_name);
    }
}
