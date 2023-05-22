//! Import resolution pass.
//! Ok this should be pretty easy.  ...I just cursed it, didn't I.  Fak.
//!
//! So when we load a file, we can through its toplevel and make a list of all the functions, types, etc
//! in it.

use std::collections::BTreeSet;
use std::path::PathBuf as FilePath;

use crate::hir::*;
use crate::passes::*;
use crate::*;

/// A fully qualified identifier path.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Path(Vec<Sym>);

impl From<Sym> for Path {
    fn from(other: Sym) -> Self {
        Path(vec![other])
    }
}

/// Useful for testing
impl From<&str> for Path {
    fn from(other: &str) -> Self {
        Path(vec![Sym::new(other)])
    }
}

/// Useful for testing
impl From<&[&str]> for Path {
    fn from(other: &[&str]) -> Self {
        let v = other.iter()
            .map(|s| Sym::new(s))
            .collect();
        Path(v)
    }
}


/// All the symbols and such in a particular file
#[derive(Debug, Clone)]
struct Package {
    /// The fully-qualified package name
    name: Path,
    /// The filename containing the package
    file: FilePath,
    /// The functions the package contains
    functions: BTreeMap<Sym, Signature>,
    /// Constants the package contains.  Totally not the same as functions, sigh
    consts: BTreeMap<Sym, Type>,
    /// Types the package contains.
    types: BTreeMap<Sym, Type>,
    /// Imported packages, and local name (if any)
    imports: BTreeMap<Path, Sym>,
}

impl Package {
    fn new(name: Path, file: FilePath) -> Self {
        Package {
            name,
            file,
            functions: Default::default(),
            consts: Default::default(),
            types: Default::default(),
            imports: Default::default(),
        }
    }
}

/*
struct GlobalSymtbl {
    packages: BTreeMap<Path, Package>,
}

fn import_package(g: &mut GlobalSymtbl, f: FilePath) {}
*/

/// Takes an `Ir` and scans through it finding what all its exported values/types
/// are.  Does not recurse.
///
/// TODO REFACTOR: This feels a lot like the typeck predeclare function...
fn construct_package(ir: &Ir, path: Path, file: FilePath) -> Package {
    let mut pk = Package::new(path, file.clone());
    for decl in &ir.decls {
        let already_existed = match decl {
            Decl::Function {
                name, signature, ..
            } => pk.functions.insert(*name, signature.clone()).is_some(),
            Decl::Const { name, typ, .. } => pk.consts.insert(*name, typ.clone()).is_some(),
            Decl::TypeDef { name, typedecl, .. } => {
                pk.types.insert(*name, typedecl.clone()).is_some()
            }
            Decl::Import { name, localname } => {
                pk.imports.insert(Path::from(*name), *localname).is_some()
            }
        };
        if already_existed {
            panic!("Duplicate decl in package {:?}", file)
        }
    }
    pk
}

/// There might be more interner- or allocator-friendly ways 
// to handle this, but eh
fn path_to_literal_name(package: &Path, sym: Sym) -> Sym {
    let mut accm = String::new();
    for symbol in path.0 {
        // Path names always start with a dot, I suppose.
        // Handy!
        accm += ".";
        accm += &*symbol.val();
    }
    accm += &*sym.val();
    Sym::new(&accm)
}

/// For now, all imports are absolute, we have no relative
/// imports.
fn path_to_filename(path: &Path) -> FilePath {
    let mut accm = String::from(".");
    for sym in &path.0 {
        accm += "/";
        accm += &*sym.val();
    }
    accm += ".gt";
    FilePath::from(&accm)
}

pub fn handle_imports(ir: Ir) -> Ir {
    // Our "new things to handle" and "things we've already handled"
    // sets, so we don't do the same import twice.
    let mut new_imports: BTreeSet<Path> = Default::default();
    let mut handled_imports: BTreeSet<Path> = Default::default();
    // Construct toplevel package
    // TODO: We need the package name and path to be part of the Ir for this
    let path = Path::from(Sym::new("TODO"));
    let file = FilePath::from("TODO");
    let mut current_package = construct_package(&ir, path.clone(), file);
    let mut packages = BTreeMap::new();

    // This loop is a little weird 'cause we start with an existing
    // package, so instead of "grab new package, process it" we do
    // "process package, grab new package".  Works though.
    // You could juggle it around if you really wanted but eh.
    // To do that you start needing to view this process as a driver
    // for the whole compiler, which it is, but I don't feel like
    // rearchitecting it right now.
    loop {
        // We just find all the imports for the current package
        // and add them to the set of packages to work on
        for (import, _localname) in &current_package.imports {
            if !handled_imports.contains(&import) {
                new_imports.insert(import.clone());
            }
        }
        handled_imports.insert(current_package.name.clone());
        packages.insert(current_package.name.clone(), current_package.clone());
        // Grab another package to process, more or less
        // at random.
        if let Some(path) = new_imports.pop_first() {
            // Find the appropriate file for that path
            let file = path_to_filename(&path);
            let file_str = file.to_str().expect("module path turned into invalid filename string, aieeee");
            let src = std::fs::read_to_string(&file).unwrap();
            // Parse and lower it to HIR
            let ir = crate::load_to_hir(&file_str, &src);
            // Create a Package from it
            current_package = construct_package(&ir, path, file);
            // Then we just loop!
        } else {
            // We are out of packages to scan, we are done!
            break;
        }
    }
    dbg!(packages);

    // TODO: Ok, now that we have all our packages loaded, we need
    // to go through them and transform all local names into fully-
    // qualified names.
    // We don't even really need to turn them into Path's, we just
    // rename them into literal strings.
    ir
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_path_to_filename() {
        let paths = vec![
            (Path::from("foo"), FilePath::from("./foo.gt")),
            (Path::from(&["foo", "bar", "baz"][..]), FilePath::from("./foo/bar/baz.gt")),
        ];
        for (p, fp) in paths.iter() {
            assert_eq!(&path_to_filename(p), fp);
        }
    }
}