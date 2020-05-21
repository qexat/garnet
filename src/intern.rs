//! Simple string interner
//! Could use `https://github.com/Robbepop/string-interner` instead, but, eh.
//!
//! Does not free its strings; free the whole thing.

use std::collections::HashMap;

/// Safe string interner.
#[derive(Debug, Default)]
pub struct Interner {
    /// We store two copies of the string, because I don't want to bother with unsafe,
    /// so.  It's fine for now.
    data: Vec<String>,
    map: HashMap<String, Sym>,
}

/// The type for an interned string.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Hash, Debug)]
pub struct Sym(usize);

impl Interner {
    /// Intern the string, if necessary, returning a token for it.
    pub fn intern(&mut self, s: impl AsRef<str>) -> Sym {
        // Apparently I'm not smart enough to use entry() currently.
        if let Some(sym) = self.map.get(s.as_ref()) {
            // We have it, great
            *sym
        } else {
            let sym = Sym(self.data.len());
            self.data.push(s.as_ref().to_owned());
            self.map.insert(s.as_ref().to_owned(), sym);
            sym
        }
    }

    /// Get a string given its token
    pub fn get(&self, sym: Sym) -> &str {
        &self.data[sym.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// This found a bug.
    #[test]
    fn test_basic() {
        let mut i = Interner::default();
        let goodval = "foo";
        let badval = "bar";
        let s1 = i.intern(goodval);
        let s2 = i.intern(goodval);
        let sbad = i.intern(badval);

        assert_eq!(s1, s2);
        assert_ne!(s1, sbad);

        let check = i.get(s1);
        assert_eq!(check, goodval);
        assert_ne!(check, badval);
    }
}
