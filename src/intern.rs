//! Simple string interner
//! Could use `https://github.com/Robbepop/string-interner` instead, but, eh.
//!
//! Does not free its strings; free the whole thing.

use std::borrow::Borrow;
use std::collections::HashMap;
use std::hash::Hash;

/// Safe string interner.
#[derive(Debug, Default)]
pub struct Interner<Val>
where
    Val: Eq + Hash,
{
    /// We store two copies of the string, because I don't want to bother with unsafe,
    /// so.  It's fine for now.
    data: Vec<Val>,
    map: HashMap<Val, Sym>,
}

/// The type for an interned thingy.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Hash, Debug)]
pub struct Sym(usize);

impl<Val> Interner<Val>
where
    Val: Clone + Sized + Eq + Hash + 'static,
{
    /// Intern the string, if necessary, returning a token for it.
    pub fn intern(&mut self, s: &Val) -> Sym {
        // Apparently I'm not smart enough to use entry() currently.
        if let Some(sym) = self.map.get(&s) {
            // We have it, great
            *sym
        } else {
            let sym = Sym(self.data.len());
            self.data.push(s.clone());
            self.map.insert(s.clone(), sym);
            sym
        }
    }

    /// Get a string given its token
    pub fn get(&self, sym: Sym) -> &Val {
        &self.data[sym.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// This found a bug.
    #[test]
    fn test_basic() {
        let mut i: Interner<String> = Interner::default();
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
