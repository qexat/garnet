//! Simple string interner
//! Could use `https://github.com/Robbepop/string-interner` instead, but, eh.
//!
//! Does not free its strings; free the whole thing.

use std::collections::HashMap;
use std::hash::Hash;

/// Safe string interner.
#[derive(Debug)]
pub struct Interner<Ky, Val>
where
    Val: Eq + Hash,
    Ky: From<usize> + Into<usize> + Copy + Clone,
{
    /// We store two copies of the string, because I don't want to bother with unsafe,
    /// so.  It's fine for now.
    data: Vec<Val>,
    map: HashMap<Val, Ky>,
}

/// The type for an interned thingy.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Hash, Debug)]
pub struct Sym(pub usize);

impl<Ky, Val> Interner<Ky, Val>
where
    Val: Clone + Sized + Eq + Hash + 'static,
    Ky: From<usize> + Into<usize> + Copy + Clone,
{
    pub fn new() -> Self {
        Self {
            data: vec![],
            map: HashMap::new(),
        }
    }
    /// Intern the string, if necessary, returning a token for it.
    pub fn intern(&mut self, s: &Val) -> Ky {
        // Apparently I'm not smart enough to use entry() currently.
        if let Some(sym) = self.map.get(&s) {
            // We have it, great
            *sym
        } else {
            let sym = Ky::from(self.data.len());
            self.data.push(s.clone());
            self.map.insert(s.clone(), sym);
            sym
        }
    }

    /// Get a string given its token
    pub fn get(&self, sym: Ky) -> &Val {
        &self.data[sym.into()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// This found a bug.
    #[test]
    fn test_basic() {
        let mut i: Interner<usize, String> = Interner::new();
        let goodval = "foo";
        let badval = "bar";
        let s1 = i.intern(&goodval.to_owned());
        let s2 = i.intern(&goodval.to_owned());
        let sbad = i.intern(&badval.to_owned());

        assert_eq!(s1, s2);
        assert_ne!(s1, sbad);

        let check = i.get(s1);
        assert_eq!(check, goodval);
        assert_ne!(check, badval);
    }
}
