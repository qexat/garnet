//! Simple generic interner
//!
//! Does not free its contents; you should just free the whole thing when you're done with it.

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, RwLock};

/// Safe interner.
/// The mutable caches and other guts are RefCell'd, so this is logically "immutable".
#[derive(Debug)]
pub struct Interner<Ky, Val>
where
    Val: Eq + Hash,
    Ky: From<usize> + Into<usize> + Copy + Clone,
{
    /// Map from index/key to value
    /// We just Arc the value type, which is fine.
    /// This saves us the trouble of returning copies of it anyway.
    data: RwLock<Vec<Arc<Val>>>,
    /// Map back from value to key
    map: RwLock<HashMap<Arc<Val>, Ky>>,
}

impl<Ky, Val> Interner<Ky, Val>
where
    Val: Clone + Sized + Eq + Hash + 'static,
    Ky: From<usize> + Into<usize> + Copy + Clone,
{
    pub fn new() -> Self {
        Self {
            data: RwLock::new(vec![]),
            map: RwLock::new(HashMap::new()),
        }
    }
    /// Intern a value if necessary, returning a key/handle uniquely identifying it.
    pub fn intern(&self, s: &Val) -> Ky {
        // Apparently I'm not smart enough to use entry() currently.
        let mut data = self.data.write().unwrap();
        let mut map = self.map.write().unwrap();
        if let Some(sym) = map.get(&*s) {
            // We have it, great
            *sym
        } else {
            // We create the key from the length of the data.
            // This works because we never remove elements.
            let sym = Ky::from(data.len());
            let s = Arc::new(s.clone());
            data.push(s.clone());
            map.insert(s, sym);
            sym
        }
    }

    /// Get a value given its token
    /// Clones the value, which is kinda ugly, but otherwise it runs
    /// into borrowing issues with its RefCell.
    pub fn fetch(&self, sym: Ky) -> Arc<Val> {
        self.data.read().unwrap()[sym.into()].clone()
    }

    /// Number of symbols held.
    pub fn count(&self) -> usize {
        self.data.read().unwrap().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// This found a bug.
    #[test]
    fn test_basic() {
        let i: Interner<usize, String> = Interner::new();
        let goodval = "foo";
        let badval = "bar";
        let s1 = i.intern(&goodval.to_owned());
        let s2 = i.intern(&goodval.to_owned());
        let sbad = i.intern(&badval.to_owned());

        assert_eq!(s1, s2);
        assert_ne!(s1, sbad);

        let check = i.fetch(s1);
        assert_eq!(&*check, goodval);
        assert_ne!(&*check, badval);
    }
}
