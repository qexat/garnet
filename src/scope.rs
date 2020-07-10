//! A simple structure implementing a scoped symbol table.
//! Because I'm sick of doing this over and over.

pub(crate) struct Symbols<K, V> {
    /// We just use a Vec for each scope, and a stack of them,
    /// and search backwards when looking for a var.  We could use
    /// a linked list instead without needing a stack, but heck.
    /// Can't use std's HashMap, since we allow shadowing and
    /// want to conceal old bindings, not annihilate them.
    pub(crate) bindings: Vec<Vec<(K, V)>>,
}

impl<K, V> Default for Symbols<K, V> {
    fn default() -> Self {
        Symbols {
            bindings: vec![vec![]],
        }
    }
}

impl<K, V> Symbols<K, V>
where
    K: Eq + Copy,
    V: 'static,
{
    pub fn push_scope(&mut self) {
        self.bindings.push(vec![]);
    }
    pub fn pop_scope(&mut self) {
        assert!(self.bindings.len() > 1);
        self.bindings.pop();
    }

    /// Add an already-existing local binding to the top of the current scope.
    /// Create a new binding with LocalVar::new()
    pub fn add(&mut self, name: K, val: V) {
        self.bindings.last_mut().unwrap().push((name, val))
    }

    /// Get a reference to an arbitrary binding, if it exists
    pub fn get(&self, name: K) -> Option<&V> {
        for scope in self.bindings.iter().rev() {
            if let Some((_name, v)) = scope.iter().rev().find(|(varname, _)| *varname == name) {
                return Some(v);
            }
        }
        None
    }
}
