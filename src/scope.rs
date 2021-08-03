//! A simple structure implementing a scoped symbol table.
//! Because I'm sick of doing this over and over.

use std::cell::RefCell;
use std::rc::Rc;

pub(crate) struct Symbols<K, V> {
    /// We just use a Vec for each scope, and a stack of them,
    /// and search backwards when looking for a var.  We could use
    /// a linked list instead without needing a stack, but heck.
    /// Can't use std's HashMap, since we allow shadowing and
    /// want to conceal old bindings, not annihilate them.
    pub(crate) bindings: Rc<RefCell<Vec<Vec<(K, V)>>>>,
}

impl<K, V> Default for Symbols<K, V> {
    fn default() -> Self {
        Symbols {
            bindings: Rc::new(RefCell::new(vec![vec![]])),
        }
    }
}

/// Pops the scope when it is dropped.
pub struct ScopeGuard<K, V> {
    bindings: Rc<RefCell<Vec<Vec<(K, V)>>>>,
}

impl<K, V> Drop for ScopeGuard<K, V> {
    fn drop(&mut self) {
        assert!(self.bindings.borrow().len() > 1);
        self.bindings.borrow_mut().pop();
    }
}

impl<K, V> Symbols<K, V>
where
    K: Eq + Copy,
    V: 'static + Clone,
{
    pub fn push_scope(&mut self) -> ScopeGuard<K, V> {
        self.bindings.borrow_mut().push(vec![]);
        ScopeGuard {
            bindings: self.bindings.clone(),
        }
    }

    /// Add an already-existing local binding to the top of the current scope.
    /// Create a new binding with LocalVar::new()
    pub fn add(&mut self, name: K, val: V) {
        self.bindings
            .borrow_mut()
            .last_mut()
            .unwrap()
            .push((name, val))
    }

    /// Get a reference to an arbitrary binding, if it exists
    ///
    /// TODO: ...except this can't be a reference because we refcell things
    /// so that we can make the drop guard.
    pub fn get(&self, name: K) -> Option<V> {
        let b = self.bindings.borrow();
        for scope in b.iter().rev() {
            if let Some((_name, v)) = scope.iter().rev().find(|(varname, _)| *varname == name) {
                return Some(v.clone());
            }
        }
        None
    }
}
