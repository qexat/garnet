//! Symbol table.
//!
//! Basically we need to make the symtable persistent,
//! so we only need to figure out all meta-info about it
//! once instead of needing to walk through scopes multiple
//! times and then throw the scope information away when
//! we're done with it.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use crate::*;
// TODO: Dependency inversion
use crate::typeck::TypeId;

#[derive(Clone, Default)]
struct ScopeFrame {
    /// Values are (type, mutability)
    symbols: BTreeMap<Sym, (TypeId, bool)>,
    types: BTreeMap<Sym, Type>,
}

/// Basic symbol table that maps names to type ID's
/// and manages scope.
/// Looks ugly, works well.
#[derive(Clone)]
pub struct Symtbl {
    frames: Rc<RefCell<Vec<ScopeFrame>>>,
}

impl Default for Symtbl {
    /// We start with an empty toplevel scope existing,
    /// then add some builtin's to it.
    fn default() -> Self {
        Self {
            frames: Rc::new(RefCell::new(vec![ScopeFrame::default()])),
        }
    }
}

pub struct ScopeGuard {
    scope: Symtbl,
}

impl Drop for ScopeGuard {
    fn drop(&mut self) {
        self.scope
            .frames
            .borrow_mut()
            .pop()
            .expect("Scope stack underflow");
    }
}

impl Symtbl {
    pub(crate) fn push_scope(&self) -> ScopeGuard {
        self.frames.borrow_mut().push(ScopeFrame::default());
        ScopeGuard {
            scope: self.clone(),
        }
    }

    pub fn add_var(&self, var: Sym, ty: TypeId, mutable: bool) {
        self.frames
            .borrow_mut()
            .last_mut()
            .expect("Scope stack underflow")
            .symbols
            .insert(var, (ty, mutable));
    }

    /// Checks whether the var exists in the currently alive scopes
    pub fn get_var_binding(&self, var: Sym) -> Option<(TypeId, bool)> {
        for scope in self.frames.borrow().iter().rev() {
            let v = scope.symbols.get(&var);
            if v.is_some() {
                return v.cloned();
            }
        }
        None
    }

    pub(crate) fn add_type(&self, name: Sym, ty: &Type) {
        self.frames
            .borrow_mut()
            .last_mut()
            .expect("Scope stack underflow")
            .types
            .insert(name, ty.to_owned());
    }

    pub(crate) fn get_type(&self, ty: Sym) -> Option<Type> {
        for scope in self.frames.borrow().iter().rev() {
            let v = scope.types.get(&ty);
            if v.is_some() {
                return v.cloned();
            }
        }
        None
    }

    pub fn for_toplevel(&self, f: impl Fn(Sym, TypeId)) {
        for (name, (id, _)) in &self.frames.borrow().last().unwrap().symbols {
            f(*name, *id)
        }
    }
}
