use std::cell::RefCell;
use std::collections::BTreeMap;

use super::*;

#[derive(Clone)]
pub struct Env(Arc<RefCell<Inner>>);

struct Inner {
    names1: BTreeMap<Symbol, Val>,
    names2: BTreeMap<Symbol, FnRef>,
    parent: Option<Env>,
}

impl Env {
    pub fn with_parent(env: Env) -> Self {
        let parent = Some(env);
        let names1 = BTreeMap::default();
        let names2 = BTreeMap::default();
        Env(Arc::new(RefCell::new(Inner { parent, names1, names2 })))
    }

    pub fn lookup1(&self, sym: Symbol) -> Option<Val> {
        let env = self.0.borrow();
        let parent = env.parent.clone();

        env.names1.get(&sym).cloned().or_else(|| {
            parent.and_then(|e| e.lookup1(sym))
        })
    }

    pub fn lookup2(&self, sym: Symbol) -> Option<FnRef> {
        let env = self.0.borrow();
        let parent = env.parent.clone();

        env.names2.get(&sym).cloned().or_else(|| {
            parent.and_then(|e| e.lookup2(sym))
        })
    }

    pub fn insert1(&self, sym: Symbol, val: Val) -> Result<()> {
        let mut env = self.0.borrow_mut();
        if env.names1.contains_key(&sym) {
            Err(Error::NameNotFound)
        } else {
            env.names1.insert(sym, val);
            Ok(())
        }
    }

    pub fn insert2(&self, sym: Symbol, func: FnRef) -> Result<()> {
        let mut env = self.0.borrow_mut();
        env.names2.insert(sym, func);
        Ok(())
    }

    pub fn update1(&self, sym: Symbol, val: Val) -> Result<()> {
        let mut env = self.0.borrow_mut();
        if env.names1.contains_key(&sym) {
            env.names1.insert(sym, val);
            Ok(())
        } else if let Some(parent) = env.parent.clone() {
            parent.update1(sym, val)
        } else {
            Err(Error::NameNotFound)
        }
    }
}

impl Default for Env {
    fn default() -> Self {
        Env(Arc::new(RefCell::new(Inner {
            names1: Default::default(),
            names2: Default::default(),
            parent: None,
        })))
    }
}
