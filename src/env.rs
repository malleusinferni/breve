use std::cell::RefCell;
use std::collections::BTreeMap;

use super::*;

#[derive(Clone)]
pub struct Env(Arc<RefCell<Inner>>);

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum FnKind {
    Function,
    Macro,
}

struct Inner {
    names1: BTreeMap<Symbol, Val>,
    names2: BTreeMap<Symbol, (FnKind, FnRef)>,
    parent: Option<Env>,
}

impl Env {
    pub fn child(&self) -> Self {
        let parent = Some(self.clone());
        let names1 = BTreeMap::default();
        let names2 = BTreeMap::default();
        let inner = Inner { parent, names1, names2 };
        Env(Arc::new(RefCell::new(inner)))
    }

    pub fn lookup1(&self, sym: Symbol) -> Option<Val> {
        let env = self.0.borrow();
        let parent = env.parent.clone();

        env.names1.get(&sym).cloned().or_else(|| {
            parent.and_then(|e| e.lookup1(sym))
        })
    }

    pub fn lookup2(&self, sym: Symbol) -> Option<(FnKind, FnRef)> {
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

    pub fn insert2(&self, sym: Symbol, func: (FnKind, FnRef)) -> Result<()> {
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

#[test]
fn closure() {
    let mut names = NameTable::default();
    let t = names.intern("t");
    let u = names.intern("u");
    let x = names.intern("x");

    let root = Env::default();
    root.insert1(x, Val::Symbol(t)).unwrap();
    assert_eq!(t, root.lookup1(x).unwrap().expect().unwrap());

    let child1 = root.child();
    assert_eq!(t, child1.lookup1(x).unwrap().expect().unwrap());

    child1.update1(x, Val::Symbol(u)).unwrap();
    assert_eq!(u, root.lookup1(x).unwrap().expect().unwrap());

    let _ = root;
    assert_eq!(u, child1.lookup1(x).unwrap().expect().unwrap());
}
