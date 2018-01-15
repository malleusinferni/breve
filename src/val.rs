use std::sync::Arc;

use ordermap::OrderMap;

use opcode::Closure;

use super::*;

#[derive(Clone, Eq, PartialEq)]
pub enum Val {
    Nil,
    True,
    Cons(Arc<(Val, Val)>),
    Int(i32),
    Symbol(Symbol),
    FnRef(FnRef),
}

#[derive(Clone)]
pub enum FnRef {
    Closure(Arc<Closure>),
    Native(Arc<Fn(ListIter) -> Result<Val>>),
}

#[derive(Copy, Clone, Debug, Hash, Eq, Ord, PartialEq, PartialOrd)]
pub struct Symbol(usize);

#[derive(Clone, Debug, Default)]
pub struct NameTable(OrderMap<String, Symbol>);

pub struct ListIter(Vec<Val>);

impl Val {
    pub fn is_list(&self) -> bool {
        match *self {
            Val::Cons(ref pair) => pair.1.is_list(),
            Val::Nil => true,
            _ => false,
        }
    }

    pub fn is_nil(&self) -> bool {
        self == &Val::Nil
    }

    pub fn as_list(mut self) -> Result<ListIter> {
        let mut stack = vec![];

        loop {
            if self.is_nil() {
                stack.reverse();
                return Ok(ListIter(stack));
            }

            let (car, cdr) = self.uncons()?;
            stack.push(car);
            self = cdr;
        }
    }

    pub fn expect<T: Valuable>(self) -> Result<T> {
        T::from_value(self)
    }

    pub fn uncons(self) -> Result<(Self, Self)> {
        match self {
            Val::Cons(pair) => Ok(pair.as_ref().clone()),

            other => Err(Error::WrongType {
                wanted: "cons",
                found: other.type_name(),
            }),
        }
    }

    pub fn type_name(&self) -> &'static str {
        match *self {
            Val::Nil => "nil",
            Val::True => "boolean",
            Val::Int(_) => "int",
            Val::Symbol(_) => "symbol",
            Val::Cons(_) => "cons",
            Val::FnRef(_) => "closure",
        }
    }
}

pub trait Valuable: Sized {
    fn from_value(Val) -> Result<Self>;
}

impl Valuable for Val {
    fn from_value(val: Self) -> Result<Self> {
        Ok(val)
    }
}

impl Valuable for i32 {
    fn from_value(val: Val) -> Result<Self> {
        match val {
            Val::Int(i) => Ok(i),

            other => Err(Error::WrongType {
                wanted: "int",
                found: other.type_name(),
            }),
        }
    }
}

impl Valuable for Symbol {
    fn from_value(val: Val) -> Result<Self> {
        match val {
            Val::Symbol(sym) => Ok(sym),
            other => Err(Error::WrongType {
                wanted: "symbol",
                found: other.type_name(),
            }),
        }
    }
}

impl Valuable for (Val, Val) {
    fn from_value(val: Val) -> Result<Self> {
        match val {
            Val::Cons(pair) => {
                Ok(pair.as_ref().clone())
            },

            other => Err(Error::WrongType {
                wanted: "cons",
                found: other.type_name(),
            }),
        }
    }
}

impl Valuable for FnRef {
    fn from_value(val: Val) -> Result<Self> {
        match val {
            Val::FnRef(func) => Ok(func),

            other => Err(Error::WrongType {
                wanted: "closure",
                found: other.type_name(),
            }),
        }
    }
}

macro_rules! decode_tuple {
    ( $( $param:ident ),* ) => {
        impl<$( $param: Valuable ),*> Valuable for ( $( $param ),* ) {
            fn from_value(val: Val) -> Result<Self> {
                let mut list = val.as_list()?;
                let result = ( $( list.expect::<$param>()? ),* );
                list.end()?;
                Ok(result)
            }
        }
    }
}

decode_tuple!(A, B, C);
decode_tuple!(A, B, C, D);
decode_tuple!(A, B, C, D, E);

impl PartialEq for FnRef {
    fn eq(&self, rhs: &Self) -> bool {
        match (self, rhs) {
            (&FnRef::Closure(ref lhs), &FnRef::Closure(ref rhs)) => {
                Arc::ptr_eq(lhs, rhs)
            },

            (&FnRef::Native(ref lhs), &FnRef::Native(ref rhs)) => {
                Arc::ptr_eq(lhs, rhs)
            },

            _ => false,
        }
    }
}

impl Eq for FnRef {

}

impl Iterator for ListIter {
    type Item = Val;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.pop()
    }
}

impl ListIter {
    pub fn from_iter<I: Iterator<Item=Val>>(iter: I) -> Self {
        let mut stack: Vec<Val> = iter.collect();
        stack.reverse();
        ListIter(stack)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn expect<T: Valuable>(&mut self) -> Result<T> {
        self.next().ok_or(Error::TooFewArgs)?.expect()
    }

    pub fn end(self) -> Result<()> {
        if self.0.is_empty() {
            Ok(())
        } else {
            Err(Error::TooManyArgs)
        }
    }

    pub fn has_next(&self) -> bool {
        self.0.len() > 0
    }
}

impl NameTable {
    pub fn intern(&mut self, name: &str) -> Symbol {
        if !self.0.contains_key(name) {
            let sym = Symbol(self.0.len());
            self.0.insert(name.into(), sym);
        }

        self.0.get(name).cloned().unwrap()
    }

    pub fn gensym(&mut self) -> Symbol {
        let n = format!("#:{:x}", self.0.len());
        self.intern(&n)
    }

    pub fn resolve(&self, sym: Symbol) -> Result<&str> {
        if let Some((name, _)) = self.0.get_index(sym.0) {
            Ok(name)
        } else {
            Err(Error::NoSuchSymbol { symbol: sym })
        }
    }

    pub fn convert_err(&self, err: NameErr) -> Error {
        match err {
            NameErr::NotFound { name } => match self.resolve(name) {
                Ok(name) => Error::NameNotFound { name: name.to_owned() },
                Err(err) => err,
            },

            NameErr::Redefined { name } => match self.resolve(name) {
                Ok(name) => Error::LocalRedefined { name: name.to_owned() },
                Err(err) => err,
            },

            NameErr::TooManyArgs => Error::TooManyArgs,
            NameErr::TooFewArgs => Error::TooFewArgs,
        }
    }
}

impl FromIterator<Val> for Val {
    fn from_iter<T: IntoIterator<Item=Val>>(iter: T) -> Self {
        let mut items: Vec<Val> = iter.into_iter().collect();
        let mut tail = Val::Nil;

        while let Some(car) = items.pop() {
            tail = Val::Cons((car, tail).into());
        }

        tail
    }
}
