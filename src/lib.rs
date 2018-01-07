extern crate ordermap;

use std::sync::Arc;

use std::iter::FromIterator;

use ordermap::OrderMap;

pub mod env;
pub mod parse;
pub mod opcode;

pub type Result<T, E=Error> = std::result::Result<T, E>;

pub use env::*;

#[derive(Clone, Debug, Default)]
pub struct NameTable(OrderMap<String, Symbol>);

#[derive(Debug)]
pub enum Error {
    WrongType {
        wanted: &'static str,
        found: &'static str,
    },
    NotAList,
    NoSuchSymbol,
    StackUnderflow,
    UnmatchedRightParen,
    FailedToParseList,
    UnexpectedEof,
    NameNotFound,
    NoSuchLabel,
    TooFewArgs,
    TooManyArgs,
    LabelRedefined,
    ArgRedefined,
    IllegalToken,
    MacroCall,
    UnimplementedForm { form: String },
}

#[derive(Clone, Eq, PartialEq)]
pub enum Val {
    Nil,
    Cons(Arc<(Val, Val)>),
    Int(i32),
    Symbol(Symbol),
    FnRef(FnRef),
}

#[derive(Clone)]
pub enum FnRef {
    Closure(Arc<opcode::Closure>),
    Native(Arc<Fn(ArgIter) -> Result<Val>>),
}

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

#[derive(Copy, Clone, Debug, Hash, Eq, Ord, PartialEq, PartialOrd)]
pub struct Symbol(usize);

pub struct Interpreter {
    names: NameTable,
    root: Env,
}

struct Eval<'a> {
    _names: &'a mut NameTable,
    root: Frame,
    call_stack: Vec<Frame>,
}

struct Frame {
    env: Env,
    data: Vec<Val>,
    func: opcode::Func,
    pc: usize,
}

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

    fn expect<T: Valuable>(self) -> Result<T> {
        T::from_value(self)
    }

    pub fn type_name(&self) -> &'static str {
        match *self {
            Val::Nil => "nil",
            Val::Int(_) => "int",
            Val::Symbol(_) => "symbol",
            Val::Cons(_) => "cons",
            Val::FnRef(_) => "closure",
        }
    }
}

trait Valuable: Sized {
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

impl IntoIterator for Val {
    type Item = Result<Val>;
    type IntoIter = ListIter;

    fn into_iter(self) -> ListIter {
        ListIter(self)
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

pub struct ListIter(Val);

pub struct ArgIter(Vec<Val>);

impl Iterator for ListIter {
    type Item = Result<Val>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.0.clone() {
            Val::Cons(pair) => {
                let (car, cdr) = pair.as_ref().clone();
                self.0 = cdr;
                Some(Ok(car))
            },

            Val::Nil => None,

            _ => Some(Err(Error::NotAList)),
        }
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

    pub fn resolve(&self, sym: Symbol) -> Result<&str> {
        if let Some((name, _)) = self.0.get_index(sym.0) {
            Ok(name)
        } else {
            Err(Error::NoSuchSymbol)
        }
    }
}

impl Frame {
    fn fetch(&mut self) -> Option<opcode::Op> {
        self.func.fetch(self.pc).map(|op| {
            self.pc += 1;
            op
        })
    }
}

impl ArgIter {
    pub fn next(&mut self) -> Result<Val> {
        self.0.pop().ok_or(Error::TooFewArgs)
    }

    pub fn has_next(&self) -> bool {
        self.0.len() > 0
    }

    pub fn end(self) -> Result<()> {
        if self.has_next() {
            Err(Error::TooManyArgs)
        } else {
            Ok(())
        }
    }
}

impl Interpreter {
    pub fn new() -> Result<Self> {
        let mut it = Interpreter {
            root: Env::default(),
            names: NameTable::default(),
        };

        it.def("cons", |mut argv| {
            let car = argv.next()?;
            let cdr = argv.next()?;
            argv.end()?;
            Ok(Val::Cons((car, cdr).into()))
        })?;

        it.def("car", |mut argv| {
            let (car, _) = argv.next()?.expect()?;
            argv.end()?;
            Ok(car)
        })?;

        it.def("cdr", |mut argv| {
            let (_, cdr) = argv.next()?.expect()?;
            argv.end()?;
            Ok(cdr)
        })?;

        it.def("list", |mut argv| {
            argv.0.reverse();
            Ok(argv.0.into_iter().collect())
        })?;

        it.def("+", |mut argv| {
            let mut sum: i32 = argv.next()?.expect()?;
            while argv.has_next() {
                sum += argv.next()?.expect::<i32>()?;
            }
            Ok(Val::Int(sum))
        })?;

        it.def("-", |mut argv| {
            let mut sum: i32 = argv.next()?.expect()?;

            if argv.has_next() {
                while argv.has_next() {
                    sum -= argv.next()?.expect::<i32>()?;
                }
            } else {
                sum = -sum;
            }

            Ok(Val::Int(sum))
        })?;

        let t = it.names.intern("t");

        it.def("=", move |mut argv| {
            let lhs = argv.next()?;
            let rhs = argv.next()?;
            argv.end()?;

            if lhs == rhs {
                Ok(Val::Symbol(t))
            } else {
                Ok(Val::Nil)
            }
        })?;

        it.def("<", move |mut argv| {
            let mut lhs: i32 = argv.next()?.expect()?;

            for rhs in argv.0.drain(..) {
                let rhs: i32 = rhs.expect()?;
                if lhs < rhs {
                    lhs = rhs;
                } else {
                    return Ok(Val::Nil);
                }
            }

            Ok(Val::Symbol(t))
        })?;

        it.def(">", move |mut argv| {
            let mut lhs: i32 = argv.next()?.expect()?;

            for rhs in argv.0.drain(..) {
                let rhs: i32 = rhs.expect()?;
                if lhs > rhs {
                    lhs = rhs;
                } else {
                    return Ok(Val::Nil);
                }
            }

            Ok(Val::Symbol(t))
        })?;

        Ok(it)
    }

    pub fn def<F>(&mut self, name: &str, body: F) -> Result<()>
        where F: 'static + Fn(ArgIter) -> Result<Val>
    {
        let sym = self.names.intern(name);
        let func = FnRef::Native(Arc::new(body));
        let kind = FnKind::Function;
        self.root.insert2(sym, (kind, func))?;
        Ok(())
    }

    pub fn eval(&mut self, val: Val) -> Result<Val> {
        let func = self.compile(vec![val])?;
        let env = self.root.clone();

        let root = Frame {
            env,
            func,
            data: vec![],
            pc: 0,
        };

        let eval = Eval {
            _names: &mut self.names,
            root,
            call_stack: vec![],
        };

        eval.finish()
    }

    pub fn expand(&mut self, name: Symbol, args: Vec<Val>) -> Result<Val> {
        let (kind, func) = self.root.lookup2(name)
            .ok_or(Error::NameNotFound)?;

        if let FnKind::Function = kind {
            return Err(Error::MacroCall);
        }

        match func {
            FnRef::Closure(func) => {
                let (env, func) = func.call(args)?;

                let root = Frame {
                    env,
                    func,
                    data: vec![],
                    pc: 0,
                };

                let mut eval = Eval {
                    root,
                    _names: &mut self.names,
                    call_stack: vec![],
                };

                eval.finish()
            },

            FnRef::Native(func) => {
                let mut args = ArgIter(args);
                args.0.reverse();
                func(args)
            },
        }
    }

    pub fn name_table(&mut self) -> &mut NameTable {
        &mut self.names
    }

    pub fn show(&self, val: Val) -> Result<String> {
        let mut buf = String::new();

        {
            let mut f = Fmt {
                buf: &mut buf,
                names: &self.names,
            };

            f.write(&val)?;
        }

        Ok(buf)
    }
}

impl<'a> Eval<'a> {
    fn finish(mut self) -> Result<Val> {
        while let Some(op) = self.frame().fetch() {
            use opcode::Op;

            if self.call_stack.is_empty() {
                if let Op::RET = op {
                    break;
                }
            }

            self.step(op)?;
        }

        Ok(self.root.data.pop().unwrap_or(Val::Nil))
    }

    fn frame(&mut self) -> &mut Frame {
        if let Some(frame) = self.call_stack.last_mut() {
            frame
        } else {
            &mut self.root
        }
    }

    fn push(&mut self, val: Val) {
        self.frame().data.push(val);
    }

    fn pop<T: Valuable>(&mut self) -> Result<T> {
        self.frame().data.pop().ok_or(Error::StackUnderflow)?.expect()
    }

    fn step(&mut self, op: opcode::Op) -> Result<()> {
        use opcode::Op;

        match op {
            Op::LET => {
                let val = self.pop()?;
                let name: Symbol = self.pop()?;
                self.frame().env.insert1(name, val)?;
                self.push(Val::Symbol(name));
            },

            Op::DEF => {
                let body: FnRef = self.pop()?;
                let name: Symbol = self.pop()?;
                let kind = FnKind::Function;
                self.root.env.insert2(name, (kind, body))?;
                self.push(Val::Symbol(name));
            },

            Op::SYN => {
                let body: FnRef = self.pop()?;
                let name: Symbol = self.pop()?;
                let kind = FnKind::Macro;
                self.root.env.insert2(name, (kind, body))?;
                self.push(Val::Symbol(name));
            },

            Op::LOAD1 => {
                let name: Symbol = self.pop()?;
                let val = self.frame().env.lookup1(name)
                    .ok_or(Error::NameNotFound)?;
                self.push(val);
            },

            Op::LOAD2 => {
                let name: Symbol = self.pop()?;
                let (kind, func) = self.frame().env.lookup2(name)
                    .ok_or(Error::NameNotFound)?;

                if let FnKind::Function = kind {
                    self.push(Val::FnRef(func));
                } else {
                    return Err(Error::MacroCall);
                }
            },

            Op::STORE1 => {
                let name: Symbol = self.pop()?;
                let val: Val = self.pop()?;
                self.frame().env.update1(name, val)?;
            },

            Op::APPLY(argc) => {
                let args = self.collect(argc)?;
                let func: FnRef = self.pop()?;
                self.apply(func, args)?;
            },

            Op::RET => {
                let value: Val = self.pop()?;
                let _ = self.call_stack.pop()
                    .ok_or(Error::StackUnderflow)?;
                self.push(value);
            },

            Op::DROP => {
                self.frame().data.pop();
            },

            Op::QUOTE(val) => {
                self.push(val);
            },

            Op::JUMP(label) => {
                let pc = self.frame().func.jump(label)
                    .ok_or(Error::NoSuchLabel)?;
                self.frame().pc = pc;
            },

            Op::JNZ(label) => {
                let test: Val = self.pop()?;

                if !test.is_nil() {
                    let pc = self.frame().func.jump(label)
                        .ok_or(Error::NoSuchLabel)?;
                    self.frame().pc = pc;
                }
            },

            Op::LAMBDA(lambda) => {
                let closure = lambda.eval(&self.frame().env);
                self.push(Val::FnRef(FnRef::Closure(closure.into())));
            },
        }

        Ok(())
    }

    fn collect(&mut self, len: usize) -> Result<Vec<Val>> {
        let start = self.frame().data.len().checked_sub(len)
            .ok_or(Error::StackUnderflow)?;
        Ok(self.frame().data.drain(start ..).collect())
    }

    fn apply(&mut self, func: FnRef, mut args: Vec<Val>) -> Result<()> {
        match func {
            FnRef::Native(native) => {
                args.reverse();
                let args = ArgIter(args);
                self.push(native(args)?);
            },

            FnRef::Closure(closure) => {
                let (env, func) = closure.call(args)?;

                let frame = Frame {
                    env,
                    func,
                    data: vec![],
                    pc: 0,
                };

                use opcode::Op;

                if let Some(Op::RET) = self.frame().fetch() {
                    *self.frame() = frame;
                } else {
                    self.call_stack.push(frame);
                }
            },
        }

        Ok(())
    }
}

struct Fmt<'a> {
    buf: &'a mut String,
    names: &'a NameTable,
}

impl<'a> Fmt<'a> {
    fn write(&mut self, val: &Val) -> Result<()> {
        if val.is_list() {
            self.buf.push('(');

            let mut first = true;

            for item in val.clone() {
                let item = item?;

                if first {
                    first = false;
                } else {
                    self.buf.push(' ');
                }

                self.write(&item)?;
            }

            self.buf.push(')');
        } else {
            match *val {
                Val::Symbol(sym) => {
                    let name = self.names.resolve(sym)?;
                    self.buf.push_str(name);
                },

                Val::Int(int) => {
                    self.buf.push_str(&format!("{}", int));
                },

                Val::Nil => self.buf.push_str("nil"),

                Val::Cons(ref pair) => {
                    let (ref car, ref cdr) = *pair.as_ref();
                    self.buf.push('(');
                    self.write(car)?;
                    self.buf.push_str(" . ");
                    self.write(cdr)?;
                    self.buf.push(')');
                },

                Val::FnRef(_) => {
                    self.buf.push_str("#'function");
                }
            }
        }

        Ok(())
    }
}

#[test]
fn basic_parse() {
    let source = "(car cdr eval apply)";

    let mut interpreter = Interpreter::new().unwrap();
    let parsed = interpreter.parse(source).unwrap();
    assert_eq!(parsed.len(), 1);
    let list = parsed.into_iter().next().unwrap();
    let showed = interpreter.show(list).unwrap();
    assert_eq!(source, &showed);
}

#[test]
fn basic_eval() {
    let source = "(def double (x) (+ x x)) (double 2); comment";
    let mut interpreter = Interpreter::new().unwrap();
    let four = interpreter.parse(source).and_then(|ast| -> Result<Option<Val>> {
        let mut ret = None;
        for expr in ast {
            ret = Some(interpreter.eval(expr)?);
        }
        Ok(ret)
    }).unwrap();

    assert!(four == Some(Val::Int(4)));
}
