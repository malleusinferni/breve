extern crate ordermap;

#[macro_use]
extern crate failure_derive;
extern crate failure;

use std::sync::Arc;

use std::iter::FromIterator;

pub mod val;
pub mod env;
pub mod parse;
pub mod opcode;

pub type Result<T, E=Error> = std::result::Result<T, E>;

pub use env::*;
pub use val::*;

#[derive(Debug, Fail)]
pub enum Error {
    #[fail(display="expected {}, found {}", wanted, found)]
    WrongType {
        wanted: &'static str,
        found: &'static str,
    },

    #[fail(display="expression is not a list")]
    NotAList,

    #[fail(display="no such symbol {:?}", symbol)]
    NoSuchSymbol {
        symbol: Symbol,
    },

    #[fail(display="stack underflow in call stack")]
    CallStackUnderflow,

    #[fail(display="stack underflow in local expression")]
    ExprStackUnderflow,

    #[fail(display="unmatched closing parenthesis")]
    UnmatchedRightParen,

    #[fail(display="failed to parse list")]
    FailedToParseList,

    #[fail(display="unexpected end of input")]
    UnexpectedEof,

    #[fail(display="name not found: {}", name)]
    NameNotFound {
        name: String,
    },

    #[fail(display="no such label")]
    NoSuchLabel,

    #[fail(display="too few arguments for function call")]
    TooFewArgs,

    #[fail(display="too many arguments in function call")]
    TooManyArgs,

    #[fail(display="internal error: label redefined")]
    LabelRedefined,

    #[fail(display="argument {} redefined", name)]
    ArgRedefined {
        name: String,
    },

    #[fail(display="local variable {} redefined", name)]
    LocalRedefined {
        name: String,
    },

    #[fail(display="encountered an illegal token")]
    IllegalToken,

    #[fail(display="illegal macro invocation")]
    MacroCall,

    #[fail(display="unimplemented form {}", form)]
    UnimplementedForm { form: String },
}

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

impl Frame {
    fn fetch(&mut self) -> Option<opcode::Op> {
        self.func.fetch(self.pc).map(|op| {
            self.pc += 1;
            op
        })
    }

    fn will_return(&self) -> bool {
        use opcode::Op;

        match self.func.fetch(self.pc) {
            None | Some(Op::RET) => true,
            _ => false,
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
            let car = argv.expect()?;
            let cdr = argv.expect()?;
            argv.end()?;
            Ok(Val::Cons((car, cdr).into()))
        })?;

        it.def("+", |mut argv| {
            let mut sum: i32 = argv.expect()?;
            while argv.has_next() {
                sum += argv.expect::<i32>()?;
            }
            Ok(Val::Int(sum))
        })?;

        it.def("-", |mut argv| {
            let mut sum: i32 = argv.expect()?;

            if argv.has_next() {
                while argv.has_next() {
                    sum -= argv.expect::<i32>()?;
                }
            } else {
                sum = -sum;
            }

            Ok(Val::Int(sum))
        })?;

        it.def("*", |mut argv| {
            let mut prod: i32 = argv.expect()?;

            while argv.has_next() {
                prod *= argv.expect::<i32>()?;
            }

            Ok(Val::Int(prod))
        })?;

        let t = it.names.intern("t");

        it.def("=", move |mut argv| {
            let lhs: Val = argv.expect()?;
            let rhs: Val = argv.expect()?;
            argv.end()?;

            if lhs == rhs {
                Ok(Val::Symbol(t))
            } else {
                Ok(Val::Nil)
            }
        })?;

        it.def("<", move |mut argv| {
            let mut lhs: i32 = argv.expect()?;

            while argv.has_next() {
                let rhs: i32 = argv.expect()?;
                if lhs < rhs {
                    lhs = rhs;
                } else {
                    return Ok(Val::Nil);
                }
            }

            Ok(Val::Symbol(t))
        })?;

        it.def(">", move |mut argv| {
            let mut lhs: i32 = argv.expect()?;

            while argv.has_next() {
                let rhs: i32 = argv.expect()?;
                if lhs > rhs {
                    lhs = rhs;
                } else {
                    return Ok(Val::Nil);
                }
            }

            Ok(Val::Symbol(t))
        })?;

        it.def("list?", move |mut argv| {
            let expr: Val = argv.expect()?;
            argv.end()?;

            if expr.is_list() {
                Ok(Val::Symbol(t))
            } else {
                Ok(Val::Nil)
            }
        })?;

        {
            let nil = it.names.intern("nil");
            let int = it.names.intern("int");
            let symbol = it.names.intern("symbol");
            let cons = it.names.intern("cons");
            let closure = it.names.intern("closure");

            it.def("type", move |mut argv| {
                let expr = argv.expect()?;
                argv.end()?;

                Ok(Val::Symbol(match expr {
                    Val::Nil => nil,
                    Val::Int(_) => int,
                    Val::Symbol(_) => symbol,
                    Val::Cons(_) => cons,
                    Val::FnRef(_) => closure,
                }))
            })?;
        }

        let stdlib = include_str!("stdlib.breve");
        it.parse(&stdlib).and_then(|forms| {
            for form in forms {
                it.eval(form)?;
            }

            Ok(())
        })?;

        Ok(it)
    }

    pub fn def<F>(&mut self, name: &str, body: F) -> Result<()>
        where F: 'static + Fn(ListIter) -> Result<Val>
    {
        let sym = self.names.intern(name);
        let func = FnRef::Native(Arc::new(body));
        let kind = FnKind::Function;
        let _redef = self.root.insert2(sym, (kind, func));

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

    pub fn expand(&mut self, name: Symbol, expr: Val) -> Result<Val> {
        let (kind, func) = self.root.lookup2(name).map_err(|err| {
            self.names.convert_err(err)
        })?;

        if let FnKind::Function = kind {
            return Err(Error::MacroCall);
        }

        match func {
            FnRef::Closure(func) => {
                let (env, func) = func.call(expr).map_err(|err| {
                    self.names.convert_err(err)
                })?;

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
                func(expr.as_list()?)
            },
        }
    }

    pub fn name_table(&mut self) -> &mut NameTable {
        &mut self.names
    }

    pub fn show(&self, val: Val) -> Result<String> {
        val.show(&self.names)
    }
}

impl Val {
    pub fn show(&self, names: &NameTable) -> Result<String> {
        let mut buf = String::new();

        {
            let mut f = Fmt {
                buf: &mut buf,
                names,
            };

            f.write(self)?;
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
        self.frame().data.pop().ok_or(Error::ExprStackUnderflow)?.expect()
    }

    fn step(&mut self, op: opcode::Op) -> Result<()> {
        use opcode::Op;

        match op {
            Op::LET(bindings) => {
                let argv = self.pop::<Val>()?;

                bindings.collect(&self.frame().env, argv).map_err(|err| {
                    self._names.convert_err(err)
                })?;

                self.push(Val::Nil);
            },

            Op::LET2(name) => {
                let body: FnRef = self.pop()?;
                self.frame().env.let2(name, body).map_err(|err| {
                    self._names.convert_err(err)
                })?;
                self.push(Val::Symbol(name));
            },

            Op::DEF(name) => {
                let body: FnRef = self.pop()?;
                let kind = FnKind::Function;
                let _redef = self.root.env.insert2(name, (kind, body));
                self.push(Val::Symbol(name));
            },

            Op::SYN(name) => {
                let body: FnRef = self.pop()?;
                let kind = FnKind::Macro;
                let _redef = self.root.env.insert2(name, (kind, body));
                self.push(Val::Symbol(name));
            },

            Op::LOAD1(name) => {
                let val = self.frame().env.lookup1(name).map_err(|err| {
                    self._names.convert_err(err)
                })?;

                self.push(val);
            },

            Op::LOAD2(name) => {
                let (kind, func) = self.frame().env.lookup2(name).map_err(|err| {
                    self._names.convert_err(err)
                })?;

                if let FnKind::Function = kind {
                    self.push(Val::FnRef(func));
                } else {
                    return Err(Error::MacroCall);
                }
            },

            Op::STORE1(name) => {
                let val: Val = self.pop()?;
                match self.frame().env.update1(name, val) {
                    Ok(previous) => self.push(previous),
                    Err(err) => Err(self._names.convert_err(err))?,
                }
            },

            Op::COLLECT(argc) => {
                let args = self.collect(argc)?;
                self.push(args);
            },

            Op::APPLY => {
                let args = self.pop()?;
                let func: FnRef = self.pop()?;
                self.apply(func, args)?;
            },

            Op::RET => {
                let value: Val = self.pop().unwrap_or(Val::Nil);
                let _ = self.call_stack.pop()
                    .ok_or(Error::CallStackUnderflow)?;
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

            Op::DISAS => {
                let name: Symbol = self.pop()?;

                let (_, func) = self.frame().env.lookup2(name).map_err(|err| {
                    self._names.convert_err(err)
                })?;

                if let FnRef::Closure(closure) = func {
                    self.disas(closure.as_func())?;
                } else {
                    println!("Native function");
                }
            },
        }

        Ok(())
    }

    fn collect(&mut self, len: usize) -> Result<Val> {
        let start = self.frame().data.len().checked_sub(len)
            .ok_or(Error::ExprStackUnderflow)?;
        Ok(ListIter::from_iter(self.frame().data.drain(start ..)).collect())
    }

    fn apply(&mut self, func: FnRef, args: Val) -> Result<()> {
        match func {
            FnRef::Native(native) => {
                self.push(native(args.as_list()?)?);
            },

            FnRef::Closure(closure) => {
                let (env, func) = closure.call(args).map_err(|err| {
                    self._names.convert_err(err)
                })?;

                let frame = Frame {
                    env,
                    func,
                    data: vec![],
                    pc: 0,
                };

                if self.frame().will_return() {
                    *self.frame() = frame;
                } else {
                    self.call_stack.push(frame);
                }
            },
        }

        Ok(())
    }

    fn disas(&self, func: &opcode::Func) -> Result<()> {
        use opcode::Op;

        let label = |pc| func.jump(pc).ok_or(Error::NoSuchLabel);

        let sym = |sym| match self._names.resolve(sym) {
            Ok(s) => s.to_owned(),
            Err(_) => format!("#:{:?}", sym),
        };

        for (pc, op) in func.iter().enumerate() {
            print!("{:04X}\t", pc);

            match *op {
                Op::LET(ref bindings) => {
                    println!("LET {}", bindings.show().show(&self._names)?)
                },

                Op::LET2(name) => println!("LET2 {}", sym(name)),
                Op::DEF(name) => println!("DEF {}", sym(name)),
                Op::SYN(name) => println!("SYN {}", sym(name)),
                Op::LOAD1(name) => println!("LOAD1 {}", sym(name)),
                Op::LOAD2(name) => println!("LOAD2 {}", sym(name)),
                Op::STORE1(name) => println!("STORE1 {}", sym(name)),

                Op::COLLECT(argc) => println!("COLLECT {}", argc),
                Op::APPLY => println!("APPLY"),

                Op::RET => println!("RET"),
                Op::DROP => println!("DROP"),

                Op::QUOTE(ref val) => {
                    println!("QUOTE {}", val.show(&self._names)?)
                },

                Op::JUMP(pc) => println!("JUMP {:04X}", label(pc)?),
                Op::JNZ(pc) => println!("JNZ {:04X}", label(pc)?),

                Op::LAMBDA(_) => println!("LAMBDA ..."),

                Op::DISAS => println!("DISAS"),
            }
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
        match *val {
            Val::Symbol(sym) => {
                let name = self.names.resolve(sym)?;
                self.buf.push_str(name);
            },

            Val::Int(int) => {
                self.buf.push_str(&format!("{}", int));
            },

            Val::Nil => self.buf.push_str("()"),

            Val::Cons(ref pair) => {
                let mut pair: &(Val, Val) = pair.as_ref();
                let mut first = true;
                self.buf.push('(');

                loop {
                    let (ref car, ref cdr) = *pair;

                    if first {
                        first = false;
                    } else {
                        self.buf.push(' ');
                    }

                    self.write(car)?;

                    pair = match cdr {
                        &Val::Cons(ref pair) => pair,

                        &Val::Nil => break,

                        other => {
                            self.buf.push_str(" & ");
                            self.write(other)?;
                            break;
                        },
                    }
                }

                self.buf.push(')');
            },

            Val::FnRef(_) => {
                self.buf.push_str("#'function");
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
