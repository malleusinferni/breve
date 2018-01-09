use std::collections::BTreeMap;

use ordermap::OrderSet;

use super::*;

#[derive(Clone)]
pub enum Op {
    LET,
    DEF,
    SYN,
    LOAD1,
    LOAD2,
    STORE1,
    APPLY(usize),
    RET,
    DROP,
    QUOTE(Val),
    JUMP(Label),
    JNZ(Label),
    LAMBDA(Lambda),
    DISAS,
}

#[derive(Copy, Clone, Eq, Ord, PartialEq, PartialOrd)]
pub struct Label(usize);

#[derive(Clone)]
pub struct Func {
    inner: Arc<func::Inner>,
}

mod func {
    use super::*;

    use std::slice::Iter;

    pub struct Inner {
        code: Arc<[Op]>,
        labels: BTreeMap<Label, usize>,
    }

    impl Func {
        pub fn new(code: Vec<Op>, labels: BTreeMap<Label, usize>) -> Self {
            let code = code.into();
            let inner = Inner { code, labels }.into();
            Func { inner }
        }

        pub fn expose(self) -> (Arc<[Op]>, BTreeMap<Label, usize>) {
            (self.inner.code.clone(), self.inner.labels.clone())
        }

        pub fn fetch(&self, pc: usize) -> Option<Op> {
            self.inner.code.get(pc).cloned()
        }

        pub fn jump(&self, label: Label) -> Option<usize> {
            self.inner.labels.get(&label).cloned()
        }

        pub fn iter(&self) -> Iter<Op> {
            self.inner.code.iter()
        }
    }
}

#[derive(Clone)]
pub struct Lambda {
    args: OrderSet<Symbol>,
    body: Func,
}

pub struct Closure {
    env: Env,
    args: OrderSet<Symbol>,
    body: Func,
}

impl Lambda {
    pub fn eval(self, env: &Env) -> Closure {
        let env = env.clone();
        let Lambda { args, body } = self;
        Closure { env, args, body }
    }
}

impl Closure {
    pub fn call(&self, argv: Vec<Val>) -> Result<(Env, Func)> {
        let Closure { ref env, ref args, ref body } = *self;
        let env = env.child();

        let mut argv = argv.into_iter();
        for &name in args {
            let val = argv.next().ok_or(Error::TooFewArgs)?;
            env.insert1(name, val).expect("Thing");
        }

        let _rest: Val = argv.collect();
        // TODO: Use rest

        Ok((env, body.clone()))
    }

    pub fn as_func(&self) -> &Func {
        &self.body
    }
}

impl Op {
    pub fn stack_effect(&self) -> (usize, usize) {
        match *self {
            Op::LET => (2, 1),
            Op::DEF => (2, 1),
            Op::SYN => (2, 1),
            Op::LOAD1 => (2, 1),
            Op::LOAD2 => (2, 1),
            Op::STORE1 => (2, 0),
            Op::APPLY(argc) => (argc + 1, 1),
            Op::RET => (1, 0),
            Op::DROP => (1, 0),
            Op::QUOTE(_) => (0, 1),
            Op::JUMP(_) => (0, 0),
            Op::JNZ(_) => (1, 0),
            Op::LAMBDA(_) => (0, 1),
            Op::DISAS => (1, 0),
        }
    }
}

impl Interpreter {
    pub fn compile(&mut self, input: Vec<Val>) -> Result<Func> {
        let mut compiler = Compiler::new(self);
        for expr in input {
            compiler.tr_expr(expr)?;
        }
        compiler.emit(Op::RET);
        let Compiler { code, labels, .. } = compiler;
        Ok(Func::new(code, labels))
    }
}

struct Compiler<'a> {
    code: Vec<Op>,
    labels: BTreeMap<Label, usize>,
    interpreter: &'a mut Interpreter,
    next_label: usize,
}

impl<'a> Compiler<'a> {
    fn new(interpreter: &'a mut Interpreter) -> Self {
        let code = vec![];
        let labels = BTreeMap::new();
        let next_label = 0;
        Compiler { code, labels, interpreter, next_label }
    }

    fn emit(&mut self, op: Op) {
        self.code.push(op);
    }

    fn make_label(&mut self) -> Label {
        let label = Label(self.next_label);
        self.next_label += 1;
        label
    }

    fn set_label(&mut self, label: Label) -> Result<()> {
        if self.labels.contains_key(&label) {
            Err(Error::LabelRedefined)
        } else {
            let pc = self.code.len();
            self.labels.insert(label, pc);
            Ok(())
        }
    }

    fn is_macro(&mut self, name: Symbol) -> bool {
        self.interpreter.root.lookup2(name)
            .map(|(kind, _)| kind == FnKind::Macro)
            .unwrap_or(false)
    }

    fn tr_expr(&mut self, expr: Val) -> Result<()> {
        match expr {
            Val::Symbol(sym) => {
                self.emit(Op::QUOTE(Val::Symbol(sym)));
                self.emit(Op::LOAD1);
            },

            Val::Cons(pair) => {
                let (car, cdr) = pair.as_ref().clone();
                match car {
                    Val::Symbol(sym) => {
                        let mut args = vec![];
                        for arg in cdr {
                            args.push(arg?);
                        }

                        self.tr_form(sym, args)?;
                    },

                    form => {
                        let form = self.interpreter.show(form)?;
                        return Err(Error::UnimplementedForm { form });
                    },
                }
            },

            other => {
                self.emit(Op::QUOTE(other));
            },
        }

        Ok(())
    }

    fn tr_form(&mut self, name: Symbol, mut args: Vec<Val>) -> Result<()> {
        let string = self.interpreter.name_table()
            .resolve(name)?.to_owned();

        match string.as_str() {
            "quote" => {
                let arg = args.pop().ok_or(Error::TooFewArgs)?;
                guard(args.is_empty(), || Error::TooManyArgs)?;
                self.emit(Op::QUOTE(arg));
            },

            "fnquote" => {
                let name = args.pop().ok_or(Error::TooFewArgs)?.expect()?;
                guard(args.is_empty(), || Error::TooManyArgs)?;
                self.emit(Op::QUOTE(Val::Symbol(name)));
                self.emit(Op::LOAD2);
            },

            "let" => {
                let value = args.pop().ok_or(Error::TooFewArgs)?;
                let name = args.pop().ok_or(Error::TooFewArgs)?.expect()?;
                guard(args.is_empty(), || Error::TooManyArgs)?;
                self.emit(Op::QUOTE(Val::Symbol(name)));
                self.tr_expr(value)?;
                self.emit(Op::LET);
            },

            "def" => {
                let body = args.drain(2 ..).collect();
                let argv = args.pop().ok_or(Error::TooFewArgs)?;
                let name = args.pop().ok_or(Error::TooFewArgs)?.expect()?;
                guard(args.is_empty(), || Error::TooManyArgs)?;
                self.emit(Op::QUOTE(Val::Symbol(name)));
                self.tr_lambda(argv, body)?;
                self.emit(Op::DEF);
            },

            "fn" => {
                let body = args.drain(1 ..).collect();
                let argv = args.pop().ok_or(Error::TooFewArgs)?;
                guard(args.is_empty(), || Error::TooManyArgs)?;
                self.tr_lambda(argv, body)?;
            },

            "if" => {
                let or_else = args.pop().ok_or(Error::TooFewArgs)?;
                let then_do = args.pop().ok_or(Error::TooFewArgs)?;
                let test = args.pop().ok_or(Error::TooFewArgs)?;
                guard(args.is_empty(), || Error::TooManyArgs)?;

                let before = self.make_label();
                let after = self.make_label();

                self.tr_expr(test)?;
                self.emit(Op::JNZ(before));
                self.tr_expr(or_else)?;
                self.emit(Op::JUMP(after));
                self.set_label(before)?;
                self.tr_expr(then_do)?;
                self.set_label(after)?;
            },

            "do" => {
                guard(args.len() > 0, || Error::TooFewArgs)?;

                args.reverse();
                while let Some(arg) = args.pop() {
                    self.tr_expr(arg)?;

                    if args.len() > 0 {
                        self.emit(Op::DROP);
                    }
                }
            },

            "syn" => {
                let body = args.drain(2 ..).collect();
                let argv = args.pop().ok_or(Error::TooFewArgs)?;
                let name = args.pop().ok_or(Error::TooFewArgs)?.expect()?;
                guard(args.is_empty(), || Error::TooManyArgs)?;
                self.emit(Op::QUOTE(Val::Symbol(name)));
                self.tr_lambda(argv, body)?;
                self.emit(Op::SYN);
            },

            "call" => {
                let argc = args.len().checked_sub(1)
                    .ok_or(Error::TooFewArgs)?;

                for arg in args.drain(..) {
                    self.tr_expr(arg)?;
                }

                self.emit(Op::APPLY(argc));
            },

            "disas" => {
                let name = args.pop().ok_or(Error::TooFewArgs)?.expect()?;
                self.emit(Op::QUOTE(Val::Symbol(name)));
                self.emit(Op::DISAS);
            },

            _ => if self.is_macro(name) {
                let thing = self.interpreter.expand(name, args)?;
                self.tr_expr(thing)?;
            } else {
                self.emit(Op::QUOTE(Val::Symbol(name)));
                self.emit(Op::LOAD2);

                let argc = args.len();

                for arg in args {
                    self.tr_expr(arg)?;
                }

                self.emit(Op::APPLY(argc));
            },
        }

        Ok(())
    }

    fn tr_lambda(&mut self, argv: Val, body: Vec<Val>) -> Result<()> {
        let mut args = OrderSet::new();
        for arg in argv {
            let name: Symbol = arg?.expect()?;
            if args.contains(&name) {
                self.interpreter.name_table().resolve(name).and_then(|name| {
                    let name = name.to_owned();
                    Err(Error::ArgRedefined { name })
                })?;
            }
            args.insert(name);
        }

        let body = self.interpreter.compile(body)?;

        self.emit(Op::LAMBDA(Lambda { args, body }));

        Ok(())
    }
}

fn guard<F: FnOnce() -> Error>(test: bool, fail: F) -> Result<()> {
    if test {
        Ok(())
    } else {
        Err(fail())
    }
}

impl From<Label> for usize {
    fn from(Label(u): Label) -> Self {
        u
    }
}
