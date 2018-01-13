use std::collections::BTreeMap;

use ordermap::OrderSet;

use super::*;

#[derive(Clone)]
pub enum Op {
    LET(Arc<Bindings>),
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
    args: Bindings,
    body: Func,
}

pub struct Closure {
    env: Env,
    args: Bindings,
    body: Func,
}

#[derive(Clone)]
pub struct Bindings {
    body: OrderSet<Symbol>,
    foot: Option<Symbol>,
}

impl Lambda {
    pub fn eval(self, env: &Env) -> Closure {
        let env = env.clone();
        let Lambda { args, body } = self;
        Closure { env, args, body }
    }
}

impl Closure {
    pub fn call(&self, argv: Val) -> Result<(Env, Func), NameErr> {
        let Closure { ref env, ref args, ref body } = *self;
        let env = env.child();
        args.collect(&env, argv)?;
        Ok((env, body.clone()))
    }

    pub fn as_func(&self) -> &Func {
        &self.body
    }
}

impl Bindings {
    pub fn parse(mut list: Val, names: &NameTable) -> Result<Self> {
        let mut body = OrderSet::new();
        let mut foot = None;

        loop {
            match list {
                Val::Symbol(name) => {
                    foot = Some(name);
                    break;
                },

                Val::Cons(pair) => {
                    let (car, cdr) = pair.as_ref().clone();
                    let name: Symbol = car.expect()?;

                    if body.contains(&name) {
                        names.resolve(name).and_then(|name| {
                            let name = name.to_owned();
                            Err(Error::ArgRedefined { name })
                        })?;
                    }

                    body.insert(name);
                    list = cdr;
                },

                Val::Nil => break,

                _ => return Err(Error::FailedToParseList),
            }
        }

        Ok(Bindings { body, foot })
    }

    pub fn collect(&self, env: &Env, mut val: Val) -> Result<(), NameErr> {
        for &name in self.body.iter() {
            let (car, cdr) = val.uncons().map_err(|_| {
                NameErr::TooFewArgs
            })?;

            env.insert1(name, car)?;
            val = cdr;
        }

        if let Some(name) = self.foot.clone() {
            env.insert1(name, val)?;
            Ok(())
        } else if val.is_nil() {
            Ok(())
        } else {
            Err(NameErr::TooManyArgs)
        }
    }

    pub fn show(&self) -> Val {
        let mut tail = if let Some(foot) = self.foot {
            Val::Symbol(foot)
        } else {
            Val::Nil
        };

        for name in self.body.iter().rev().cloned() {
            tail = Val::Cons((Val::Symbol(name), tail).into());
        }

        tail
    }
}

impl Interpreter {
    pub fn compile(&mut self, input: Vec<Val>) -> Result<Func> {
        let mut compiler = Compiler::new(self);
        for expr in input {
            compiler.tr_expr(expr)?;
        }
        compiler.emit(Op::RET);
        compiler.optimize()?;
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
                        self.tr_form(sym, cdr.as_list()?)?;
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

    fn tr_form(&mut self, name: Symbol, mut args: ListIter) -> Result<()> {
        let string = self.interpreter.name_table()
            .resolve(name)?.to_owned();

        match string.as_str() {
            "quote" => {
                let arg = args.expect()?;
                args.end()?;

                self.emit(Op::QUOTE(arg));
            },

            "fnquote" => {
                let name = args.expect()?;
                args.end()?;

                self.emit(Op::QUOTE(Val::Symbol(name)));
                self.emit(Op::LOAD2);
            },

            "let" => {
                let bindings = Bindings::parse(args.expect()?, &self.interpreter.name_table())?;
                let value = args.expect()?;
                args.end()?;

                self.tr_expr(value)?;
                self.emit(Op::LET(bindings.into()));
            },

            "mut!" => {
                let name = args.expect()?;
                let value = args.expect()?;
                args.end()?;

                self.emit(Op::QUOTE(Val::Symbol(name)));
                self.tr_expr(value)?;
                self.emit(Op::STORE1);
            },

            "def" => {
                let name = args.expect()?;
                let argv = args.expect()?;
                let body = args.collect();

                self.emit(Op::QUOTE(Val::Symbol(name)));
                self.tr_lambda(argv, body)?;
                self.emit(Op::DEF);
            },

            "fn" => {
                let argv = args.expect()?;
                let body = args.collect();

                self.tr_lambda(argv, body)?;
            },

            "if" => {
                let test = args.expect()?;
                let then_do = args.expect()?;
                let or_else = args.expect()?;
                args.end()?;

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
                if !args.has_next() {
                    return Err(Error::TooFewArgs);
                }

                while let Some(arg) = args.next() {
                    self.tr_expr(arg)?;

                    if args.has_next() {
                        self.emit(Op::DROP);
                    }
                }
            },

            "syn" => {
                let name = args.expect()?;
                let argv = args.expect()?;
                let body = args.collect();

                self.emit(Op::QUOTE(Val::Symbol(name)));
                self.tr_lambda(argv, body)?;
                self.emit(Op::SYN);
            },

            "call" => {
                self.tr_expr(args.expect()?)?;

                let argc = args.len();

                for arg in args {
                    self.tr_expr(arg)?;
                }

                self.emit(Op::APPLY(argc));
            },

            "disas" => {
                let name = args.expect()?;
                self.emit(Op::QUOTE(Val::Symbol(name)));
                self.emit(Op::DISAS);
            },

            _ => if self.is_macro(name) {
                let thing = self.interpreter.expand(name, args.collect())?;
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
        let args = Bindings::parse(argv, &self.interpreter.name_table())?;
        let body = self.interpreter.compile(body)?;
        self.emit(Op::LAMBDA(Lambda { args, body }));

        Ok(())
    }

    fn optimize(&mut self) -> Result<()> {
        loop {
            let mut changed = false;

            for pc in 0 .. self.code.len() {
                let label = match &self.code[pc] {
                    &Op::JUMP(label) => label,
                    _ => continue,
                };

                let dst = self.labels[&label];

                if let Op::RET = self.code[dst].clone() {
                    self.code[pc] = Op::RET;
                    changed = true;
                }
            }

            if changed {
                continue;
            }

            return Ok(());
        }
    }
}

impl From<Label> for usize {
    fn from(Label(u): Label) -> Self {
        u
    }
}
