use super::*;

use std::str::Chars;
use std::iter::Peekable;

impl Interpreter {
    pub fn parse(&mut self, input: &str) -> Result<Vec<Val>> {
        let mut values = vec![];
        let mut stream = Stream {
            input: input.chars().peekable(),
            names: &mut self.names,
        };

        loop {
            stream.skip_whitespace()?;

            if stream.lookahead().is_some() {
                values.push(stream.next_value()?);
            } else {
                break;
            }
        }

        Ok(values)
    }
}

struct Stream<'a> {
    input: Peekable<Chars<'a>>,
    names: &'a mut NameTable,
}

impl<'a> Stream<'a> {
    fn lookahead(&mut self) -> Option<char> {
        self.input.peek().cloned()
    }

    fn next_value(&mut self) -> Result<Val> {
        match self.input.next().ok_or(Error::UnexpectedEof)? {
            '(' => self.list_tail(),

            ')' => Err(Error::UnmatchedRightParen),

            '\'' => self.quoted("quote"),

            '`' => self.quoted("quasi"),

            ',' => if self.lookahead() == Some('@') {
                self.input.next();
                self.quoted("unsplice")
            } else {
                self.quoted("unquote")
            },

            w => {
                let mut name = String::new();
                name.push(w);

                while let Some(w) = self.lookahead() {
                    if "(',`);".contains(w) || w.is_whitespace() {
                        break;
                    } else {
                        name.push(w);
                        self.input.next();
                        continue;
                    }
                }

                if let Ok(int) = name.parse::<i32>() {
                    return Ok(Val::Int(int));
                }

                let sym = self.names.intern(&name);
                Ok(Val::Symbol(sym))
            },
        }
    }

    fn quoted(&mut self, name: &'static str) -> Result<Val> {
        self.next_value().map(|arg| {
            let name = Val::Symbol(self.names.intern(name));
            Val::Cons((name, Val::Cons((arg, Val::Nil).into())).into())
        })
    }

    fn list_tail(&mut self) -> Result<Val> {
        self.skip_whitespace()?;

        match self.lookahead().ok_or(Error::UnexpectedEof)? {
            ')' => {
                self.input.next();
                Ok(Val::Nil)
            },

            _ => {
                let car = self.next_value()?;
                let cdr = self.list_tail()?;
                Ok(Val::Cons((car, cdr).into()))
            },
        }
    }

    fn skip_whitespace(&mut self) -> Result<()> {
        while let Some(next) = self.lookahead() {
            match next {
                ';' => while let Some(comment) = self.input.next() {
                    if comment == '\n' {
                        break;
                    }
                },

                ' ' | '\n' | '\t' => {
                    self.input.next();
                    continue;
                },

                _ => break,
            }
        }

        Ok(())
    }
}
