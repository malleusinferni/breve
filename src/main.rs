fn main() {
    let mut it = breve::Interpreter::new().unwrap();

    if let Some(path) = std::env::args().nth(1) {
        let source = readfile(&path).unwrap();

        it.parse(&source).and_then(|input| {
            it.eval_and_print(input.into_iter())
        }).unwrap_or_else(|err| {
            println!("Error: {}", err);
        });

        return;
    }

    let mut editor = rustyline::Editor::<()>::new();

    loop {
        use rustyline::error::ReadlineError;

        match editor.readline("breve> ") {
            Err(ReadlineError::Eof) => {
                return;
            },

            Err(err) => {
                println!("Error: {:?}", err);
                return;
            },

            Ok(line) => it.parse(&line).and_then(|input| {
                editor.add_history_entry(&line);
                it.eval_and_print(input.into_iter())
            }).unwrap_or_else(|err| {
                println!("Error: {}", err);
            }),
        }
    }
}

use std::io::{self, Read};

fn readfile(path: &str) -> io::Result<String> {
    use std::fs::File;

    let mut buf = String::new();
    File::open(path)?.read_to_string(&mut buf)?;

    Ok(buf)
}
