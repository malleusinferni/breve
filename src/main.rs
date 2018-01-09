extern crate breve;

use breve::*;

fn main() {
    let mut it = breve::Interpreter::new().unwrap();

    if let Some(path) = std::env::args().nth(1) {
        let source = readfile(&path).unwrap();

        eval_print(&mut it, &source).unwrap_or_else(|err| {
            println!("Error: {}", err);
        });

        return;
    }

    loop {
        let line = readline().unwrap();

        if line.is_empty() {
            return;
        }

        eval_print(&mut it, &line).unwrap_or_else(|err| {
            println!("Error: {}", err);
        });
    }
}

use std::io::{self, stdin, stdout, BufRead, Read, Write};

fn readline() -> io::Result<String> {
    let mut buf = String::new();

    let () = {
        print!("breve> ");
        stdout().flush()?;
    };

    let stdin = stdin();
    stdin.lock().read_line(&mut buf)?;

    Ok(buf)
}

fn readfile(path: &str) -> io::Result<String> {
    use std::fs::File;

    let mut buf = String::new();
    File::open(path)?.read_to_string(&mut buf)?;

    Ok(buf)
}

fn eval_print(it: &mut Interpreter, input: &str) -> Result<()> {
    let input = it.parse(input)?;

    //for val in &input {
    //    println!("INPUT: {}", it.show(val.clone())?);
    //}

    for expr in input {
        let result = it.eval(expr)?;
        println!("{}", it.show(result)?);
    }

    Ok(())
}
