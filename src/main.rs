extern crate breve;

use breve::*;

fn main() {
    breve::Interpreter::new().and_then(|mut it| {
        loop {
            let input = read()?;

            if input.is_empty() {
                return Ok(());
            }

            eval_print(&mut it, &input)?;
        }
    }).unwrap();
}

fn read() -> Result<String> {
    use std::io::Result;

    let body = || -> Result<String> {
        use std::io::{stdin, stdout, BufRead, Write};

        let mut buf = String::new();

        let () = {
            print!("breve> ");
            stdout().flush()?;
        };

        let stdin = stdin();
        stdin.lock().read_line(&mut buf)?;

        Ok(buf)
    };

    body().map_err(|_| Error::NotAList)
}

fn eval_print(it: &mut Interpreter, input: &str) -> Result<()> {
    let input = it.parse(input)?;
    for val in &input {
        println!("{}", it.show(val.clone())?);
    }

    let mut result = None;

    for expr in input {
        result = Some(it.eval(expr)?);
    }

    if let Some(expr) = result {
        it.show(expr)?;
    }

    Ok(())
}
