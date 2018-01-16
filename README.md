# Breve

A sketch of a Lisp dialect. Nothing interesting or original. Cons cells and environments are refcounted, not actually garbage collected. Forms are bytecode-compiled before evaluation. Native functions can be used in all the same places as interpreted functions.

Closure over lexical environments:

```
(def foo (n)
  (fn () n))

(call (foo 5)) ; => 5
```

Separate namespaces for functions and values:

```
(def fac (n)
  (apply #'* (iota n)))

(map #'fac (iota 10)) ; => (1 2 6 24 120 720 5040 40320 362880 3628800)
```

Locally scoped functions introduced with `label`:

```
(def fib (n)
  (label iter (i a b)
    (if (> i 1)
      (iter (- i 1) (+ a b) a)
      a))
  (iter n 1 0))

(map #'fib (iota 10)) ; => (1 1 2 3 5 8 13 21 34 55)
```

Destructuring let-bindings:

```
(def last (xs)
  (let (car & cdr) xs) ; Cons syntax uses & instead of .
  (if cdr
    (last cdr)
    car))

(last '(1 2 3 4 5)) ; => 5
```

Strings, and text IO, are not yet implemented.

I have no plans to finish this.
