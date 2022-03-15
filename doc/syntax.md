Broad syntax thoughts, yanked from parser

 * Everything is an expression
 * Lua-style preference of keywords rather than punctuation
 * Don't go ham sacrificing familiarity for How It Should Be

So essentially Rust-like syntax, but:

 * Keyword-delimited blocks instead of curly braces
 * and/or/not keywords for logical operators instead of ||, && etc
 * Keep | and & and ~ for bit operations for now
 * Make sure trailing commas are always allowed
 * [] or maybe () instead of <> for generics
 * {} instead of () for tuples, that should keep things unambiguous
 * {} for structs as well, to emphasize relationship with tuples


Deliberate choices so that we don't go ham:

 * we use C/Lua-style function application
   `foo(bar, baz)` instead of Lisp style `(foo bar baz)` or ML/Haskell
   style `foo bar baz`.
 * I kinda want Erlang-style pattern matching in function args, but
   the point of this language is to KISS.
 * Parens for tuples would make life simpler in some ways, but I feel
   also make syntax and parsing more confusing, so let's go with Erlang
   curly braces.  If we use curly braces for structs too, then this also
   emphasizes the equivalence between structs and tuples.
 * Tempting as it is, use `fn foo() ...` for functions, not `let foo = fn() ...`
   or other OCaml-y variations on it.
 * Don't bother letting type decl's or `fn foo()` functions be nested
   inside functions yet.

The result looks quite pleasing, IMO!  And we can mostly use Rust syntax
highlighting.


```
decl =
  | function_decl
  | const_decl

const_decl = DOC_COMMENT "const" ident ":" typename "=" expr
function_decl = DOC_COMMENT "fn" ident [generic_signature] fn_signature "=" exprs "end"

value =
  | NUMBER
  | BOOL
  | UNIT
  | ident

constructor =
  // Tuple constructor
  | "{" [expr {"," expr} [","] "}"

// We may have separators between expressions
exprs = expr [";"] [expr]

expr =
  | let
  | if
  | loop
  | block
  | funcall
  | lambda
  | return
  | constructor
  | binop
  | prefixop
  | postfixop

// Currently, type inference is not a thing
let = "let" ident ":" typename "=" expr
if = "if" expr "then" exprs {"else" "if" expr "then" exprs} ["else" exprs] "end"
loop = "loop" exprs "end"
block = "do" exprs "end"
funcall = expr "(" [expr {"," expr}] ")"
lambda = "fn" fn_signature "=" exprs "end"
return = "return" expr

// Pointer ops should be postfix, leads to much nicer chaining of calls.
// We will use "^" for our deref operator and "&" for our reference
// operator.

generic_signature = "[" [ident {"," ident} [","]] "]"
fn_args = "(" [ident ":" typename {"," ident ":" typename}] ")"
fn_signature = fn_args [":" typename]

typename =
  | "I32"
  | "Bool"
  | "fn" "(" [typename {"," typename} [","]] ")" [":" typename]
  // Tuples with curly braces like Erlang seem less ambiguous than the more traditional parens...
  // I hope that will let us get away without semicolons.
  | "{" [typename {"," typename}] [","] "}"
  // Fixed-size arrays
  // | "[" typename ";" INTEGER} "]"
  // TODO: Generics?
  // | ID "[" typename {"," typename} [","] "]"
  // slices can just then be slice[...]
  // Borrows
  | typename "&"
  // Mutable borrows
  | typename "&mut"
  // Raw pointers
  | typename "*const"
  | typename "*mut"

// Things to add, roughly in order
// Break and return
// while/for loops
// arrays/slices
// enums
// structs (anonymous and otherwise?)
// assignments
// generics
// references

```



Ok I gotta ramble about types and syntax now.  Let's take a full look at
how Rust does it:

```
// Alias names
type Foo = Bar;

// Define a struct with a fixed name that is its own type
struct Foo {
}

// Define an enum with a fixed name that is its own type
enum Bar {
    Bar1,
}

// Newtype struct: A kinda weird mix-up that
struct Foo(i32);
```


Maybe something like:

```
// Alias names
alias Foo = {i32, f32}
alias Bar = struct
    x: i32
end

let foo: Foo = {3, 4.5}
// We need an anonymous struct constructor now, so I guess this is it!
let bar: Bar = ${ x = 12 }

// Non-interchangable names
type Bop = {i32, f32}
type Beep = struct
    x: i32
end

let bop: Bop = Bop {3, 4.5}
let beep: Beep = Beep ${ x = 12 }

Or, could constructors actually just be functions?
If we have named arguments in functions, then, that could work.
```


Universal function call syntax is `foo:bar(baz)` which simply parses into
the same thing as `bar(foo, baz)`.  This means it is not ambiguous wiht
a struct `foo` with a field `bar` that happnens to be callable; that
would be `foo.bar(baz)`.  The exact rule is that `COLON` is a postfix
operator, so it parses `expr COLON IDENT funargs`.  It's an ident
instead of an arbitrary expression because that gets the job done,
prevents `x : y (z)` becoming some kind of ternary operator thing, and
it means we can't do horrible things like `foo:fn(x) = println(x) end()`

## Functions

```
fn foo(x: I32, y: F32): Rettype =
    ...body...
end
```

Ok, what about generic functions?

```
fn foo[T](x: T, y: F32): T =
    ...body...
end
```

Call them like this:

```
foo(x,y)

-- or UFCS
x:foo(y)

-- Or, perhaps, to disambiguate a generic type a la a turbofish?
foo[T](x, y)
x:foo[T](y)
```

Not sure if that is too ambiguous with some kind of general type
constructor syntax, or whether it is absolutely perfect.  We will see!
Not worrying about the turbofish right now.  (Since it's [] instead of
<>, maybe call it the mechafish?)

# Data

## Structs

Ok, so declaring a struct type goes like this:

```
alias Foo = struct {
    x: Bar,
    y: I16,
    z: I64,
}
```

This declares a newtype around an anonymous struct type.  This will be
structurally typed a la OCaml, not nominally typed.

To actually create an instance of this struct, you can do:

```
let x: Foo = struct {
    x = some_bar,
    y = 12,
    z = 91,
}
```

Then for a nominally typed struct, you just use `type` instead of
`alias` to declare a newtype.

Right, now for struct types with generics:

```
alias Foo[T] = struct[T] {
    a: T,
    b: I32,
}
```

I dislike the double-declaration of `[T]` there, it's one of the irksome
things about Rust.  Can we just do this?

```
alias Foo = struct[T] {
    a: T,
    b: I32,
}
```

Mmmmmaybe.  Let's aim for it and see what happens.

Creating said struct then:

```
let x: Foo[I32] = struct {
    x = 12I32
    y = 12,
    z = 91,
}
```

## Arrays

```
I32[4]
I32[]
I32 []
```

## References & lifetimes

```
-- Shared references
I32 ^
-- lifetime names... no fukkin clue
I32^(foo)
I32 ^ 'foo
I32 ^'foo

-- Unique references
I32 ^uniq
```

## Pointers

```
I32 *const
I32 *mut
```
