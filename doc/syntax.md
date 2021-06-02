Broad syntax thoughts, yanked from parser

 * Everything is an expression
 * Go for Lua-style preference of keywords rather than punctuation
 * But don't go ham sacrificing familiarity for How It Should Be

So essentially Rust-like syntax, but:

 * Keyword-delimited blocks instead of curly braces
 * and/or/not keywords for logical operators instead of ||, && etc
 * Keep | and & and ~ for binary operations
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

The result looks quite pleasing, IMO!  And we can mostly use Rust syntax
highlighting.


```
decl =
  | function_decl
  | const_decl

const_decl = DOC_COMMENT "const" ident ":" typename "=" expr
function_decl = DOC_COMMENT "fn" ident fn_signature "=" exprs "end"

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
