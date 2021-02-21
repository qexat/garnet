Broad syntax thoughts, yanked from parser

Everything is an expression
Go for Lua-style preference of keywords rather than punctuation
But don't go ham sacrificing familiarity for How It Should Be

Keyword-delimited blocks instead of curly braces
and/or/not keywords for logical operators instead of ||, && etc
Keep | and & and ~ for binary operations
Make sure trailing commas are always allowed


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


```
decl =
  | function_decl
  | const_decl

const_decl = DOC_COMMENT "const" ident ":" typename "=" expr
function_decl = DOC_COMMENT "fn" ident fn_signature "=" {expr} "end"

value =
  | NUMBER
  | BOOL
  | UNIT
  | ident

constructor =
  // Tuple constructor
  | "{" [expr {"," expr} [","] "}"

expr =
  | let
  | if
  | loop
  | block
  | funcall
  | lambda
  | constructor
  | binop
  | prefixop
  | postfixop

// Currently, type inference is not a thing
let = "let" ident ":" typename "=" expr
if = "if" expr "then" {expr} {"else" "if" expr "then" {expr}} ["else" {expr}] "end"
loop = "loop" {expr} "end"
block = "do" {expr} "end"
funcall = expr "(" [expr {"," expr}] ")"
lambda = "fn" fn_signature "=" {expr} "end"

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
