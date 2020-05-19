# Garnet

[![builds.sr.ht status](https://builds.sr.ht/~icefox/garnet.svg)](https://builds.sr.ht/~icefox/garnet?)

An experiment in a systems programming language along the lines of Rust
but more minimal.  Where Rust is a C++ that doesn't suck, it'd be nice
for this to be a C that doesn't suck.

Loosely based on <https://wiki.alopex.li/BetterThanC>, far more loosely
based on <https://wiki.alopex.li/GarnetLanguage> which is an older set
of concepts and has a different compiler entirely.

# Assumptions

Things where you go "it's a modern language, of COURSE it has this".
If it doesn't have something like this, it's a hard error.

 * Unambiguous, context-free syntax
 * Good error messages
 * Cross-compile everywhere
 * Type inference (eventually)

# Runtime/language model goals

Things where you might need to need to make explicit design tradeoffs.
It concerns the overlap of design and implementation.  These are
essentially directions explore rather than hard-and-fast rules, and may
change with time.

 * Simplicity over runtime performance -- Rust and Go are very different
   places on this spectrum, but I think OCaml demonstrates you should be
   able to have a bunch of both.  There needs to be more points on this
   spectrum.
 * Fast compiler -- This is a pain point for Rust for various reasons,
   and one of those things where having it work well is real nice
 * Simplicity of compiler -- I'd rather have a GOOD compiler in 50k
   lines than a FANTASTIC compiler in 500k lines; investigate
   [qbe](https://c9x.me/compile/) for example.
 * I feel like these two things together should combine to (eventually)
   make compiler-as-library more of a thing, which seems like an
   overlooked field of study.  It can be useful to aid JIT,
   metaprogramming, powerful dynamic linking, etc.  It seems very silly
   that this remains Dark Magic outside of anything that isn't Lisp or
   Erlang.  (That said, when you don't want this, you REALLY don't want
   it.)
 * No undefined behavior -- This may be hard to do, but it would be
   really nice to eliminate this scourge from existence, or at least
   demonstrate that it can be eliminated in a reasonable way.
 * I am not CONVINCED that a linker is the best way to handle things.
   This has implications on things like distributing libraries, defining
   ABI's, using DLL's, and parallelizing the compiler itself.  No solid
   thoughts here yet, but it is an area worth thinking about.


# Toolchain

 * rustc
 * custom parser (recursive descent)
 * cranelift (tentatively)

Things to consider:

 * structopt (for arg parsing)
 * rustyline (for repl)
 * codespan (for error reporting)
 * lasso or `string-interner` (for string interning)

# License

MIT

# References

 * <https://rustc-dev-guide.rust-lang.org/>
 * <https://digitalmars.com/articles/b90.html>
 * <https://c9x.me/compile/> (and the related <https://myrlang.org/> )
 * <https://gankra.github.io/blah/swift-abi/> (and related,
   <https://swift.org/blog/library-evolution/>)
 * <https://en.wikipedia.org/wiki/Operator-precedence_parser>
