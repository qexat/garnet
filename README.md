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
   spectrum.  Investigate more.
 * Fast compiler -- This is a pain point for Rust for various reasons,
   and one of those things where having it work well is real nice.
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
 * Some more thoughts on the lack of undefined behavior is... you COULD
   define "read an invalid pointer" to be "return unknown value, or
   crash the program".  But only if you knew that pointer could never
   aim at memory-mapped I/O.  *Writing* to an invalid pointer could
   literally do anything in terms of corrupting program state.  Some
   slightly-heated discussion with `devsnek` breaks the problem down
   into two parts: For example, WASM does not have undefined behavior.
   If you look at a computer from the point of view of assembly
   language + OS, it MOSTLY lacks undefined behavior, though some things
   like data races still can result in it.  If you smash a stack in
   assembly you can look at it and define what *is* going to happen.
   But from the point of view of the assumptions made by a higher-level
   language, especially one free to tinker with the ABI a little,
   there's no way you can define what will happen when a stack gets
   smashed.  And even if you're writing in assembly on a microcontroller
   then you might still end up doing Undefined things by poking
   memory-mapped I/O.  So, Undefined Behavior isn't really Undefined,
   rather it's defined by a system out of the scope of the language
   definition.  So let's just stop calling it Undefined Behavior and
   call it an `out of context problem`.
 * I am not CONVINCED that a linker is the best way to handle things.
   This has implications on things like distributing libraries, defining
   ABI's, using DLL's, and parallelizing the compiler itself.  No solid
   thoughts here yet, but it is an area worth thinking about.  Rust, C,
   Go and Swift present different points in this area to look at.

## C's advantages that I want to have

 * Easy to port to new systems
 * Easy to use on embedded systems
 * Easy to control code size bloat
 * Easy to get a (partial) mental model, which is low-level enough to
   teach you a lot
 * Simple and universal ABI -- easy for higher level stuff to call it
 * Compiles fast

## Pain points in Rust to think about

 * You can't be generic over mutability and ownership, so for example
   you end up with `iter()`, `into_iter()`, and `iter_mut()`.
 * Related, the pile of AsRef, Deref, Borrow, ToOwned etc. traits.
 * Rust's hacky generic-ness over length of sequences/tuples is pretty lame
 * The slightly-magical relationship between `String` and `&str`, and `&[]`
   and `[]` and `[T;N]`, is a little distressing
 * Magical `AsRef` and `Deref` behavior is a little distressing
 * `std` vs `core` vs `alloc` -- it'd be better if `std` didn't actually
   re-export `core`, because then more programs could be `no_std`
   implicitly.  `alloc` is kinda a red-headed stepchild in this
   hierarchy; Zig's approach of explicit allocator objects everywhere
   may or may not be superior.  Talk to some of the stdlib or embedded
   people about how they'd *want* to arrange it if they could; papering
   over weird platforms like wasm is a known annoyance.
   Maybe something like `core` for pure computational things, `sys` for
   platform-specific low-level stuff like threading and timekeeping
   primitives that may appear in a microcontroller or low-level VM
   without a full OS, then `os` or something for stuff like filesystems,
   processes, etc.  Need better names though.
 * Syntax inconsistencies/nuisances: Fiddly match blocks, <>'s for
   generics (though the turbofish is wonderful), i32 is both a type and
   a module, -> and => being different is a PITA, you declare values
   with `=` in let statements but `:` in struct constructors,
 * Tail call optimization is not guarenteed -- Drop impl's get in the
   way, but it should be possible to avoid that, or at least make it so
   the compiler gives a warning if it can't guarentee that
 * Lack of construct-on-heap is occasionally kinda awful.
 * Rather mediocre support for data type reflection at either compile or
   run time, such as RTTI in general.  Also bites us in trying to make
   C-like enums, separate enum discriminants from enums or vice versa
   (which makes them awkward to compose),
 * Rust's closures are awful.

## Glory points in Rust to exploit or even enhance

 * Move semantics everywhere
 * Derive traits
 * methods <-> functions
 * True, if conservative, constexpr's
 * Iterators just return Option
 * Math is checked by default
 * Stack unwinding without recovery -- very nice compromise of
   complexity
 * UTF-8 everywhere
 * Lack of magical constructors

## Functionality sacrificed for simplicity

 * match blocks on function params, like Erlang -- just syntactic sugar
 * Monomorphized generics -- for now?
 * Cool arbitrary/rational number types -- can be a lib.

# Toolchain

 * rustc
 * custom parser (recursive descent + Pratt)
 * output wasm with `walrus`

Things to consider:

 * structopt (for arg parsing)
 * rustyline (for repl)
 * codespan (for error reporting)
 * logos (for lexer)
 * lasso or `string-interner` (for string interning)

## Backend thoughts

Something I need to consider a little is what I want in terms of a
compiler backend, since emitting `x86_64` opcodes myself basically
sounds like the least fun thing ever.

Goals:

 * Not huge
 * Operates pretty fast
 * Outputs pretty good/fast/small code
 * Doesn't require binding to C/C++ code
 * Produces `x86_64`, ideally also Aarch64 and WASM, SPIR-V would be a
   nice bonus

Non-goals:

 * Makes best code evar
 * Super cool innovative research project
 * Supports every platform evar, or anything less than 32-bits (it'd be
   cool, but it's not a goal)

Options:

 * Write our own -- ideal choice in the long run, worst choice in the
   short run
 * LLVM -- Fails at "not huge", "operates fast" and "doesn't require C++
   bindings"
 * Cranelift -- Might actually be a good choice, but word on the street
   (as of early 2020) is it's poorly documented and unstable.
   Investigate more.
 * QBE -- Fails at "doesn't require C bindings", but initially looks
   good for everything else.  Its Aarch64 unit tests have some failures
   though, and it doesn't output wasm.  Probably my top pick currently.
 * WASM -- Just output straight-up WASM and use wasmtime to run it.
   Cool idea in the short term, WASM is easy to output and doesn't need
   us to optimize it much in theory, and would work well enough to let
   us bootstrap the compiler if we want to.  Much easier to output than
   raw asm, there's good libraries to output it, and I know how to do it.
 * C -- Just output C Code. The traditional solution, complicates build
   process, but will work.
 * Rust -- Rust compiles slow but that's the only downside, complicates
   build process, but will work.  Might be useful if we can proof
   whatever borrow checking type stuff we implement against Rust's

Current thoughts: Try out QBE and Cranelift, then if those don't work out
either output Rust or WASM.

# License

MIT

# References

 * <https://rustc-dev-guide.rust-lang.org/>
 * <https://digitalmars.com/articles/b90.html>
 * <https://c9x.me/compile/> (and the related <https://myrlang.org/> )
 * <https://gankra.github.io/blah/swift-abi/> (and related,
   <https://swift.org/blog/library-evolution/>)
 * <https://en.wikipedia.org/wiki/Operator-precedence_parser>
 * <https://play.rust-lang.org/?version=stable&mode=debug&edition=2018&gist=174ca95a8b938168764846e97d5e9a2c>
 * <https://www.evanmiller.org/statistical-shortcomings-in-standard-math-libraries.html>

# Random notes

## CI

 * Actual build takes ~1-2 minutes
 * Adding end-to-end unit tests it takes 5 minutes
 * Adding code coverage it takes ~15 minutes -- cargo-tarpaulin ain't instant but most of it is still spent in building it.
 * Making a `.deb` package for cargo-tarpaulin would help a lot then.

## Out Of Context Problems

nee "Undefined Behavior"

Things that I think we CAN define context for:

 * Integer overflow either overflows or panics.
 * **Constructing** an undefined pointer just is a number in a register.
 * **Reading** an undefined pointer, in the absence of memmapped I/O,
   either gives a random result, panics, or causes the host system to
   produce an error (ie by segfaulting your program)
 * We **may** have well-defined pointers that are never valid,
   ie, null pointer.  Reading and writing to these can do whatever we
   feel like.  We should probably make them either panic or cause the
   host system to produce an error.
 * Reading uninitialized data should be a compile-time error.  Manually
   eliding initialization for performance reasons just means your
   compiler isn't good enough at avoiding it itself.
 * Order of evaluation of function arguments.

Here's a list of things that I don't see a way of defining in any
reasonable/performant way:

 * **Writing** an undefined pointer may do anything, ie by smashing the
   stack.  If correctly executing a program requires assuming an
   un-smashed stack, well, that's tricky.

Todo list of other common sources of UB in C, from <https://stackoverflow.com/questions/367633/what-are-all-the-common-undefined-behaviours-that-a-c-programmer-should-know-a>:

 * Converting pointers to objects of incompatible types
 * Left-shifting values by a negative amount (right shifts by negative amounts are implementation defined)
 * Evaluating an expression that is not mathematically defined (ie, div
   by 0)
 * Evaluating an expression that is not mathematically defined
 * Casting a numeric value into a value that can't be represented by the target type (either directly or via `static_cast`)
 * Attempting to modify a string literal or any other const object during its lifetime
 * A pile of other things that are mostly C++'s fault

Reading material: <https://blog.regehr.org/archives/213> and the follow-on
articles.  Also, apparently Zig has opt-*in* undefined behavior for
things like integer overflow, shift checking, etc which sounds pretty
hot.  Another option is to provide compiler-intrinsic escape hatches,
similar to Rust's various `unchecked_foo()` functions.
