# Garnet -- what if Rust was small?

[![builds.sr.ht status](https://builds.sr.ht/~icefox/garnet.svg)](https://builds.sr.ht/~icefox/garnet?)

# The Pitch

People keep talking about what a small Rust would look like, and how
nice it would be, and whether or not Zig or Hare or whatever fits the
bill.  So, I think it's time to start advertising, at least in a small
way: I'm trying to make basically the language these people want, a
language that asks "What would Rust look like if it were small?".  I call
it [Garnet](https://hg.sr.ht/~icefox/garnet). I'm at the point where I
am fairly sure my design is gonna do at least *something* vaguely
useful, but I also think it's time to ask for interested parties to talk
or help.

Garnet strives for smallness by having three basic features: functions,
types, and structs.  Somewhat like Zig, structs double as modules when
evaluated at compile time. A lot like SML/OCaml, structs also double as
your tool for defining interfaces/traits/typeclasses/some kind of way of
reasoning about generics.  If you make sure your code can be evaluated at
compile time, you can treat types mostly like normal values, and
"instantiating a generic" becomes literally just a function that returns
a function or struct.  Again, this is kinda similar to Zig, but I want
to avoid the problem where you don't know whether the code will work
before it's actually instantiated.  Again, this is quite similar to ML
languages, but without an extra layer of awkwardness from having a
separate module languages.  [It seems to work out in
theory](https://people.mpi-sws.org/~rossberg/1ml/).  Whether it can be
made actually convenient... well, I hope so.

I also want to solve various things that irk me about Rust, or at least
make *different* design decisions and see what the result looks like.
Compile times should be fast, writing/porting the compiler should be
easy, ABI should be well-defined, and behavior of low-level code should
be easy to reason about (even if it misses some optimization
opportunities).  The language is intended for low level stuff more than
applications: OS's and systems and embedded programming.  This makes
some Rust features unnecessary, like async/await. Better ergonomics
around borrowing would be nice too, though I'm not sure how to do that
yet, I just hate that there's no way to abstract over ownership and so
we have `Fn` and `FnMut` and `FnOnce`. However, trading some runtime
refcounting/etc for better borrowing ergonomics as suggested in [Notes
on a Smaller Rust](https://without.boats/blog/notes-on-a-smaller-rust/)
and [Swift's
work](https://github.com/apple/swift/blob/01c22b718cfc80a10feaefaf598aa1087f3766c8/docs/OwnershipManifesto.md)
is not on the cards; I think there's a lot of potential for a language
that does this sort of thing, but Garnet is not that language.

Right now the project as a whole is rough around the edges 'cause I've
spent a lot of time going in circles trying to learn how to write a type
checker that works the way I want it to.  I'm still not done (if
anything I feel like I've moved backwards), but I consider the language
itself maybe like 75% decided on.  The main design hole right now is in
fact lifetimes and borrowing; my original plan was to just implement
lexical lifetimes a la Rust 1.0, then sit down and have a good hard
think about that, but I frankly haven't gotten far enough to start
working on that in earnest.

# Code example

```
fn fib(x: I32): I32 =
    if x < 2 then x
    else fib(x-1) + fib(x - 2)
    end
end

-- {} is an empty tuple, "unit"
-- __println() is for now just a compiler builtin.
fn main(): {} =
    __println(fib(40))
end
```

There’s a bunch of small programs in its test suite here: <https://hg.sr.ht/~icefox/garnet/browse/tests/programs?rev=12ee941c3da958f037ba0a9509d0ebc00c6c0465>

And some slightly-more-interesting-but-often-still-hypothetical bits of programs here: <https://hg.sr.ht/~icefox/garnet/browse/gt?rev=12ee941c3da958f037ba0a9509d0ebc00c6c0465>

# Current state

Just to make sure people have some realistic expectations.

 * [✓] = done
 * [⛭] = WIP
 * [?] = Active design concern, probably in flux
 * [ ] = not started

Realistic language goals:

 * [✓] Technically Turing complete
 * [✓] Basic structs/tuples (did a proof of concept, now rewriting it)
 * [✓] Arrays, sum types, other useful basic types
 * [⛭] Generics and type inference
 * [⛭] Full ML-y modules
 * [?] Move semantics, references and borrowing
 * [?] Stdlib of some kind
 * [ ] Pattern matching
 * [ ] Function properties (`const`, `pure`, `noalloc`, etc)
 * [ ] Lots of little ergonomic things

Giant scary tooling goals necessary for Real Use:

 * [⛭] Backend support: C or Rust
 * [ ] Self-host
 * [ ] Basic optimizing backend
 * [ ] Debugger/profiler tooling
 * [ ] Build/packaging system
 * [ ] Language spec
 * [ ] ABI spec
 * [ ] Documentation generator
 * [ ] Semver checker
 * [ ] GOOD backend.  Not sure how to best achieve this.  LLVM is slow,
   QBE left a bad taste in my mouth but might be worth another look.
 * [ ] Backend support: Webassembly
 * [ ] Backend support: Actual CPU's

# Assumptions

Things where you go "it's a modern language, of COURSE it has this".
If it doesn't have something like this, it's a hard error.

 * Unambiguous, context-free syntax
 * Good error messages
 * Cross-compile everywhere
 * Type inference
 * Sum types, no null, all that good jazz

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
 * As little undefined behavior as possible -- If the compiler is
   allowed to assume something can't happen, then the language should
   prevent it from happening if at all feasible.  Let's stop calling it
   "undefined behavior" and call it an `out of context problem`, since
   it's often not undefine*able*, but rather it's something that the
   compiler doesn't have the information to reason about.
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
 * Simple and universal ABI for every platform -- easy for higher level
   stuff to call it, easy for it to call arbitrary stuff.
 * Compiles fast

Another way to think about it is "Garnet wants to be the Lua of system
programming languages".  Small, flexible, made of a few powerful parts
that fit together well, easy to port and implement and toy around with,
reasonably fast.

## Pain points in Rust to think about

 * You can't be generic over mutability and ownership, so for example
   you end up with `iter()`, `into_iter()`, and `iter_mut()`.
 * Related, the pile of AsRef, Deref, Borrow, ToOwned etc. traits.
 * Related, the pile of various things that look kinda like
   references/pointers but aren't, and all the hacks that go into making
   them work.  Example: Box.  Seems fine, right?  Can't pattern match on
   it.  See the `box_pattern` RFC.
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
   processes, etc.  Need better names though.  I do like the idea of
   splitting out specific capabilities into specific parts that may or
   may not be present on all platforms though, instead of having a
   strictly additive model.
 * Syntax inconsistencies/nuisances: Fiddly match blocks, <>'s for
   generics (though the turbofish is wonderful), i32 is both a type and
   a module, -> and => being different is a PITA, you declare values
   with `=` in let statements but `:` in struct constructors,
 * Tail call optimization is not guarenteed -- Drop impl's get in the
   way, but it should be possible to avoid that, or at least make it so
   the compiler gives a warning if it can't guarentee that
 * Lack of construct-on-heap is occasionally kinda awful, though far
   more often totally unnoticable.
 * Rather mediocre support for data type reflection at either compile or
   run time, such as RTTI in general.  Also bites us in trying to make
   C-like enums, separate enum discriminants from enums or vice versa
   (which makes them awkward to compose),
 * Rust's closures are awful.
 * On the note of boilerplate-y stuff, see
   <https://github.com/rustwasm/walrus/blob/121340d3113e0102707b2b07cab3e764cea1ed6b/crates/macro/src/lib.rs>
   for an example of a giant, complex, heavy proc macro that is used
   exactly once to generate a huge amount of entirely uninteresting
   --but nonetheless necessary-- code.  It's good that you can use a
   macro for it, but it's kinda less good that you need to.
 * No function currying is rather a pain sometimes, especially when it's
   really just syntactic sugar for a trivial closure.
 * Rust's trait orphan rules are annoying, but may be too hard to be
   worth trying to solve.
 * Heckin gorram `->` vs `=>` still bleedin' trips me up after five
   years

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

## Functionality we sacrificed for simplicity

 * match blocks on function params, like Erlang -- just syntactic sugar
 * Monomorphized generics -- for now?
 * Cool arbitrary/rational number types -- can be a lib.
 * Though it is tempting, we will NOT do arbitrary-precision integer
   types such as being able to define an integer via an arbitrary range
   such as `[-1, 572)` or arbitrary size such as `i23`.  Maybe later.
 * Like Rust, we don't need to target architectures smaller than 32 bits

## Wishlist items

 * I want to explore having a more orthogonal relationship between
   tuples, structs, and function args.  Enums as well.  See
   <https://todo.sr.ht/~icefox/garnet/7>
 * Better methodology for boilerplate-y things like visitor patterns /
   big ol' honkin' pattern matches would be very interesting.  Some nice
   sugar here may be very productive, personally.  Also
   <https://todo.sr.ht/~icefox/garnet/7>
 * It would be surprisingly appealing if there were no methods, traits,
   or anything like that... just data structures, functions and
   namespaces/modules.  Look at OCaml's modules for inspiration perhaps.
   See <https://todo.sr.ht/~icefox/garnet/8>
 * Consider some sort of templates or macros -- <https://dlang.org/spec/template.html>?
   or just use Handlebars :P.
   Also see <https://todo.sr.ht/~icefox/garnet/8>

## Goals

 * Being effectively finished someday.
 * A compilation model that doesn't necessitate a slow compiler
 * Being able to reason about what kind of code the compiler will
   actually output

## Antigoals

 * Async, promises, other fanciness for nonblocking I/O
 * Ultimate max performance in all circumstances
 * Anything requiring a proof solver as part of the type system

# Toolchain

 * rustc
 * `logos` lexer
 * handrolled parser (recursive descent + Pratt)
 * output Rust, just to make things work.
 * `argh` for command line opts
 * `codespan` for error reporting
 * `lang_tester` for integration tests

Things to consider:

 * rustyline (for repl)
 * `ryu` for parsing floats
 * `tree-sitter` for parsing/formatting tooling (someday?)

## Backend thoughts

Something I need to consider a little is what I want in terms of a
compiler backend, since emitting `x86_64` opcodes myself basically
sounds like the least fun thing ever.

Goals:

 * Not huge
 * Operates pretty fast
 * Outputs pretty good/fast/small code
 * Doesn't require binding to C++ code (pure C may be acceptable)
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

Output Rust for right now, bootstrap the compiler, then think about it.

Trying out QBE and Cranelift both seem reasonable choices, and writing a
not-super-sophisticated backend that outputs many targets seems
semi-reasonable.  Outputting WASM is probably the simplest low-level
thing to get started with, but is a little weird since it is kinda an IR
itself, so to turn an SSA IR into wasm you need a step such as LLVM's
"relooper".  So one might end up with non-optimized WASM that leans on
the implementation's optimizer.

# License

LGPL 3.0 only 