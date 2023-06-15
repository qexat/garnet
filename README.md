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
useful, so I'm making it more or less public.

Garnet strives for smallness by having three basic abstractions: functions,
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


# Code example

```
fn fib(x I32) I32 =
    if x < 2 then x
    else fib(x-1) + fib(x - 2)
    end
end

-- {} is an empty tuple, "unit"
-- __println() is for now just a compiler builtin.
fn main() {} =
    __println(fib(40))
end
```
 
There’s a bunch of small programs in its test suite here: <https://hg.sr.ht/~icefox/garnet/browse/tests/programs>
And some slightly-more-interesting-but-often-still-hypothetical bits of programs here: <https://hg.sr.ht/~icefox/garnet/browse/gt>
Syntax is *mostly* fixed but there's plenty of details that can still be a WIP.

# Current state

As of spring 2023 he typechecker seems to more or less work but some backend work needs to happen before certain code using generics can be actually compiled.  So while I work on that there's a bunch of other things that can be worked on more or less outside of that critical path.
 
Just to make sure people have some realistic expectations.

 * [✓] = done
 * [⛭] = WIP
 * [?] = Active design concern, probably in flux
 * [ ] = not started

Realistic language goals:

 * [✓] Technically Turing complete
 * [✓] Basic structs/tuples (did a proof of concept, now rewriting it)
 * [✓] Arrays, sum types, other useful basic types
 * [✓] Generics and type inference
 * [⛭] Full ML-y modules
 * [⛭] Packages for multiple files and separate compilation
 * [?] Move semantics, references and borrowing
 * [?] Stdlib of some kind
 * [?] Function properties (`const`, `pure`, `noalloc`, etc)
 * [ ] Pattern matching
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
 * [ ] GOOD backend.
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

These are the goals I'm thinking about where we might need to need to make
explicit design tradeoffs to make something that is possible to implement
effectively. These are essentially directions to explore rather than hard-and-
fast rules, and may change with time.

 * Simplicity over runtime performance -- Rust and Go are very different
   places on this spectrum, but I think OCaml demonstrates you should be
   able to have a bunch of both.  There needs to be more points on this
   spectrum.  There haven't been many innovative new low-level languages
   designed since the 1980's.
 * Fast compiler -- This is a pain point for Rust for various reasons,
   and one of those things where having it work well is real nice.
 * Simplicity of compiler -- I'd rather have a GOOD compiler in 50k
   lines than a FANTASTIC compiler in 500k lines
 * I feel like these two things together should combine to (eventually)
   make compiler-as-library more of a thing, which seems like an
   overlooked field of study.  It can be useful to aid JIT,
   metaprogramming, powerful dynamic linking, etc.  It seems very silly
   that this remains Dark Magic outside of anything that isn't Lisp or
   Erlang.  (That said, when you don't want this, you REALLY don't want
   it.)  Haven't done much work on this yet, but it should be far easier
   than Rust makes it.
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
 * Everything interoperates with it via FFI

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
   with `=` in let statements but `:` in struct constructors, piddly 
   stuff like that.
 * Tail call optimization is not guarenteed -- Drop impl's get in the
   way, but it should be possible to avoid that, or at least make it so
   the compiler gives a warning if it can't guarentee TCO.
 * Lack of construct-on-heap is occasionally kinda awful, though far
   more often totally unnoticable.
 * Rather mediocre support for data type reflection at either compile or
   run time, such as RTTI in general.  Also bites us in trying to make
   C-like enums, separate enum discriminants from enums or vice versa
   (which makes them awkward to compose),
 * Rust's closures are awful when you try to use them like you would
   use closures in other functional languages.  Passing them down the
   stack usually works ok, but passing them up the stack or stuffing 
   them into structures is a giant `Box`-y PITA.
 * On the note of boilerplate-y stuff, see
   <https://github.com/rustwasm/walrus/blob/121340d3113e0102707b2b07cab3e764cea1ed6b/crates/macro/src/lib.rs>
   for an example of a giant, complex, heavy proc macro that is used
   exactly once to generate a huge amount of entirely uninteresting
   --but nonetheless necessary-- code.  It's good that you can use a
   macro for it, but it's kinda less good that you need to.
 * No function currying is rather a pain sometimes, especially when it's
   really just syntactic sugar for a trivial closure.
 * Rust's trait orphan rules are annoying.  Getting around them with
   modules solves that problem but brings up new ones, so we will 
   just have to try it out and see at scale whether the new ones are 
   worse.
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

 * Async, promises, other fanciness for nonblocking I/O or stack juggling
 * Ultimate max performance in all circumstances
 * Anything requiring a proof solver as part of the type system
 * Anything that is an active research project

# Toolchain

 * rustc
 * `logos` lexer
 * handrolled parser (recursive descent + Pratt)
 * Backend outputs Rust, just to make things work.
 * `argh` for command line opts
 * `codespan` for error reporting
 * `lang_tester` for integration tests

Things to consider:

 * rustyline (for repl)
 * `ryu` for parsing floats
 * `tree-sitter` for parsing/formatting tooling (someday?)


# License

LGPL 3.0 only 