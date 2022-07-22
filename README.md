# Garnet -- what if Rust was small?

[![builds.sr.ht status](https://builds.sr.ht/~icefox/garnet.svg)](https://builds.sr.ht/~icefox/garnet?)

# The Pitch

Poeple keep talking about what a small Rust would look like, and how
nice it would be, and whether or not Zig or Hare or whatever fits the
bill.  So, I think it's time to start advertising, at least in a small
way: I'm trying to make basically the language these people want, a
language that asks "What would Rust look like if it were small?"  I call
it [Garnet](https://hg.sr.ht/~icefox/garnet). I'm at the point where I
am fairly sure my design is gonna do at least *something* vaguely
useful, but I also think it's time to ask for interested parties to talk
or help.

Garnet strives for smallness by having three basic features: functions,
types, and structs.  Somewhat like Zig, structs double as modules when
evaluated at compile time. A lot like SML/OCaml, structs also double as
your tool for defining interfaces/traits/typeclasses/some kind of way of
reasoning like generics.  If you make sure your code can be evaluated at
compile time, you can treat types mostly like normal values, and
"instantiating a generic" becomes literally just a function that returns
a function or struct.  Again, this is kinda similar to Zig, but I want
to avoid the problem where you don't know whether the code will work
before it's actually instantiated.  Again, this is quite similar to ML
languages, but without an extra layer of awkwardness from having a
separate module languages.  [It seems to work out in
theory](https://people.mpi-sws.org/~rossberg/1ml/).

I also want to solve various things that irk me about Rust, or at least
make *different* design decisions and see what the result looks like.
Compile times should be fast, writing/porting the compiler should be
easy, ABI should be well-defined, and behavior of low-level code should
be easy to reason about (even if it misses some optimization
opportunities).  The language is intended for low level stuff more than
applications, OS's and systems and embedded programming, so async/await
is unnecessary.  Better ergonomics around borrowing would be nice too,
though I'm not sure how to do that yet, I just hate that there's no way
to abstract over ownership and so we have `Fn` and `FnMut` and `FnOnce`.
However, trading some runtime refcounting/etc for better borrowing
ergonomics as suggested in [Notes on a Smaller
Rust](https://without.boats/blog/notes-on-a-smaller-rust/) and [Swift's
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
 * custom parser (recursive descent + Pratt)
 * output Rust, just to make things work.
 * `argh` for command line opts
 * `codespan` for error reporting

Things to consider:

 * rustyline (for repl)
 * lasso or `string-interner` (for string interning)
 * `ryu` for parsing floats

Programs-as-separate-files tests:

 * Roll our own, prolly not that hard
 * `test-generator` generates Rust code and recompiles if files are
   changed, demonstrated here:
   https://github.com/devsnek/scratchc/blob/main/tests/out.rs
 * `goldentests` crate
 * http://insta.rs/

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

Output Rust for right now, bootstrap the compiler, then think about it.

Trying out QBE and Cranelift both seem reasonable choices, and writing a
not-super-sophisticated backend that outputs many targets seems
semi-reasonable.  Outputting WASM is probably the simplest low-level
thing to get started with, but is a little weird since it is kinda an IR
itself, so to turn an SSA IR into wasm you need a step such as LLVM's
"relooper".  So one might end up with non-optimized WASM that leans on
the implementation's optimizer.

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
 * <https://thume.ca/2019/07/14/a-tour-of-metaprogramming-models-for-generics/>
 * <https://boats.gitlab.io/blog/post/notes-on-a-smaller-rust/>
 * <https://without.boats/blog/revisiting-a-smaller-rust/>
 * <https://dl.acm.org/doi/pdf/10.1145/3428195> (on type inference)
 * <https://queue.acm.org/detail.cfm?id=3212479> (C is not a low level language; the good one)
 * <https://old.reddit.com/r/rust/comments/my3ipa/if_you_could_redesign_rust_from_scratch_today/>
 * <https://lobste.rs/s/j7zv69/if_you_could_re_design_rust_from_scratch>
 * <https://people.mpi-sws.org/~rossberg/1ml/> -- 1ML programming language
 * <http://moscova.inria.fr/~maranget/papers/ml05e-maranget.pdf> -- Compiling pattern matching
 * <https://research.swtch.com/gomm> and related posts in introduction -- Memory models
 * <https://arxiv.org/abs/1306.6032> -- Complete and Easy Bidirectional Typechecking for Higher-Rank Polymorphism
 * <https://github.com/nikomatsakis/bidir-type-infer> -- Implementation of the above paper
 * <https://belkadan.com/blog/2021/08/Swift-Regret-Tuples-and-Argument-Lists/?tag=swift-regrets> -- Swift design retrospective, on tuples and argument lists.  Garnet may have fewer problems with this than Swift does, but Garnet still has some of the problems mentioned here.
 * <https://arxiv.org/abs/1512.01895> -- Modular Implicits, stapling
   typeclasses onto an ML module system.

References on IR stuff:

 * <https://github.com/bytecodealliance/wasmtime/blob/main/cranelift/docs/ir.md>
 * <https://blog.rust-lang.org/2016/04/19/MIR.html>
 * <https://llvm.org/docs/LangRef.html>

References on backend stuff:

 * <http://troubles.md/posts/wasm-is-not-a-stack-machine/> and the
   following posts -- Interesting thoughts on webassembly's design

References on borrow checking:

 * <http://smallcultfollowing.com/babysteps/blog/2018/04/27/an-alias-based-formulation-of-the-borrow-checker/>
 * <https://github.com/rust-lang/polonius>
 * <https://gankra.github.io/blah/deinitialize-me-maybe/> -- Destroy All Values: Designing Deinitialization in Programming Languages

Technically-irrelelvant but cool papers:

 * <https://www.mathematik.uni-marburg.de/~rendel/rendel10invertible.pdf> -- Invertable parsers

What not to do:

 * <https://arxiv.org/abs/2201.07845> -- How ISO C became unusable for operating systems development -- I'm a little dubious of the simplicity of its arguments, but it contains lots of references

# Random notes

## CI

 * Actual build takes ~1-2 minutes
 * Adding end-to-end unit tests it takes 5 minutes
 * Adding code coverage it takes ~15 minutes -- cargo-tarpaulin ain't
   instant but most of it is still spent in building it rather than
   running it.
 * Making a `.deb` package for cargo-tarpaulin would help a lot then.
   Talked to the Debian Rust packaging team and they're in favor, very
   helpful folks, but of course understaffed.

## Out Of Context Problems

nee "Undefined Behavior"

Some more thoughts on the lack of undefined behavior is... you COULD
define "read an invalid pointer" to be "return unknown value, or crash
the program".  But only if you knew that pointer could never aim at
memory-mapped I/O.  *Writing* to an invalid pointer could literally do
anything in terms of corrupting program state.  Some slightly-heated
discussion with `devsnek` breaks the problem down into two parts: For
example, WASM does not have undefined behavior.  If you look at a
computer from the point of view of assembly language + OS, it MOSTLY
lacks undefined behavior, though some things like data races still can
result in it.  If you smash a stack in assembly you can look at it and
define what *is* going to happen.  But from the point of view of the
assumptions made by a higher-level language, especially one free to
tinker with the ABI a little, there's no way you can define what will
happen when a stack gets smashed.  And even if you're writing in
assembly on a microcontroller then you might still be able to do things
that put the hardware in an inconsistent state by poking memory-mapped
I/O.  So, Undefined Behavior usually isn't really Undefined, rather it's
defined by a system out of the scope of the language definition.  So
let's just stop calling it Undefined Behavior and call it an `out of
context problem`.

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
   compiler isn't good enough at avoiding it itself.  A good
   middle-ground might be some setup where in debug mode you can have
   runtime checks for reading uninitialized data before it's been
   written.
 * Order of evaluation of function arguments.
 * Constructing an impossible value, such as having an enum representing
   integer values 0-5 and stuffing the integer 6 into its slot.  Not
   sure *how* to deal with this, but it's a case that needs handling.

Here's a list of things that I don't see a way of defining in any
reasonable/performant way:

 * **Writing** an undefined pointer may do anything, ie by smashing the
   stack.  If correctly executing a program requires assuming an
   un-smashed stack, well, that's tricky.
 * Reading a pointer pointing to mmapped I/O may similarly do anything.
   A notable recent experience was when reading a value from memory was
   *required* to acknowledge an interrupt.

Todo list of other common sources of UB in C, from <https://stackoverflow.com/questions/367633/what-are-all-the-common-undefined-behaviours-that-a-c-programmer-should-know-a>:

 * Converting pointers to objects of incompatible types
 * Left-shifting values by a negative amount (right shifts by negative
   amounts are implementation defined)
 * Evaluating an expression that is not mathematically defined (ie, div
   by 0)
 * Casting a numeric value into a value that can't be represented by the
   target type (either directly or via `static_cast`)
 * Attempting to modify a string literal or any other const object
   during its lifetime
 * A pile of other things that are mostly C++'s fault

Reading material: <https://blog.regehr.org/archives/213> and the follow-on
articles.  Also, apparently Zig has opt-*in* undefined behavior for
things like integer overflow, shift checking, etc which sounds pretty
hot.  Another option is to provide compiler-intrinsic escape hatches,
similar to Rust's various `unchecked_foo()` functions.

Some of the optimizations that pointer UB enables are talked about here:
<https://plv.mpi-sws.org/rustbelt/stacked-borrows/paper.pdf>  Would be
very interesting to have some emperical numbers about how much UB helps
optimizations.

Again, note that we are NOT trying to make incorrect programs do
something that could be considered correct, and we are NOT trying to
define things that are inherently undefinable (such as writing to a
truly random/unknown pointer).  Sooner or later, defining what is
defined is up to the *programmer*, and we are trying to make it so that
there are as few rules and hidden gotchas as possible for the
*programmer* to handle when dealing with these things.

Other reading material from people doing interesting unsafe Rust things:
<https://github.com/TimelyDataflow/abomonation/issues/32>

Mandating a different pointer type for referring to mmapped I/O is
probably not unreasonable, tbqh, and removes a source of semantic
weirdness that compilers have problems dealing with.  Basically have a
`MMIOPtr[T]` type that is kinda similar to Rust's `UnsafeCell<T>`,
but doesn't allow reads/writes to be reordered or optimized away, and
which has different constraints on what values are allowed for it than a
normal pointer does.

## On pointers/references

```
Evy
A downside of your #[offset] design is that you can create a lot of nonsense and (partial) overlaps all of which you have to check for.  May i suggest a different approach instead? Where you have nominally ordered struct layouts, have a field type padding. Padding is a parameterised type that can take either an alignment/offset, or a size. padding(size: 4) inserts 4 bytes of padding, padding(align: 16) rounds the offset of the next field up to the next multiple of 16 bytes, padding(offset: 0x100) rounds up to offset 256, may be zero-sized, and will error out of the fields before the padding exceed that offset. And padding cannot be used in types which's fields aren't nominally ordered.
￼
icefox — Today at 12:03 PM
Oooo
￼
Evy
So e.g.:
type PlicLayout = #[ordered] ${
  priority:             [U32; 1024]               padding(align:     0x1000),
  pending:              [U32;   32]               padding(align:     0x1000),
  enable_for_context:  [[U32;   32]      ; 15872] padding(offset: 0x20_0000),
  threshold_and_claim: [ThresholdAndClaim; 15872] padding(align:     0x1000),
}

You can even make align and size paddings available for arbitrary fields or variables. Like:
let mut x: Atomic(U32) align(cache) padding(align: cache) = 0;

let mut raw_u32_le: [u8; 4] align(4) = …;
￼
icefox — Today at 12:06 PM
ooooOOOOOooooOOOOOooo
Downside is I'd have to figure out how all those options interact
￼
Evy
In this example, the type of priority is actually ([U32; 1024], [Garbage; PaddingSize]), except that you can only safely touch the .0 payload. In this case, PaddingSize = 0. For enable_for_context it's however much padding is needed to round up to offset 0x20_0000, etc.  Now, two funny things happen now:

1. You could simply demand that types with padded fields will automagically be nominally ordered. Then you can throw the redundant #[ordered] away. If you didn't, well… let's just say that ordering padded fields optimally is not trivial, significantly more complex than a vanilla order by align, size desc.

2. For things like fast zero-copy serialisation, you can demand padding to have specific values instead of »undefined/uninit«. For example by annotating: x: T padding(align: N, fill: 0x00)
3. To create a paddingless C-like struct, all that's needed is to append padding(offset: 0) to the first field.
￼
icefox — Today at 12:16 PM
1) If you need to have things ordered properly in memory then its up to you to define the ordering. That's fine.
￼
Evy
Btw.: [Atomic(U32) align(cache); N] align(page)
Nicely flexible.
￼
Evy
This raises the question of how to write something like Rust's
Box<[i32]>.
This one is tricky. Technically there are two competing pointer types: Single element pointers and pointers to arrays. The latter ones are fat pointers with a length attached. And actually you get even more pointer types. Like pointers to &dyn T things, where it's data-pointer and vtable-pointer packed together. Or function pointers, which on some platforms are magical.

If you have a more powerful type system than Rust 1.0 had, you wouldn't store a contents: *T, but instead a contents: impl Pointer, no second field needed.
Nice, when did Rust finally get these traits?
I feel like reading TWiR is useless, given how often awesome stuff went below my radar.
￼
repnop, Resident RISC-V Shill — Today at 12:35 PM
Yeah it's definitely a toss up
￼
Evy
#![feature(ptr_metadata)]

assert_eq!(std::ptr::metadata("foo"), 3_usize);

I could've used that a century ago! D:
[12:39 PM]
# Characters and strings

Do what Rust does.  No need to innovate here.
@icefox Do what Swift does. :<
￼
icefox — Today at 12:42 PM
What does Swift do?
￼
@Evy
I feel like reading TWiR is useless, given how often awesome stuff went below my radar.
￼
icefox — Today at 12:43 PM
The problem is what is useful varies by person, and often starts small and slow
But yes
￼
Evy
@icefox
# References and pointers
Having looked plenty at proglangs like Herb Sutter's simplified C++ or Ada or in parts C#, i came to have the opinion that this is the wrong approach when designing proglangs. What seems to be a better design is having in, out, in out, move, forward function args. If you think you explicitly need pointers or references, use generic library types like Ptr(T) or Ref(T). Even mutability wrappers like Ptr(Box(T)) (Lisp naming style.) or Ref(Mut(T)) (Rust naming style.). In all other scenarios, it's better to let the language and calling conventions decide whether to use copies or references or what. For example, for tiny args the proglang may decide to turn an in out T into pass-by-value-and-return-by-value behind the scenes. Plus, these annotations allow for way better static checking whether you actually used your function args accordingly to what you specified.
￼
Evy
> @icefox
> What does Swift do?  ￼
It doesn't just have UTF-8 slice plus Unicode Scalar Values, but also has built-in Grapheme Cluster support.(edited)
￼
icefox — Today at 12:48 PM
You still need raw pointers sometimes, but that's an interesting idea. Do you have any references for the "simplified C++"?
￼
Evy
Swift is the proglang with the currently best Unicode handling out there. »Best« being defined by how much spec is covered by the letter.
￼
icefox — Today at 12:48 PM
The flip side of that is, how do you specify those things in struct fields?
￼
@icefox
You still need raw pointers sometimes, but that's an interesting idea. Do you have any references for the "simplified C++"?
￼
Evy
Lemme try find the paper.
https://github.com/hsutter/708
￼
￼
@icefox
The flip side of that is, how do you specify those things in struct fields?
￼
Evy
What things?
Pointers?
Same thing, wrapper types.
￼
icefox — Today at 12:52 PM
Like, how do you say &mut Foo or &Foo in that in/out/etc paradigm
Especially mutability
￼
Evy
The sexy thing about in, out,… is that it gives you a lot of »dark magic« and other patterns for free and safely. in T automagically picks whether to use memcpy(T) or &T or &mut T behind the scenes. You you never have to overload functions or traits based on referenceness. And out T gives you safe access to uninitialised buffers, because the compiler can guarantee you never read before you write. forward is »in, until the last use, which is move«, etc.
￼
@icefox
Like, how do you say &mut Foo or &Foo in that in/out/etc paradigm
￼
Evy
&mut T etc. are impl-details.

If you take an argument in and don't modify it, use in T. If you take it in and observably modify it, you take in out T. If you only return something but never read what you got in (e.g. uninitialised buffers) you use out T. If you take ownership, you use move T. If you are allowed to inspect the data without modifying and then pass along however the caller intended (whether to move or borrow or whatever), you use forward T.

The compiler is free to decide whether or not to use pointers or pass by register or what not. You are 100% oblivious to whether pointers or copies are given.
￼
icefox — Today at 12:57 PM
No no
￼
Evy
So if you e.g. must have a pointer, e.g. to map a physical address to a virtual one, you take e.g. virt: in Pointer(T).
￼
icefox — Today at 12:57 PM
Not for functions
For structs
Oh, the wrapper type
￼
@icefox
For structs
￼
Evy
Vec = struct(T: type) {
  ptr: Ptr(Mut(T)),
  len: USize,
  cap: USize,
}
￼
Evy
Btw., a mutability wrapper would allow for some crazy code patterns. For example, you only need to impl iterators once, and they'll handily allow you to use the same generic iterator type to iterate over T, Ref(Mut(T)),  or just Ref(T). And it gives you dark magic interior mutability patterns like having a field Mut(Bool) for storing a dirty bit. And it does what Rust wants, getting rid of static mut by instead having a static wrapper with interior mutability.
I/O pointers
Ptr(T, volatile: true) or something-something. Maybe even Ptr(Volatile(T)). Where  stand-aloneVolatile(T) does its thing depending on in out etc. (i.e. standalone is useless)(edited)
fn, Fn, FnMut, FnOnce, sigh... can we do anything about that?
The problem is that upvalues, aka. closure captures, are magic hidden payloads, directly contradicting the explicit self design. If you want to get rid of the army of function traits, you have to make captures explicit. Then your »onceness« or »mutness« etc. emerge from whether your captures are in, in out, etc.
If any of your captures is out or in out, you got a FnMut. Got a move, it may be a FnOnce. Etc.
Ada goes a step further by distinguishing between procedures and functions, the latter being allowed to have magic, the former not. The former can even only return stuff via out parameters.
￼
@icefox
You still need raw pointers sometimes, but that's an interesting idea. Do you have any references for the "simplified C++"?
￼
Phlopsi — Today at 2:36 PM
only because we use programming languages, that are too close to the hardware.
￼
icefox — Today at 3:17 PM
Sometimes you need to be close to the hardware.
I wait with bated breath for someone to write a UART driver in Haskell.
￼
@Phlopsi
only because we use programming languages, that are too close to the hardware.  @Evy I agree with you on all the in out stuff. I still don't get the practical usefulness about forward, but I'm tired, so that's fine. ￼
￼
Evy
Imagine things like debug-logging your stuff you pass right through. Your debug logger just says: fn debug(T: type)(x: forward T) -> forward T. If it came in as move, it comes out as move. If it came in as in out, it comes out as in out, etc. But within debug it is treated as an in.
￼
Evy
@icefox
Sometimes you need to be close to the hardware.  ￼
Ideally only when you're smarter than the compiler/language and making the compiler/language smarter than you is too hard of a problem to solve.

C++ is the perfect example for why e.g. in out is better in the 99% case than spelling out refs vs. moves vs. mutable refs. For an utterly extreme example, look at the amount of junk you need to program to implement a correct forwarding function using current C++. Further more, look at the amount of overload code bloat needed to fine-tune your code for all possible kinds of function args and returns. And that complexity grows exponentially with every added argument or return value. Having in out reduces hundreds of lines of »perfect« C++ code to what looks like a pretty naïve single function.

And i'm frequently wondering whether there are also better ways to abstract over data sharing and memory-mapped structures. I have not yet found the answer.
'cause like in function args, pointers only give you a »how«, not a »what«. And we apply patches and bandages to things like »volatile« and magical linker scripts etc. to add a »what« to the implemented »how«.
￼
icefox — Today at 3:39 PM
Sure, and Rust performs the exact same kind of abstraction of arg representation you're talking about
￼
Evy
Rust is like C++ just a »how«.
The only thing Rust has over C++ is that you can state what ref input a ref output belongs to.
Which is a biggie, no downplaying that.
￼
icefox — Today at 3:49 PM
Not really, functions have no defined ABI
The compiler can and does turn references into copies and vice versa
And, there's always going to be times when I'm smarter than the language. The lang is an amplifier for my own reasoning capabilities, and a special purpose amplifier is going to give me a lot more leverage (for lower cost in its problem domain) than one that tries to be equally good at everything.
￼
Evy
Very limitedly. Copies to refs is demanded by ABI if your struct size exceeds two registers, or if you have too many args. The reverse is a magical recent optimisation of LLVM. But it gives you none of the semantics and semantic guarantees. You still have to overload traits and shit for ref vs. move, you still cannot distinguish between out and in out with a &mut T (it's always treated as in out), etc.
￼
icefox — Today at 3:55 PM
Yeah that part I'm definitely interested in
```

## On Motivation

A quote from Graydon, original creator of Rust, from
<https://github.com/graydon/rust-prehistory>:

While reading this -- if you're foolish enough to try -- keep in mind
that I was balanced between near-total disbelief that it would ever come
to anything and miniscule hope that if I kept at experiments and
fiddling long enough, maybe I could do a thing.

I had been criticizing, picking apart, ranting about other languages for
years, and making doodles and marginalia notes about how to do one "right"
or "differently" to myself for almost as long.  This lineage representes
the very gradual coaxing-into-belief that I could actually make
something that runs

As such, there are long periods of nothing, lots of revisions of
position, long periods of just making notes, arguing with myself,
several false starts, digressions into minutiae that seem completely
absurd from today's vantage point (don't get me started on how long I
spent learning x86 mod r/m bytes and PE import table structures, why?)
and self-important frippery.

The significant thing here is that I had to get to the point of
convincing myself that there was something *there* before bothering to
show anyone; the uptick in work in mid-to-late 2009 is when Mozilla
started funding me on the clock to work on it, but it's significant that
there were years and years of just puttering around in circles, the kind
of snowball-rolling that's necessary to go from nothing to "well...
maybe..."

I'd encourage reading it in this light: Delusional dreams very gradually
coming into focus, not any sort of grand plan being executed.
