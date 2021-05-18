I've been reading about 1ML so let's try to formalize the type system in
some way or another.

# Design guidelines

 * Needs to be quick to compile -- so no undecidable things
 * I don't want this to be a research language, so keep Shiny New Ideas
   to a minimum
 * Needs to stay simple, so highly orthogonal ideas are good.
 * On the other hand, if a special case or a bit of RTTI can do the job
   of a very complex formal type system for common cases, do it.
 * When in doubt, do what Rust does.

# Specific goals

Nominal types instead of structural.  Because I like it when types are
real, instead of just pretending they're real.

I want to be able to have modules and structs be basically
interchangable, 'cause...  modules are just compile-time singleton
structs.  Zig does something like this.

I'm not sure I want to have explicit compile-time evaluation, though it
might end up being there anyway.

I also want nice enums for numerical values.

I do *not* want dependent types, because cool as they are, it seems like
they add a lot of complexity.  A limited form of dependent types for
variable-length arrays may be acceptable.

I *really* want to be able to be polymorphic over owned types, shared
borrows, and unique borrows.  Unfortunately nobody seems to know how the
hell to do this.

TODO: We haven't even talked about borrowing and move semantics yet,
heck.

# Numbers

There are multiple number types with different domains.  `U8`, `I8`,
etc, increasing by powers of two up to 64 or 128 bits.  Numbers are not
interchangable without casts, there is no automatic conversion.

Manual conversion needs to be precise.  Rust is doing a little bit of
flailing in the process of discovering this: they have an `x as Y`
expression that just does C-style casts, which are a pain in the ass,
and then `From` and `TryFrom` traits don't necessarily say enough about
what they're actually doing..  We can be more rigorous about it, so let
us examine the cases we want to cast between, considering changing type
`A` to `B`:

 * `A` is completely contained within `B`.  `U8` -> `U16`.  This is
   easy, we just do the conversion.
 * `A` is not completely contained within `B`.  `I8` -> `U8`, `U16` ->
   `I8`.  It may be bounded at one side or the other, or both.  In this
   case, we may want to round out-of-bounds values towards `B`'s
   endpoints, or we may want to just extract the low bits/do a bit cast.

For signed and unsigned integers this is pretty straightforward, but
when you add floats to the mix you get more complicated cases:

 * `A` may still be completely contained within `B`, for example `U8` ->
   `F32`.
 * `A` cannot be completely contained within `B`, but the values beyond
   its bounds can be represented with some loss of fidelity.  `I32` ->
   `F32`
 * `A` cannot be completely contained within `B`, it's a float going to
   an integer that needs a rounding strategy.  Is this conceptually the
   same as the case above?  I was thinking "beyond the domain" vs.
   "within the domain but not representable" (ie, 5.5 in an `I32`) would
   be different cases, but maybe they're not.

Like Rust, we also have `USize`, which is the size of an array index,
and `ISize`, which is the size of a pointer offset.  TODO: Maybe call
them `Size` and `Offset` or something?  hm, idk.  Maybe we shouldn't get
fancy.

TODO: I was thinking of NOT having arbitrary-sized numbers (ie, `I8 =
Number(-128,127)`), buuuuuuut....  it might be a useful case.  Not going
to worry about it now.  If we do, don't be too scared of doing runtime
checks for bounded numbers and such, and trusting the backend to
optimize them out.

# Typedef's and aliases

A typedef creates a new nominal type that is not exchangable with any
other type, and contains a single member.  This is similar to Rust's
"typedef struct"'s, but while Rust's approach of having it be a
single-element tuple is kinda nice, in reality it's just a slightly
weird construct to learn.  So maybe it will be nicer to have a specific
operators for constructing and deconstructing values of this type.

TODO: Figure out whether this is actually true, and figure out syntax
for reals.  Right now the compiler just generates functions for
wrapping/unwrapping typedefs, which is a little silly.  If we can make
those functions generic then maybe that's fine though.

Aliases create a new name for a type.  They should work as well as a
textual search-and-replace would.

# Enumerations

There are two intertwined kinds of enumerations: enums and sum types.
TODO: Better names would be welcome, if there are any to be had.

Enums are what Rust people tend to call "C-like enums": they are a set
of integers with no duplicates, where each integer gets a constant name
attached to it.  By default they're consecutive positive integers
starting from 0, but the programmer may assign arbitrary values to them.
Because of these they are always an integer number type under the hood,
and TODO we need to define what the default is; `U32`?  `Usize`?  TODO
we also need to be able to specify what the type is, a la Rust's `repr`
declaration.

There's also some common operations on enums that we would like to
support, via auto-generated functions or whatever: iterate over them,
tell whether or not an integer is a valid member of the set (ie,
do a checked conversion), to convert them to and from strings, and to be
able to combine them into bitmasks.  Because we have nominal types
everywhere, the values of an enum may not be treated as an integer, they
need an explicit cast.

Sums are real tagged unions, like Rust's enum types.  There's a few
differences: Each member is its own type, and can be separated from the
discriminant.

```rust
-- The $ syntax denotes a struct instead of a tuple; TODO: This is
-- far from final
sum Foo =
    Bar { I32, I32 },
    Bop $ { x: F32, y: F32 },
end

-- These are the same type
let x: Foo = Foo.Bar { 3, 4 }
let y: Foo = Foo.Bop { x: 3.0, y: 4.0 }

-- These are not the same type
let x: Foo.Bar = Foo.Bar { 3, 4 }
let y: Foo.Bop = Foo.Bop { x: 3.0, 4.0 }

-- So you could do:
fn something(t: Foo.Bar): {} =
    ...
end

-- What we *don't* do is the Typescript style thing of being
-- able to build ad-hoc types out of other enums, you can't
-- do `fn something(t: Foo.Bar | Foo.Bop) = ... end`.  That
-- sort of thing turns into dynamic/duck typing fast, or
-- else becomes really hard to reason about.
```

So, because of this we can have functions to automatically go from
type+enum to a sum and back.

TODO: WIP, this is kinda a cartoon of how I imagine it working.

```rust
sum Foo =
    Bar { I32, I32 },
    Bop $ { x: F32, y: F32 },
end

-- The sum type creates another type as well, which looks like this:
enum Foo.Discriminant =
    Bar,
    Bop,
end

-- This lets you make strongly-typed constructors and deconstructors...
fn bar_from(f: {I32, I32}: Foo.Bar = ... end
-- TODO: Anonymous struct; is that a good idea?
-- We don't have anonymous structs anywhere else, so...
fn bop_from(f: ${x: I32, y: I32}: Foo.Bop = ... end

fn foo_from_bar(f: Foo.Bar): Foo = ... end
fn foo_from_bop(f: Foo.Bop): Foo = ... end

fn bar_from_foo(f: Foo): Option[Foo.Bar] = ... end
fn bop_from_foo(f: Foo): Option[Foo.Bop] = ... end

-- Is the given value a Foo.Bar?  We can make this generic over any sum
fn is(f: Foo, d: Foo.Discriminant): Bool = ... end

-- TODO: I think this is safe?  Other way around, maybe not.
fn foo_to_raw(f: Foo): {Foo.Discriminant, [U8;size_of_foo]} = ... end
```

C-like unions might be a thing in the future, but are currently not
useful or interesting, so we don't have them.

# Structs

This gets surprisingly complicated.

TODO: As shown above, I'm not sure whether or not anonymous structs are
a thing we want to have.  If so, then you might be able to have fairly
beefy metaprogramming by being able to test whether two different
structs are byte-compatible and be able to put together and take apart
structs and enums from pieces while proving that they submit to certain
invariants.  That might not *actually* be very useful though.  Why is
duck typing so tempting?

I DO want to be able to go from struct to tuple with the same members
and back nicely.  But again that may or may not actually be very useful,
and it means that structs suddenly have a fixed ordering, which

What if we don't have traits or modules, but just structs acting as
namespaces?  This gets us kinda into a 1ML-ish type system, which I've
never been great at thinking about, but let's try to work out the
implications:

 * Structs contain type declarations
 * Type declarations in a struct may be generic, ie refer to types
   passed into the struct
 * structs/modules have instantiations where you declare concrete types
   for any generic types in them(?) -- I think you kinda end up with
   explicit type constructors being called where you say `type Thing =
   Vector<I32>;`
 * Interfaces/traits are just structs, which may have multiple versions
   created of them.  This is where the good shit happens.
 * Vtables/trait objects just become pointers to structs.  But that only
   works if they are concrete structs, with no generics that have not
   been made concrete.
 * Structs can contain nested structs inside them and that's our module
   program hierarchy.
 * Do we want a "flatten" operation to say "this struct has all the
   fields of that other struct, just stuffed inline into it"?  That's a
   way people tend to do things like inheritance-like struct extension.
   On the other hand, Rust don't need no inheritance, so fuckit.

```rust
-- To experiment with this, let's define Rust's std::ops::Add trait
struct Add[Lhs, Rhs] =
    type Output
    pub add: fn(self: Lhs, rhs: Rhs): Output
end

-- implement for I32
-- We know all these values at compile time, so this just gets
-- inlined away hopefully?  Can we make a comptime attribute or such
-- so it's an error if it doesn't?
const AddI32: Add[I32, I32] = Self {
    output = type I32,
    add = fn(self: I32, rhs: I32): I32 = ... end
}

-- Call it
let x: I32 = 3
AddI32.add(x, 4)

-- Ooooor, we can use postfix syntax of some kind...
-- How the heck do we disambiguate it from our path syntax?
x:AddI32.add(4)

-- Import the definition I suppose
use AddI32.add
x:add(4)
```

```rust
-- Trying to give this idea a workout.  Can we implement some handy Rust
-- traits?
--
-- TODO: Think about the Self type
-- TODO: Think about references
-- TODO: Think about generic syntax
-- TODO: We can make default methods pretty easy if we just have default
-- values for structs in general

-- Hmm, think about Self
struct Hasher =
    type Self
    pub finish: fn(self: Self&): U64
    pub write_u8: fn(self: Self mut&, i: U8): {}
    pub write_i8: fn(self: Self mut&, i: U8): {}
    ...
end

-- TODO:
-- The type constraints here are wacky, how does that work?
-- In Rust it's this:
-- pub trait Hash {
--    pub fn hash<H>(&self, state: &mut H)
--    where H: Hasher;
-- }

struct Hash =
    pub hash: fn[H](self: Self&, state: H mut&, impl: Hasher&): {}
    -- Or like this???
    pub hash: fn[H](self: Self&, hasher: Hasher&, state: hasher.Self): {}
end

-- What if we define the implementation via a functor or something?
fn impl_hash(t: Type, hasher: Hasher): Hash =
    Hash {
        hash = fn(self: t&, state: hasher.Self): {}
    }
end
```


# Arrays and slices

Mostly, do what Rust does.  Can we manage const generics, at least for
array types?  I don't know.

I do want slices to be less "magical", Rust has a lot of syntactic sugar
around slices-to-arrays, references, references-to-arrays,
references-to-Vec, iterators, etc. and it would be nice to just toss it
all out and see how it looks raw and unadorned.

```rust
let x: I32[3] = [1, 2, 3]
let y: Slice['a, I32] = x.as_slice()

```

This raises the question of how to write something like Rust's
`Box<[i32]>`.  `[i32]` is an unsized type, and it *seems* like this
results in Rust sneakily stashing away the length of the value with the
pointer to it.  This is the cause of some of the Magic I want to avoid,
a `Box<i32>` and a `Box<[i32]>` are different, so `Box` actually has two
different representations.  Same with `&i32` vs `&[i32]`.

So, does this mean that the representation of `Box` is really

```rust
struct Box<T, some_const_value> {
    contents: *T,
    size = some_const_value,
}
```

and the representation gets optimized down when `some_const_value` is
known at compile-time?  Hmmmmm.  Two `Box<[i32]>`'s are the same type
even when the `[i32]`'s are different lengths, so...

Dang it, the Rust people are ahead of me once again: https://doc.rust-lang.org/nightly/std/ptr/trait.Pointee.html
and https://doc.rust-lang.org/nightly/std/ptr/struct.DynMetadata.html

# Characters and strings

Do what Rust does.  No need to innovate here.  String slices, like array
slices, being non-magical sounds nice though.

# References and pointers

References are called "shared" and "unique", not "mutable" and
"immutable".  Otherwise they should work like Rust.  A postfix notation
for reference and dereference operators sounds interesting and potentially
smoother to write, let's try it and see how it works.  I also kinda
hate `&` and `*`, let's use `^` for "pointer type" and "make reference".  What
do we use for "dereference"?  Maybe `*` anyway.  Pascal uses `^` for
"pointer type" and "dereference", which Feels Right but is backwards
from what Rust and C do.

Heck I need to think of a lifetime syntax.  Oh well, let's experiment
with something like this:

```rust
let x: I32 = 3
let y: I32^ = x -- lifetime is inferred
let z: I32 'a^ = x -- Lifetime explicit,

-- this keeps you reading the type left to right
-- So I guess let's just keep rolling with
```

# Functions and closures

fn, Fn, FnMut, FnOnce, sigh... can we do anything about that?

I want to be able to annotate functions with properties they have.
Pure, noalloc, can-be-evaluated-at-compile-time, no-panic, etc.

If we do the Zig thing and explicitly pass allocator objects around
everywhere, then "noalloc" just kinda vanishes.

# Generics

Heck

I kinda want semi-monomorphized generics, where a generic function will
have monomorphized variants generated for a variety of small sizes, and
then one polymorphic fallback that just takes a pointer and maybe some
type metadata such as a vtable.  Swift does something like this for its
DLL calls, and calls the type metadata a "witness table" which is a
horrible name.
