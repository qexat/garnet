//! A very basic benchmark for the compiler.
//! Does not test the speed of the *generated* code,
//! generates how fast *the compiler* runs.  Does not
//! actually compile the output code with rustc.

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use garnet::*;

/// Each iteration creates about 32 sLOC
fn gen_dumb_test_code(count: usize) -> String {
    let mut buf = String::with_capacity(2048 * count);
    for i in 0..count {
        buf += &format!(
            r#"
fn rand{i}(x I32) I32 =
    32767 * x * 17 + 4000100 - 50
end

fn collatz{i}(x I32) I32 =
    if x % 2 == 0 then
        collatz{i}(x / 2)
    else
        collatz{i}(x * 3 + 1)
    end
end

fn squared{i}(x I32) I32 =
    let y I32 = x
    let z I32 = x
    y * z
end


fn foo{i}(x I32) I32 =
    squared{i}(x) + squared{i}(x)
end

fn bar{i}(x I32) I32 =
    rand{i}(collatz{i}(x))
end

fn baz{i}(x I32, y I32, z I32) I32 =
    let a I32 = bar{i}(x * y)
    let b I32 = rand{i}(z)
    let c I32 = foo{i}(y)
    a + b + c
end
        "#,
            i = i
        );
    }
    buf += "fn main() {} =\n";
    for i in 0..count {
        buf += &format!(
            r#"
        let quux{i} I32 = 10002
        let xyzzy{i} I32 = 26391
        let quuz{i} I32 = foo{i}(quux{i})
        baz{i}(quux{i}, xyzzy{i}, quuz{i});
        "#,
            i = i
        );
    }
    buf += "    {}\nend\n";
    buf
}

fn bench_with_rust_backend(c: &mut Criterion) {
    let code = gen_dumb_test_code(103);
    let lines = code.lines().count();
    let name = format!("compile {}ish lines to Rust", lines);
    c.bench_function(&name, |b| {
        b.iter(|| compile("criterion.gt", black_box(&code), backend::Backend::Rust))
    });

    let code = gen_dumb_test_code(103 * 8);
    let lines = code.lines().count();
    let name = format!("compile {}ish lines to Rust", lines);
    c.bench_function(&name, |b| {
        b.iter(|| compile("criterion.gt", black_box(&code), backend::Backend::Rust))
    });

    /*
    let code = gen_dumb_test_code(103 * 16);
    let lines = code.lines().count();
    let name = format!("compile {}ish lines", lines);
    c.bench_function(&name, |b| {
        b.iter(|| {
            garnet::compile(
                "criterion.gt",
                black_box(&code),
                garnet::backend::Backend::Rust,
            )
        })
    });
    */
}

fn bench_with_null_backend(c: &mut Criterion) {
    let code = gen_dumb_test_code(103);
    let lines = code.lines().count();
    let name = format!("compile {}ish lines to nothing", lines);
    c.bench_function(&name, |b| {
        b.iter(|| compile("criterion.gt", black_box(&code), backend::Backend::Null))
    });

    let code = gen_dumb_test_code(103 * 8);
    let lines = code.lines().count();
    let name = format!("compile {}ish lines to nothing", lines);
    c.bench_function(&name, |b| {
        b.iter(|| compile("criterion.gt", black_box(&code), backend::Backend::Null))
    });

    let code = gen_dumb_test_code(103 * 16);
    let lines = code.lines().count();
    let name = format!("compile {}ish lines to nothing", lines);
    c.bench_function(&name, |b| {
        b.iter(|| compile("criterion.gt", black_box(&code), backend::Backend::Rust))
    });
}

fn bench_stages(c: &mut Criterion) {
    let code = gen_dumb_test_code(103 * 8);
    let lines = code.lines().count();
    let name = format!("Parse {} lines", lines);
    c.bench_function(&name, |b| {
        b.iter(|| {
            let mut parser = parser::Parser::new("criterion.gt", black_box(&code));
            parser.parse()
        })
    });

    let mut parser = parser::Parser::new("criterion.gt", black_box(&code));
    let ast = parser.parse();

    c.bench_function("lower and run passes", |b| {
        b.iter(|| {
            let hir = hir::lower(black_box(&ast));
            passes::run_passes(hir)
        })
    });

    let hir = hir::lower(black_box(&ast));
    let hir = passes::run_passes(hir);

    c.bench_function("typecheck and borrowcheck", |b| {
        b.iter(|| {
            let tck = &mut typeck::typecheck(black_box(&hir)).unwrap();
            borrowck::borrowck(&hir, tck).unwrap();
        })
    });

    let tck = &mut typeck::typecheck(black_box(&hir)).unwrap();
    borrowck::borrowck(&hir, tck).unwrap();

    c.bench_function("typechecked passes", |b| {
        b.iter(|| {
            passes::run_typechecked_passes(black_box(hir.clone()), black_box(tck));
        })
    });

    let hir = passes::run_typechecked_passes(hir, tck);
    c.bench_function("codegen", |b| {
        b.iter(|| backend::output(backend::Backend::Rust, black_box(&hir), black_box(tck)))
    });
}

criterion_group!(
    benches,
    //bench_with_rust_backend,
    //bench_with_null_backend,
    bench_stages
);
criterion_main!(benches);
