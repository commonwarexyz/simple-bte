# A Simple Batched Threshold Encryption Scheme

A Rust implementation of [A Simple Batched Threshold Encryption Scheme](https://eprint.iacr.org/2026/XXX).

## Usage

Run the full benchmark suite with [Criterion](https://github.com/bheisler/criterion.rs):

```bash
cargo bench
```

HTML reports are written to `target/criterion/`.

Run a complete end-to-end example:

```bash
cargo run --release --example e2e
```

Run with `--help` to see all available options.

## Licensing

This repository is dual-licensed under both the Apache 2.0 and MIT licenses. You may choose either license when employing this code.
