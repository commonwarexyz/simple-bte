# A Simple Batched Threshold Encryption Scheme

## Usage

```bash
cargo bench
```

HTML reports are written to `target/criterion/`.

Run a complete end-to-end example:

```bash
cargo run --release --example e2e
```

Run with `--help` to see all available options.

## Pairing-free BTE

[`pairing-free/`](pairing-free) implements the pairing-free batched threshold encryption scheme
(Appendix A of *DKG Is All You Need*) over Pallas, with a Combine + Dec benchmark against this crate.
See its README for results.

## Licensing

This repository is dual-licensed under both the Apache 2.0 and MIT licenses. You may choose either license when employing this code.
