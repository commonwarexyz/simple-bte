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

## Packed Simple BTE (Remark 2)

`bte::packed` applies Remark 2 of *DKG Is All You Need*. In the ramp setting `t >= f + 2*ell - 1`,
a batch is split into `U = ceil(B / ell)` sub-batches. Parties hold a packed sharing of
`tau, ..., tau^U` instead of Shamir sharings of `tau, ..., tau^B`, and each of the `ell` slots is
opened as a Simple BTE instance of batch size `U`. Public parameters and per-party state shrink by
`ell`. Verifying a partial decryption costs `U + 1` pairings instead of `B + 1`. Partial decryptions
are still a single G1 element.

```bash
cargo bench --bench packed_bench   # n = 128, f = 42, t = 85, ell in {16, 22}; ~15 minutes
```

Results (n = 128, t = 85, single-threaded, Apple M5 Pro). Original numbers are from the unchanged
code. "After quorum" is combine + finalize; predecrypt depends only on the ciphertexts and can
run while partial decryptions are still arriving.

| B | variant | combine | predecrypt | finalize | after quorum | total | verify (1 share) |
|---|---|---|---|---|---|---|---|
| 32 | original | 1.3 ms | 112 ms | 17.3 ms | 18.7 ms | 131 ms | 4.4 ms |
| 32 | packed, ell = 16 | 23.7 ms | 62.6 ms | 17.4 ms | 41.1 ms | 104 ms | 1.7 ms |
| 128 | original | 1.3 ms | 557 ms | 68.7 ms | 70.1 ms | 627 ms | 15.3 ms |
| 128 | packed, ell = 16 | 24.1 ms | 341 ms | 70.2 ms | 94.2 ms | 436 ms | 5.6 ms |
| 512 | original | 1.4 ms | 2.68 s | 273 ms | 275 ms | 2.95 s | 59.3 ms |
| 512 | packed, ell = 16 | 23.6 ms | 1.78 s | 271 ms | 295 ms | 2.08 s | 21.0 ms |
| 2048 | original | 1.4 ms | 12.41 s | 1.09 s | 1.09 s | 13.50 s | 237 ms |
| 2048 | packed, ell = 16 | 23.7 ms | 8.95 s | 1.09 s | 1.12 s | 10.07 s | 81.5 ms |

Each slot's predecrypt pads to an FFT of size `next_pow2(2U)`, so for power-of-two `B` choose `ell`
as a power of two. With `ell = 22`, `U = 94` pads to 256 and predecrypt at `B = 2048` takes 12.1 s,
barely better than the original.

## Pairing-free BTE

[`pairing-free/`](pairing-free) implements the pairing-free batched threshold encryption scheme
(Appendix A of *DKG Is All You Need*) over Pallas, with a Combine + Dec benchmark against this crate.
See its README for results.

## Licensing

This repository is dual-licensed under both the Apache 2.0 and MIT licenses. You may choose either license when employing this code.
