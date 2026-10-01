# Pairing-free batched threshold encryption

Implementation of the pairing-free BTE scheme from Appendix A of *DKG Is All You Need*
over the Pallas curve, with roots-of-unity domains, plus a Combine + Dec comparison against
Simple BTE (the parent crate, BLS12-381).

Only the optimistic path is implemented: partial decryptions are assumed correct, so there
are no verification keys, DLEQ proofs, or `PartVerify`.

## Construction

* **Domains.** Party `k` sits at `ω^k ∈ H_N` (`N = next_pow2(n)`). Sub-batch slot `s` sits at
  `c·ν^s` in the coset `c·H_m` (`m = next_pow2(ℓ)`, only the first `ℓ` points are used). The offset
  `c` is chosen with `c^m = h` a small integer, so reducing a polynomial modulo `X^m − h` costs
  only a few doublings.
* **Setup** (dealer stand-in for the DKG): `S(X) = sk + Z_D(X)·T(X)`, `deg T ≤ f − 1`, so `S` has
  degree `≤ f + ℓ − 1` and equals `sk` on the `ℓ` slots. Party `k` gets `S(ω^k)`.
* **Enc**: ElGamal `(g^r, pk^r·M)` with a Fiat-Shamir Schnorr proof of `r`, bound to `c2`.
* **PartDec**: one `ℓ`-term MSM per sub-batch, `g^{sk_k·R(ω^k)}`. The public Lagrange
  coefficients are pre-multiplied by `sk_k`.
* **Combine** recovers `P = S·R` (degree `< t`) in the exponent and evaluates it on the slots.
  Two implementations, which return identical results:
  * `combine_lagrange`: barycentric weights on roots of unity (`w_j = x_j·Z̄(x_j)/N`), then one
    MSM of size `|T|` per ciphertext.
  * `combine_fft`: `Q = P·Z̄` (`Z̄` vanishes on the absent parties) has degree `< N`. Combine
    runs one inverse group FFT of size `N`, folds modulo `X^m − h`, applies `m − 1` coset
    twiddles, then a group FFT of size `m`, then one scaling per slot. If every party in `H_N`
    responds, `Z̄ = 1` and the erasure multiplications disappear.
  * `combine` (hybrid, the default): FFT for sub-batches with at least 16 occupied slots,
    and MSMs for a short final sub-batch.

  Group FFT twiddles and per-quorum constants are pre-decomposed for GLV: width-4 wNAF on both
  halves, an endomorphism table, and odd-multiple tables batch-normalized to affine per FFT stage.
  This is about 2× faster than arkworks' default scalar multiplication.
* **Dec**: `M_i = c2_i − pd_i`.

## Running

```bash
cd pairing-free
cargo test --release
cargo bench --bench curves    # Pallas vs Vesta primitives
cargo bench --bench compare   # Combine + Dec vs Simple BTE, n = 128 (several minutes)
SKIP_SBTE=1 cargo bench --bench compare -- pf/   # pairing-free side only
cargo run --release --example profile             # GLV / group FFT micro-timings
```

Simple BTE at B = 2048 takes about 12 s per `predecrypt` iteration, plus context construction.

## Results: Combine + Dec, n = 128

**Setup.** Apple M5 Pro, single-threaded on both sides (no arkworks `parallel` feature; `asm`
has no effect on aarch64). Pairing-free: Pallas, f = 42, t = 2f + 1 = 85, ℓ = 22 (so
t = f + 2ℓ − 1 holds with equality). Simple BTE: BLS12-381 with the same quorum size t = 85.
Partial decryptions are assumed correct, so neither side times share verification or
ciphertext-proof checks. Pairing-free Combine includes the per-quorum precomputation (≈ 0.36 ms),
since the quorum is only known once shares arrive. Pallas and Vesta measured identically
(`benches/curves.rs`).

Simple BTE's `predecrypt` depends only on the ciphertexts, so it can run while partial
decryptions are still arriving. The table reports both the total work and the post-quorum
critical path (Simple BTE combine + finalize). The pairing-free scheme has nothing to pipeline:
all of its work comes after the quorum is known.

| B | sub-batches | PF combine+dec (t=85) | PF combine+dec (all 128) | SBTE combine | SBTE predecrypt | SBTE finalize | SBTE post-quorum | SBTE total | total speedup | post-quorum ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 1 | 6.80 ms | 10.1 ms | 1.32 ms | 21.2 ms | 4.10 ms | 5.43 ms | 26.6 ms | 3.9x | 0.80x |
| 32 | 2 | 21.5 ms | 20.8 ms | 1.34 ms | 112.0 ms | 17.3 ms | 18.7 ms | 130.7 ms | 6.1x | 0.87x |
| 128 | 6 | 75.7 ms | 63.4 ms | 1.34 ms | 557.1 ms | 68.7 ms | 70.1 ms | 627.1 ms | 8.3x | 0.93x |
| 512 | 24 | 297.1 ms | 250.6 ms | 1.36 ms | 2.68 s | 273.3 ms | 274.7 ms | 2.95 s | 9.9x | 0.92x |
| 2048 | 94 | 1.17 s | 979.2 ms | 1.40 ms | 12.41 s | 1.09 s | 1.09 s | 13.50 s | 11.6x | 0.94x |

"Total speedup" is SBTE total / PF (t = 85). "Post-quorum ratio" is SBTE post-quorum / PF
(t = 85); values below 1 mean the pairing-free scheme is slower on that path. With all 128
parties responding, the erasure step disappears. The pairing-free scheme then beats Simple BTE's
post-quorum time from B = 128 onward (63 vs 70 ms, 251 vs 275 ms, 0.98 vs 1.09 s).

Pairing-free combine strategies (each including the quorum precomputation):

| B | PF combine FFT (t) | PF combine Lagrange (t) | PF combine hybrid (t) | PF dec |
|---|---|---|---|---|
| 8 | 12.3 ms | 6.84 ms | 6.80 ms | 1.1 µs |
| 32 | 25.1 ms | 27.1 ms | 21.5 ms | 4.3 µs |
| 128 | 77.2 ms | 106.0 ms | 75.7 ms | 16.8 µs |
| 512 | 302.0 ms | 429.2 ms | 297.0 ms | 67.9 µs |
| 2048 | 1.18 s | 1.69 s | 1.17 s | 279.5 µs |

The FFT path costs about 12.5 ms per sub-batch of 22 ciphertexts with |T| = 85, and about
10.4 ms when all 128 parties respond. Lagrange costs about 0.8 ms per occupied slot. Dec costs
about 135 ns per ciphertext.

Communication trade-off (compressed points):

| B | PF partial decryption (per party) | SBTE partial decryption |
|---|---|---|
| 8 | 1 × 32 B = 32 B | 48 B |
| 32 | 2 × 32 B = 64 B | 48 B |
| 128 | 6 × 32 B = 192 B | 48 B |
| 512 | 24 × 32 B = 768 B | 48 B |
| 2048 | 94 × 32 B = 3008 B | 48 B |

### Where the time goes

A combine sub-batch performs about 500 GLV scalar multiplications: 85 erasure scalings, 321
non-trivial twiddles in the 128-point inverse FFT, 31 coset shifts, 49 twiddles in the 32-point
FFT, and 22 output scalings. Each costs about 23 µs: roughly 128 doublings at 82 ns plus about
51 mixed additions. Doublings now dominate, so further gains would have to come from fewer
multiplications (for example a pruned 4×32 decomposition of the inverse FFT, ≈ 6%), not cheaper
ones.
