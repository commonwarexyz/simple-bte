//! Pairing-free batched threshold encryption from packed secret sharing
//! (Appendix A of "DKG Is All You Need").
//!
//! This implements the optimistic path: partial decryptions are assumed to be
//! correct, so there are no verification keys, no per-party DLEQ proofs and no
//! `PartVerify`.

use crate::fft::GroupFft;
use crate::glv::{GlvScalar, glv_mul_batch};
use crate::params::Params;
use ark_ec::scalar_mul::glv::GLVConfig;
use ark_ec::short_weierstrass::{Affine, Projective, SWCurveConfig};
use ark_ec::{AffineRepr, CurveConfig, CurveGroup, PrimeGroup, VariableBaseMSM};
use ark_ff::{Field, One, PrimeField, Zero};
use ark_serialize::CanonicalSerialize;
use ark_std::UniformRand;
use ark_std::rand::Rng;
use merlin::Transcript;

pub type Scalar<P> = <P as CurveConfig>::ScalarField;
type BigInt<P> = <Scalar<P> as PrimeField>::BigInt;

/// `pk = g^sk`.
#[derive(Clone, Debug)]
pub struct EncryptionKey<P: SWCurveConfig> {
    pub pk: Affine<P>,
}

/// Party `index` (0-based) holds `share = S(omega^index)`.
#[derive(Clone, Debug)]
pub struct SecretKey<F> {
    pub index: usize,
    pub share: F,
}

/// Fiat-Shamir Schnorr proof of knowledge of `r` with `c1 = g^r`, bound to `c2`.
#[derive(Clone, Debug)]
pub struct SchnorrProof<P: SWCurveConfig> {
    pub commitment: Affine<P>,
    pub response: Scalar<P>,
}

/// ElGamal ciphertext `(g^r, pk^r * M)` with a proof of knowledge of `r`.
#[derive(Clone, Debug)]
pub struct Ciphertext<P: SWCurveConfig> {
    pub c1: Affine<P>,
    pub c2: Affine<P>,
    pub proof: SchnorrProof<P>,
}

/// A batch after proof verification: ciphertexts whose proof fails are
/// replaced by the dummy `c1 = 1_G`.
#[derive(Clone, Debug)]
pub struct PreparedBatch<P: SWCurveConfig> {
    pub c1: Vec<Affine<P>>,
    pub valid: Vec<bool>,
}

/// One group element per sub-batch of `ell` ciphertexts.
#[derive(Clone, Debug)]
pub struct PartialDecryption<P: SWCurveConfig> {
    pub index: usize,
    pub values: Vec<Affine<P>>,
}

// ---------------------------------------------------------------------------
// Setup / Encrypt
// ---------------------------------------------------------------------------

/// Dealer stand-in for the DKG: a packed sharing of `sk`, i.e. a random
/// `S(X) = sk + Z_D(X) * T(X)` with `deg T <= f - 1`, so `deg S <= f + ell - 1`
/// and `S = sk` on the `ell` slot points.
pub fn setup<P: SWCurveConfig>(
    params: &Params<Scalar<P>>,
    rng: &mut impl Rng,
) -> (EncryptionKey<P>, Vec<SecretKey<Scalar<P>>>) {
    let sk = Scalar::<P>::rand(rng);
    let t_coeffs: Vec<Scalar<P>> = (0..params.f).map(|_| Scalar::<P>::rand(rng)).collect();
    let slots = &params.slot_points[..params.ell];

    let shares = (0..params.n)
        .map(|index| {
            let x = params.party_points[index];
            let z_d: Scalar<P> = slots.iter().map(|d| x - d).product();
            let t_x = t_coeffs
                .iter()
                .rev()
                .fold(Scalar::<P>::zero(), |acc, c| acc * x + c);
            SecretKey {
                index,
                share: sk + z_d * t_x,
            }
        })
        .collect();

    let pk = (Projective::<P>::generator() * sk).into_affine();
    (EncryptionKey { pk }, shares)
}

pub fn encrypt<P: SWCurveConfig>(
    ek: &EncryptionKey<P>,
    message: &Affine<P>,
    rng: &mut impl Rng,
) -> Ciphertext<P> {
    let g = Projective::<P>::generator();
    let r = Scalar::<P>::rand(rng);
    let c1 = (g * r).into_affine();
    let c2 = (ek.pk * r + message).into_affine();

    let a = Scalar::<P>::rand(rng);
    let commitment = (g * a).into_affine();
    let e = schnorr_challenge::<P>(&c1, &c2, &commitment);
    Ciphertext {
        c1,
        c2,
        proof: SchnorrProof {
            commitment,
            response: a + e * r,
        },
    }
}

fn schnorr_challenge<P: SWCurveConfig>(
    c1: &Affine<P>,
    c2: &Affine<P>,
    commitment: &Affine<P>,
) -> Scalar<P> {
    let mut transcript = Transcript::new(b"pairing-free-bte-schnorr");
    for (label, point) in [
        (&b"c1"[..], c1),
        (&b"c2"[..], c2),
        (&b"commitment"[..], commitment),
    ] {
        let mut bytes = Vec::new();
        point
            .serialize_compressed(&mut bytes)
            .expect("serialization");
        transcript.append_message(label, &bytes);
    }
    let mut buf = [0u8; 64];
    transcript.challenge_bytes(b"challenge", &mut buf);
    Scalar::<P>::from_le_bytes_mod_order(&buf)
}

fn verify_proof<P: SWCurveConfig>(ct: &Ciphertext<P>) -> bool {
    let e = schnorr_challenge::<P>(&ct.c1, &ct.c2, &ct.proof.commitment);
    Projective::<P>::generator() * ct.proof.response == ct.proof.commitment + ct.c1 * e
}

/// Verify every ciphertext proof (one randomized batch check, with a
/// per-ciphertext fallback if it fails) and replace failures by dummies.
pub fn prepare_batch<P: SWCurveConfig>(
    cts: &[Ciphertext<P>],
    rng: &mut impl Rng,
) -> PreparedBatch<P> {
    let mut lhs = Scalar::<P>::zero();
    let mut bases = Vec::with_capacity(2 * cts.len());
    let mut scalars = Vec::with_capacity(2 * cts.len());
    for ct in cts {
        let w = Scalar::<P>::rand(rng);
        let e = schnorr_challenge::<P>(&ct.c1, &ct.c2, &ct.proof.commitment);
        lhs += w * ct.proof.response;
        bases.push(ct.proof.commitment);
        scalars.push(w);
        bases.push(ct.c1);
        scalars.push(w * e);
    }
    let all_ok =
        Projective::<P>::generator() * lhs == Projective::<P>::msm(&bases, &scalars).unwrap();

    let valid: Vec<bool> = if all_ok {
        vec![true; cts.len()]
    } else {
        cts.iter().map(verify_proof).collect()
    };
    let c1 = cts
        .iter()
        .zip(&valid)
        .map(|(ct, &ok)| if ok { ct.c1 } else { Affine::zero() })
        .collect();
    PreparedBatch { c1, valid }
}

// ---------------------------------------------------------------------------
// PartDec
// ---------------------------------------------------------------------------

/// For each sub-batch `u` with randomness polynomial `R_u` (`R_u(d_s) = r_{u,s}`),
/// output `g^{sk_i * R_u(x_i)}`: an MSM of `ell` terms whose public Lagrange
/// coefficients are pre-multiplied by `sk_i`.
pub fn partial_decrypt<P: SWCurveConfig>(
    params: &Params<Scalar<P>>,
    sk: &SecretKey<Scalar<P>>,
    batch: &PreparedBatch<P>,
) -> PartialDecryption<P> {
    let x = params.party_points[sk.index];
    let coeffs: Vec<Scalar<P>> = params
        .slot_lagrange_at(x)
        .into_iter()
        .map(|l| l * sk.share)
        .collect();
    let values: Vec<Projective<P>> = batch
        .c1
        .chunks(params.ell)
        .map(|chunk| Projective::<P>::msm(chunk, &coeffs[..chunk.len()]).unwrap())
        .collect();
    PartialDecryption {
        index: sk.index,
        values: Projective::<P>::normalize_batch(&values),
    }
}

// ---------------------------------------------------------------------------
// Combine
// ---------------------------------------------------------------------------
//
// Combine recovers P = S * R (degree <= f + 2*ell - 2 < t) in the exponent
// from the quorum's evaluations on H_N and evaluates it on the ell slots.
// Writing T for the quorum and Zbar(X) = prod_{k in H_N \ T} (X - omega^k):
//
// * Lagrange: P(d_s) = sum_{j in T} lambda_j(d_s) * pd_j with barycentric
//   weights that are cheap on roots of unity, w_j = 1 / Z_T'(x_j)
//   = x_j * Zbar(x_j) / N. One MSM of size |T| per ciphertext.
//
// * FFT: Q = P * Zbar has degree < N and Q(omega^k) is pd_k * Zbar(omega^k) on
//   T and 0 off T, so A = N * coeffs(Q) is one inverse DFT of size N. Then
//   P(d_s) = A(d_s) / (N * Zbar(d_s)), and A is evaluated on the coset c * H_m
//   by folding modulo X^m - h (h = c^m small), scaling by c^r, and a DFT of
//   size m. When the whole domain responds, Zbar = 1 and the erasure step
//   disappears.

/// Per-`Params` tables for the FFT combine.
#[derive(Clone, Debug)]
pub struct CombineContext<P: GLVConfig> {
    inv_fft_n: GroupFft,
    fft_m: GroupFft,
    /// `c^r` for `r < m`.
    offset_powers: Vec<GlvScalar>,
    /// `h` as an integer.
    fold_scalar: BigInt<P>,
}

impl<P: GLVConfig> CombineContext<P> {
    pub fn new(params: &Params<Scalar<P>>) -> Self {
        let omega_inv = params.omega.inverse().unwrap();
        let mut offset_powers = Vec::with_capacity(params.m);
        let mut c = Scalar::<P>::one();
        for _ in 0..params.m {
            offset_powers.push(GlvScalar::new::<P>(c));
            c *= params.offset;
        }
        Self {
            inv_fft_n: GroupFft::new::<P>(params.big_n, omega_inv),
            fft_m: GroupFft::new::<P>(params.m, params.nu),
            offset_powers,
            fold_scalar: params.offset_pow_m.into_bigint(),
        }
    }
}

/// Per-quorum constants. Depends only on which parties responded, so it is
/// shared by every sub-batch.
#[derive(Clone, Debug)]
pub struct QuorumKey<P: GLVConfig> {
    pub indices: Vec<usize>,
    /// `lagrange[s][j] = lambda_j(d_s)` for `s < ell`, `j` in quorum order.
    lagrange: Vec<Vec<BigInt<P>>>,
    /// `Zbar(x_j)` in quorum order, or `None` when all of `H_N` responded.
    erasure: Option<Vec<GlvScalar>>,
    /// `1 / (N * Zbar(d_s))` for `s < ell`.
    out_scale: Vec<GlvScalar>,
}

impl<P: GLVConfig> QuorumKey<P> {
    pub fn new(params: &Params<Scalar<P>>, indices: &[usize]) -> Self {
        assert!(indices.len() >= params.t, "quorum too small");
        let big_n = params.big_n;
        let ell = params.ell;
        let mut present = vec![false; big_n];
        for &i in indices {
            assert!(i < params.n && !present[i], "bad or duplicate index {i}");
            present[i] = true;
        }
        let absent: Vec<Scalar<P>> = (0..big_n)
            .filter(|&k| !present[k])
            .map(|k| params.party_points[k])
            .collect();
        let zbar = |z: Scalar<P>| -> Scalar<P> { absent.iter().map(|a| z - a).product() };

        let xs: Vec<Scalar<P>> = indices.iter().map(|&i| params.party_points[i]).collect();
        let slots = &params.slot_points[..ell];
        let zbar_party: Vec<Scalar<P>> = xs.iter().map(|&x| zbar(x)).collect();
        let zbar_slot: Vec<Scalar<P>> = slots.iter().map(|&d| zbar(d)).collect();

        // Invert N, Zbar(d_s) and (d_s - x_j) in one batch.
        let tq = xs.len();
        let mut inv = Vec::with_capacity(1 + ell + ell * tq);
        inv.push(Scalar::<P>::from(big_n as u64));
        inv.extend_from_slice(&zbar_slot);
        for &d in slots {
            inv.extend(xs.iter().map(|&x| d - x));
        }
        ark_ff::batch_inversion(&mut inv);
        let n_inv = inv[0];
        let zbar_slot_inv = &inv[1..1 + ell];
        let diff_inv = &inv[1 + ell..];

        let weights: Vec<Scalar<P>> = xs
            .iter()
            .zip(&zbar_party)
            .map(|(&x, &z)| x * z * n_inv)
            .collect();
        let lagrange = (0..ell)
            .map(|s| {
                let d = slots[s];
                let z_t = (d.pow([big_n as u64]) - Scalar::<P>::one()) * zbar_slot_inv[s];
                (0..tq)
                    .map(|j| (z_t * weights[j] * diff_inv[s * tq + j]).into_bigint())
                    .collect()
            })
            .collect();

        let erasure = (!absent.is_empty())
            .then(|| zbar_party.iter().map(|&z| GlvScalar::new::<P>(z)).collect());
        let out_scale = zbar_slot_inv
            .iter()
            .map(|&z| GlvScalar::new::<P>(z * n_inv))
            .collect();

        Self {
            indices: indices.to_vec(),
            lagrange,
            erasure,
            out_scale,
        }
    }

    fn check_order(&self, pds: &[PartialDecryption<P>]) {
        assert_eq!(pds.len(), self.indices.len());
        for (pd, &i) in pds.iter().zip(&self.indices) {
            assert_eq!(pd.index, i, "partial decryptions out of quorum order");
        }
    }
}

/// Sub-batches with fewer occupied slots than this are combined with MSMs
/// rather than FFTs: one FFT sub-batch costs about as much as 16 MSMs of size
/// `t` at n = 128.
pub const LAGRANGE_CUTOFF: usize = 16;

#[derive(Clone, Copy)]
enum Strategy {
    Lagrange,
    Fft,
    Hybrid,
}

/// Combine via one MSM of size `|T|` per ciphertext. Returns `g^{sk * r_i}`
/// for every ciphertext `i < batch_len`.
pub fn combine_lagrange<P: GLVConfig>(
    params: &Params<Scalar<P>>,
    ctx: &CombineContext<P>,
    qk: &QuorumKey<P>,
    pds: &[PartialDecryption<P>],
    batch_len: usize,
) -> Vec<Projective<P>> {
    combine_with(params, ctx, qk, pds, batch_len, Strategy::Lagrange)
}

/// Combine via an inverse DFT of size `N` and a coset DFT of size `m` per
/// sub-batch. Same output as [`combine_lagrange`].
pub fn combine_fft<P: GLVConfig>(
    params: &Params<Scalar<P>>,
    ctx: &CombineContext<P>,
    qk: &QuorumKey<P>,
    pds: &[PartialDecryption<P>],
    batch_len: usize,
) -> Vec<Projective<P>> {
    combine_with(params, ctx, qk, pds, batch_len, Strategy::Fft)
}

/// FFT for sub-batches with at least [`LAGRANGE_CUTOFF`] occupied slots,
/// MSMs otherwise (in practice: a short final sub-batch, or `B < ell`).
pub fn combine<P: GLVConfig>(
    params: &Params<Scalar<P>>,
    ctx: &CombineContext<P>,
    qk: &QuorumKey<P>,
    pds: &[PartialDecryption<P>],
    batch_len: usize,
) -> Vec<Projective<P>> {
    combine_with(params, ctx, qk, pds, batch_len, Strategy::Hybrid)
}

fn combine_with<P: GLVConfig>(
    params: &Params<Scalar<P>>,
    ctx: &CombineContext<P>,
    qk: &QuorumKey<P>,
    pds: &[PartialDecryption<P>],
    batch_len: usize,
    strategy: Strategy,
) -> Vec<Projective<P>> {
    qk.check_order(pds);
    let mut out = Vec::with_capacity(batch_len);
    let mut scratch = FftScratch::new(params, ctx, qk);
    let mut bases = vec![Affine::<P>::zero(); pds.len()];

    for u in 0..params.num_subbatches(batch_len) {
        let slots = params.ell.min(batch_len - u * params.ell);
        let use_fft = match strategy {
            Strategy::Lagrange => false,
            Strategy::Fft => true,
            Strategy::Hybrid => slots >= LAGRANGE_CUTOFF,
        };
        if use_fft {
            scratch.run(params, ctx, pds, u, slots, &mut out);
        } else {
            for (b, pd) in bases.iter_mut().zip(pds) {
                *b = pd.values[u];
            }
            for s in 0..slots {
                out.push(Projective::<P>::msm_bigint(&bases, &qk.lagrange[s]));
            }
        }
    }
    out
}

struct FftScratch<'a, P: GLVConfig> {
    a: Vec<Projective<P>>,
    b: Vec<Projective<P>>,
    offset_refs: Vec<&'a GlvScalar>,
    out_refs: Vec<&'a GlvScalar>,
    erasure_refs: Option<Vec<&'a GlvScalar>>,
}

impl<'a, P: GLVConfig> FftScratch<'a, P> {
    fn new(params: &Params<Scalar<P>>, ctx: &'a CombineContext<P>, qk: &'a QuorumKey<P>) -> Self {
        Self {
            a: vec![Projective::zero(); params.big_n],
            b: vec![Projective::zero(); params.m],
            offset_refs: ctx.offset_powers[1..].iter().collect(),
            out_refs: qk.out_scale.iter().collect(),
            erasure_refs: qk.erasure.as_ref().map(|z| z.iter().collect()),
        }
    }

    fn run(
        &mut self,
        params: &Params<Scalar<P>>,
        ctx: &CombineContext<P>,
        pds: &[PartialDecryption<P>],
        u: usize,
        slots: usize,
        out: &mut Vec<Projective<P>>,
    ) {
        let (a, b) = (&mut self.a, &mut self.b);

        // Evaluations of Q = P * Zbar on H_N.
        a.iter_mut().for_each(|x| *x = Projective::zero());
        let vals: Vec<Projective<P>> = pds.iter().map(|pd| pd.values[u].into_group()).collect();
        let vals = match &self.erasure_refs {
            Some(z) => glv_mul_batch(&vals, z),
            None => vals,
        };
        for (pd, v) in pds.iter().zip(vals) {
            a[pd.index] = v;
        }
        // a <- N * coeffs(Q).
        ctx.inv_fft_n.apply(a);

        // Fold modulo X^m - h (Horner in h over the blocks of m coefficients).
        b.iter_mut().for_each(|x| *x = Projective::zero());
        for block in a.chunks(params.m).rev() {
            for (acc, coeff) in b.iter_mut().zip(block) {
                *acc = acc.mul_bigint(ctx.fold_scalar) + coeff;
            }
        }
        // Evaluate on c * H_m.
        let shifted = glv_mul_batch(&b[1..], &self.offset_refs);
        b[1..].copy_from_slice(&shifted);
        ctx.fft_m.apply(b);

        out.extend(glv_mul_batch(&b[..slots], &self.out_refs[..slots]));
    }
}

// ---------------------------------------------------------------------------
// Dec
// ---------------------------------------------------------------------------

/// `M_i = c2_i / g^{sk * r_i}` for ciphertexts whose proof verified.
pub fn decrypt<P: SWCurveConfig>(
    batch: &PreparedBatch<P>,
    cts: &[Ciphertext<P>],
    pd: &[Projective<P>],
) -> Vec<Option<Projective<P>>> {
    assert_eq!(cts.len(), pd.len());
    cts.iter()
        .zip(pd)
        .zip(&batch.valid)
        .map(|((ct, mask), &ok)| ok.then(|| -*mask + ct.c2))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_pallas::PallasConfig;
    use ark_std::rand::seq::SliceRandom;
    use ark_std::test_rng;

    type P = PallasConfig;
    type G = Projective<P>;

    struct Fixture {
        params: Params<Scalar<P>>,
        ctx: CombineContext<P>,
        sks: Vec<SecretKey<Scalar<P>>>,
        msgs: Vec<Affine<P>>,
        cts: Vec<Ciphertext<P>>,
        batch: PreparedBatch<P>,
    }

    fn fixture(params: Params<Scalar<P>>, batch_len: usize) -> Fixture {
        let mut rng = test_rng();
        let (ek, sks) = setup::<P>(&params, &mut rng);
        let msgs: Vec<Affine<P>> = (0..batch_len)
            .map(|_| G::rand(&mut rng).into_affine())
            .collect();
        let cts: Vec<_> = msgs.iter().map(|m| encrypt(&ek, m, &mut rng)).collect();
        let batch = prepare_batch(&cts, &mut rng);
        let ctx = CombineContext::new(&params);
        Fixture {
            params,
            ctx,
            sks,
            msgs,
            cts,
            batch,
        }
    }

    fn quorum(fx: &Fixture, indices: &[usize]) -> (QuorumKey<P>, Vec<PartialDecryption<P>>) {
        let pds = indices
            .iter()
            .map(|&i| partial_decrypt(&fx.params, &fx.sks[i], &fx.batch))
            .collect();
        (QuorumKey::new(&fx.params, indices), pds)
    }

    fn random_quorum(n: usize, size: usize, rng: &mut impl Rng) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..n).collect();
        idx.shuffle(rng);
        idx.truncate(size);
        idx
    }

    fn assert_roundtrip(fx: &Fixture, pd: &[G]) {
        let out = decrypt(&fx.batch, &fx.cts, pd);
        for (i, (m, o)) in fx.msgs.iter().zip(out).enumerate() {
            assert_eq!(o.expect("valid ciphertext"), *m, "message {i}");
        }
    }

    #[test]
    fn roundtrip_random_quorum_both_combines() {
        let params = Params::bft(128);
        let b = 3 * params.ell + 5; // not a multiple of ell
        let fx = fixture(params, b);
        let mut rng = test_rng();
        let idx = random_quorum(fx.params.n, fx.params.t, &mut rng);
        let (qk, pds) = quorum(&fx, &idx);

        let lag = combine_lagrange(&fx.params, &fx.ctx, &qk, &pds, b);
        let fft = combine_fft(&fx.params, &fx.ctx, &qk, &pds, b);
        assert_eq!(lag, fft);
        assert_eq!(lag, combine(&fx.params, &fx.ctx, &qk, &pds, b));
        assert_roundtrip(&fx, &lag);
    }

    #[test]
    fn different_quorums_agree() {
        let params = Params::bft(64);
        let b = 2 * params.ell;
        let fx = fixture(params, b);
        let mut rng = test_rng();
        let mut results = Vec::new();
        for _ in 0..3 {
            let idx = random_quorum(fx.params.n, fx.params.t, &mut rng);
            let (qk, pds) = quorum(&fx, &idx);
            results.push(combine_fft(&fx.params, &fx.ctx, &qk, &pds, b));
        }
        assert_eq!(results[0], results[1]);
        assert_eq!(results[1], results[2]);
        assert_roundtrip(&fx, &results[0]);
    }

    #[test]
    fn full_participation_skips_erasure() {
        let params = Params::bft(32);
        let b = params.ell + 1;
        let fx = fixture(params, b);
        let idx: Vec<usize> = (0..fx.params.n).rev().collect();
        let (qk, pds) = quorum(&fx, &idx);
        assert!(qk.erasure.is_none());
        let fft = combine_fft(&fx.params, &fx.ctx, &qk, &pds, b);
        assert_eq!(fft, combine_lagrange(&fx.params, &fx.ctx, &qk, &pds, b));
        assert_roundtrip(&fx, &fft);
    }

    #[test]
    fn non_power_of_two_committee() {
        // n = 100: N = 128 with 28 domain points never assigned to a party.
        let params = Params::bft(100);
        let b = 2 * params.ell + 3;
        let fx = fixture(params, b);
        let mut rng = test_rng();
        let idx = random_quorum(fx.params.n, fx.params.t + 4, &mut rng);
        let (qk, pds) = quorum(&fx, &idx);
        let fft = combine_fft(&fx.params, &fx.ctx, &qk, &pds, b);
        assert_eq!(fft, combine_lagrange(&fx.params, &fx.ctx, &qk, &pds, b));
        assert_roundtrip(&fx, &fft);
    }

    #[test]
    fn small_batch_and_non_bft_params() {
        // ell = 4 (m = 4), t = f + 2*ell - 1 exactly, batch smaller than ell.
        let params = Params::new(20, 5, 12, 4);
        let fx = fixture(params, 3);
        let mut rng = test_rng();
        let idx = random_quorum(fx.params.n, fx.params.t, &mut rng);
        let (qk, pds) = quorum(&fx, &idx);
        let fft = combine_fft(&fx.params, &fx.ctx, &qk, &pds, 3);
        assert_eq!(fft, combine_lagrange(&fx.params, &fx.ctx, &qk, &pds, 3));
        assert_roundtrip(&fx, &fft);
    }

    #[test]
    fn invalid_proof_becomes_dummy() {
        let params = Params::bft(32);
        let b = params.ell + 2;
        let mut fx = fixture(params, b);
        fx.cts[1].proof.response += Scalar::<P>::one();
        let mut rng = test_rng();
        fx.batch = prepare_batch(&fx.cts, &mut rng);
        assert!(!fx.batch.valid[1]);
        assert!(fx.batch.c1[1].is_zero());

        let idx = random_quorum(fx.params.n, fx.params.t, &mut rng);
        let (qk, pds) = quorum(&fx, &idx);
        let pd = combine_fft(&fx.params, &fx.ctx, &qk, &pds, b);
        let out = decrypt(&fx.batch, &fx.cts, &pd);
        for (i, o) in out.into_iter().enumerate() {
            if i == 1 {
                assert!(o.is_none());
            } else {
                assert_eq!(o.unwrap(), fx.msgs[i]);
            }
        }
    }

    #[test]
    fn shares_are_constant_on_slots() {
        // Interpolating S from f + ell shares recovers sk at every slot.
        let params = Params::<Scalar<P>>::bft(32);
        let mut rng = test_rng();
        let (ek, sks) = setup::<P>(&params, &mut rng);
        let k = params.f + params.ell;
        let xs: Vec<_> = (0..k).map(|i| params.party_points[i]).collect();
        let mut values = Vec::new();
        for s in 0..params.ell {
            let d = params.slot_points[s];
            let mut acc = Scalar::<P>::zero();
            for j in 0..k {
                let mut l = Scalar::<P>::one();
                for i in 0..k {
                    if i != j {
                        l *= (d - xs[i]) / (xs[j] - xs[i]);
                    }
                }
                acc += l * sks[j].share;
            }
            values.push(acc);
        }
        for v in &values {
            assert_eq!(G::generator() * v, ek.pk.into_group());
        }
    }
}
