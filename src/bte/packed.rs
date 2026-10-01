//! Simple BTE with packed secret sharing (Remark 2 of "DKG Is All You Need").
//!
//! In the ramp setting `t >= f + 2*ell - 1`, a batch of `B` ciphertexts is split
//! into `U = ceil(B / ell)` sub-batches of `ell`. Ciphertext `i` sits in slot
//! `s = i % ell` of sub-batch `u = i / ell`. Each slot is an independent Simple
//! BTE instance of batch size `U`, and all slots share the same powers
//! `tau, ..., tau^U`. Parties hold a *packed* sharing `S_u` of each `tau^{u+1}`,
//! constant on the `ell` slot points, so:
//!
//! * public parameters and per-party state shrink from `B` to `U` powers/shares;
//! * a partial decryption is still one G1 element,
//!   `pd_j = sum_u S_u(x_j) * R_u(x_j)`, where `R_u` interpolates the `ct1`s of
//!   sub-batch `u` over the slots. `sum_u S_u * R_u` has degree `< t` and equals
//!   the Simple BTE aggregate `sum_u tau^{u+1} * ct1^{(s,u)}` at slot `s`;
//! * verifying a partial decryption costs `U + 1` pairings instead of `B + 1`;
//! * Combine interpolates to the `ell` slots, and each slot is opened with the
//!   existing `predecrypt_fft` / finalize on `U` ciphertexts.
//!
//! Domains: party `k` (0-based) sits at `omega^k` in `H_N`, `N = next_pow2(n)`;
//! slot `s` sits at `g * nu^s` in the coset `g * H_m`, `m = next_pow2(ell)`,
//! where `g` is the field's multiplicative generator.

use super::crs::public_keys;
use super::decryption::{predecrypt_fft, verify_ciphertext_batch, CrossTerms};
use super::{Ciphertext, DecryptionKey, EncryptionKey, SchnorrProof};
use ark_ec::pairing::{Pairing, PairingOutput};
use ark_ec::{AffineRepr, CurveGroup, PrimeGroup, ScalarMul, VariableBaseMSM};
use ark_ff::{batch_inversion, FftField, Field, One, Zero};
use ark_std::rand::Rng;
use ark_std::UniformRand;

/// Thresholds, batch shape and evaluation domains.
#[derive(Clone, Debug)]
pub struct PackedParams<F: FftField> {
    /// Number of parties.
    pub n: usize,
    /// Corruption threshold.
    pub f: usize,
    /// Reconstruction threshold (quorum size).
    pub t: usize,
    /// Packing factor: slots per sub-batch.
    pub ell: usize,
    /// Number of sub-batches `U`; the batch holds up to `ell * U` ciphertexts.
    pub num_subbatches: usize,
    /// Size of the party domain `H_N`.
    pub big_n: usize,
    /// `party_points[k] = omega^k` for `k < N`; parties use the first `n`.
    pub party_points: Vec<F>,
    /// `slot_points[s] = g * nu^s` for `s < ell`.
    pub slot_points: Vec<F>,
}

impl<F: FftField> PackedParams<F> {
    /// Parameters for batches of up to `max_batch` ciphertexts.
    pub fn new(n: usize, f: usize, t: usize, ell: usize, max_batch: usize) -> Self {
        assert!(ell >= 1 && max_batch >= 1);
        assert!(t <= n, "quorum larger than the committee");
        assert!(t >= f + 2 * ell - 1, "need t >= f + 2*ell - 1");

        let big_n = n.next_power_of_two();
        let m = ell.next_power_of_two();
        let omega = F::get_root_of_unity(big_n as u64).expect("2-adicity too small for n");
        let nu = F::get_root_of_unity(m as u64).expect("2-adicity too small for ell");

        let party_points = powers(F::one(), omega, big_n);
        let slot_points = powers(F::GENERATOR, nu, ell);
        for d in &slot_points {
            assert!(!d.pow([big_n as u64]).is_one(), "slot coset meets H_N");
        }

        Self {
            n,
            f,
            t,
            ell,
            num_subbatches: max_batch.div_ceil(ell),
            big_n,
            party_points,
            slot_points,
        }
    }

    pub fn batch_capacity(&self) -> usize {
        self.ell * self.num_subbatches
    }

    /// Lagrange basis over the slot points, evaluated at `x`.
    pub fn slot_lagrange_at(&self, x: F) -> Vec<F> {
        let d = &self.slot_points;
        let mut denoms: Vec<F> = (0..self.ell)
            .map(|s| {
                let w: F = (0..self.ell)
                    .filter(|&k| k != s)
                    .map(|k| d[s] - d[k])
                    .product();
                w * (x - d[s])
            })
            .collect();
        batch_inversion(&mut denoms);
        let z: F = d.iter().map(|ds| x - ds).product();
        denoms.into_iter().map(|inv| z * inv).collect()
    }
}

fn powers<F: Field>(start: F, step: F, len: usize) -> Vec<F> {
    let mut out = Vec::with_capacity(len);
    let mut cur = start;
    for _ in 0..len {
        out.push(cur);
        cur *= step;
    }
    out
}

/// Public parameters for verification, combining and decryption.
#[derive(Clone, Debug)]
pub struct PackedDecryptionKey<E: Pairing> {
    pub params: PackedParams<E::ScalarField>,
    /// Simple BTE key for batch size `U`, shared by every slot.
    /// Its `verification_keys` field is unused (empty).
    pub inner: DecryptionKey<E>,
    /// `verification_keys[k][u] = [S_u(omega^k)]_2`.
    pub verification_keys: Vec<Vec<E::G2Affine>>,
}

/// Party `index` (0-based) holds `shares[u] = S_u(omega^index)` for `u < U`.
#[derive(Clone, Debug)]
pub struct PackedSecretKey<E: Pairing> {
    pub index: usize,
    pub shares: Vec<E::ScalarField>,
}

#[derive(Clone, Debug)]
pub struct PackedPartialDecryption<E: Pairing> {
    pub value: E::G1,
    /// 0-based party index.
    pub index: usize,
}

/// Cross terms for every slot, computable before any partial decryption.
pub struct PackedCrossTerms<E: Pairing> {
    pub per_slot: Vec<CrossTerms<E>>,
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

/// Trusted setup (dealer stand-in for the MPC). Ciphertexts are produced with
/// the ordinary [`super::encryption::encrypt`] under the returned key.
pub fn setup<E: Pairing>(
    params: &PackedParams<E::ScalarField>,
    rng: &mut impl Rng,
) -> (
    EncryptionKey<E>,
    PackedDecryptionKey<E>,
    Vec<PackedSecretKey<E>>,
) {
    let tau = E::ScalarField::rand(rng);
    setup_with_tau(params, tau, rng)
}

/// Packed sharing `S_u(X) = tau^{u+1} + Z_D(X) * T_u(X)` with `deg T_u <= f - 1`,
/// so `deg S_u <= f + ell - 1` and `S_u = tau^{u+1}` on the slots.
fn setup_with_tau<E: Pairing>(
    params: &PackedParams<E::ScalarField>,
    tau: E::ScalarField,
    rng: &mut impl Rng,
) -> (
    EncryptionKey<E>,
    PackedDecryptionKey<E>,
    Vec<PackedSecretKey<E>>,
) {
    let u_count = params.num_subbatches;
    let (ek, inner, tau_powers) = public_keys::<E>(tau, u_count, params.n, params.t);

    let xs = &params.party_points[..params.n];
    let z_d: Vec<E::ScalarField> = xs
        .iter()
        .map(|&x| params.slot_points.iter().map(|d| x - d).product())
        .collect();

    let mut shares = vec![Vec::with_capacity(u_count); params.n];
    for u in 0..u_count {
        let t_coeffs: Vec<E::ScalarField> =
            (0..params.f).map(|_| E::ScalarField::rand(rng)).collect();
        for (k, s) in shares.iter_mut().enumerate() {
            let x = xs[k];
            let t_x = t_coeffs
                .iter()
                .rev()
                .fold(E::ScalarField::zero(), |acc, c| acc * x + c);
            s.push(tau_powers[u + 1] + z_d[k] * t_x);
        }
    }

    let verification_keys = shares
        .iter()
        .map(|s| E::G2::generator().batch_mul(s))
        .collect();
    let secret_keys = shares
        .into_iter()
        .enumerate()
        .map(|(index, shares)| PackedSecretKey { index, shares })
        .collect();

    let dk = PackedDecryptionKey {
        params: params.clone(),
        inner,
        verification_keys,
    };
    (ek, dk, secret_keys)
}

// ---------------------------------------------------------------------------
// PartialDec / Verify
// ---------------------------------------------------------------------------

/// `pd_j = sum_i S_{u(i)}(x_j) * L_{s(i)}(x_j) * ct1_i`: one MSM of size `B`.
pub fn partial_decrypt<E: Pairing>(
    params: &PackedParams<E::ScalarField>,
    sk: &PackedSecretKey<E>,
    cts: &[Ciphertext<E>],
    rng: &mut impl Rng,
) -> Option<PackedPartialDecryption<E>> {
    assert!(cts.len() <= params.batch_capacity(), "batch too large");
    if !verify_ciphertext_batch(cts, rng) {
        return None;
    }
    let lagrange = params.slot_lagrange_at(params.party_points[sk.index]);
    let scalars: Vec<E::ScalarField> = (0..cts.len())
        .map(|i| sk.shares[i / params.ell] * lagrange[i % params.ell])
        .collect();
    let bases: Vec<E::G1Affine> = cts.iter().map(|ct| ct.ct1).collect();
    Some(PackedPartialDecryption {
        value: E::G1::msm(&bases, &scalars).unwrap(),
        index: sk.index,
    })
}

/// Check `e(pd_j, [1]_2) = sum_u e(R_u(x_j), [S_u(x_j)]_2)`: `U + 1` pairings,
/// plus one `ell`-term MSM per sub-batch for `R_u(x_j)`.
///
/// Assumes `verify_ciphertext_batch(cts)` has already succeeded.
pub fn verify<E: Pairing>(
    dk: &PackedDecryptionKey<E>,
    pd: &PackedPartialDecryption<E>,
    cts: &[Ciphertext<E>],
) -> bool {
    let params = &dk.params;
    assert!(cts.len() <= params.batch_capacity(), "batch too large");
    let lagrange = params.slot_lagrange_at(params.party_points[pd.index]);
    let r: Vec<E::G1> = cts
        .chunks(params.ell)
        .map(|chunk| {
            let bases: Vec<E::G1Affine> = chunk.iter().map(|ct| ct.ct1).collect();
            E::G1::msm(&bases, &lagrange[..chunk.len()]).unwrap()
        })
        .collect();

    let mut g1 = Vec::with_capacity(r.len() + 1);
    g1.push(-pd.value);
    g1.extend(r);
    let g1 = E::G1::normalize_batch(&g1);
    let mut g2 = Vec::with_capacity(g1.len());
    g2.push(E::G2Affine::generator());
    g2.extend_from_slice(&dk.verification_keys[pd.index][..g1.len() - 1]);
    E::multi_pairing(g1, g2).is_zero()
}

// ---------------------------------------------------------------------------
// Combine
// ---------------------------------------------------------------------------

/// Interpolate `sum_u S_u * R_u` (degree `< t`) in the exponent from the quorum
/// and evaluate it on the slots, returning the Simple BTE aggregate
/// `[sum_u tau^{u+1} k_{s,u}]_1` of every slot `s`.
///
/// Uses barycentric weights on `H_N`: `w_j = 1 / Z_T'(x_j) = x_j * Zbar(x_j) / N`
/// with `Zbar` vanishing on the absent domain points, then one MSM of size
/// `|T|` per slot.
pub fn combine<E: Pairing>(
    params: &PackedParams<E::ScalarField>,
    pds: &[PackedPartialDecryption<E>],
) -> Vec<E::G1> {
    assert!(pds.len() >= params.t, "quorum too small");
    let big_n = params.big_n;
    let mut present = vec![false; big_n];
    for pd in pds {
        assert!(
            pd.index < params.n && !present[pd.index],
            "bad or duplicate index"
        );
        present[pd.index] = true;
    }
    let absent: Vec<E::ScalarField> = (0..big_n)
        .filter(|&k| !present[k])
        .map(|k| params.party_points[k])
        .collect();
    let zbar = |z: E::ScalarField| -> E::ScalarField { absent.iter().map(|a| z - a).product() };

    let xs: Vec<E::ScalarField> = pds.iter().map(|pd| params.party_points[pd.index]).collect();
    let tq = xs.len();
    let ell = params.ell;

    // Invert N, Zbar(d_s) and (d_s - x_j) in one batch.
    let mut inv = Vec::with_capacity(1 + ell + ell * tq);
    inv.push(E::ScalarField::from(big_n as u64));
    inv.extend(params.slot_points.iter().map(|&d| zbar(d)));
    for &d in &params.slot_points {
        inv.extend(xs.iter().map(|&x| d - x));
    }
    batch_inversion(&mut inv);
    let n_inv = inv[0];
    let zbar_slot_inv = &inv[1..1 + ell];
    let diff_inv = &inv[1 + ell..];

    let weights: Vec<E::ScalarField> = xs.iter().map(|&x| x * zbar(x) * n_inv).collect();
    let bases = E::G1::normalize_batch(&pds.iter().map(|pd| pd.value).collect::<Vec<_>>());

    (0..ell)
        .map(|s| {
            let d = params.slot_points[s];
            let z_t = (d.pow([big_n as u64]) - E::ScalarField::one()) * zbar_slot_inv[s];
            let coeffs: Vec<E::ScalarField> = (0..tq)
                .map(|j| z_t * weights[j] * diff_inv[s * tq + j])
                .collect();
            E::G1::msm(&bases, &coeffs).unwrap()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Dec
// ---------------------------------------------------------------------------

fn dummy_ciphertext<E: Pairing>() -> Ciphertext<E> {
    Ciphertext {
        ct1: E::G1Affine::zero(),
        ct2: PairingOutput::zero(),
        proof: SchnorrProof {
            commitment: E::G1Affine::zero(),
            response: E::ScalarField::zero(),
        },
    }
}

/// **Pre-decryption** (pipelineable): `predecrypt_fft` on the `U` ciphertexts
/// of every slot, padding a short final sub-batch with `ct1 = 0` dummies.
pub fn predecrypt<E: Pairing>(
    dk: &PackedDecryptionKey<E>,
    cts: &[Ciphertext<E>],
) -> PackedCrossTerms<E> {
    let params = &dk.params;
    assert!(cts.len() <= params.batch_capacity(), "batch too large");
    let u_count = params.num_subbatches;
    let per_slot = (0..params.ell)
        .map(|s| {
            if s >= cts.len() {
                return CrossTerms { values: Vec::new() };
            }
            let slot_cts: Vec<Ciphertext<E>> = (0..u_count)
                .map(|u| {
                    cts.get(u * params.ell + s)
                        .cloned()
                        .unwrap_or_else(dummy_ciphertext)
                })
                .collect();
            predecrypt_fft(&dk.inner, &slot_cts)
        })
        .collect();
    PackedCrossTerms { per_slot }
}

/// **Finalization**: `m_i = ct2_i - (e(pd_s, h_{U-u}) - C_s[u])` for ciphertext
/// `i` in slot `s` of sub-batch `u`. One pairing per real ciphertext.
pub fn finalize<E: Pairing>(
    dk: &PackedDecryptionKey<E>,
    pd_slots: &[E::G1],
    cts: &[Ciphertext<E>],
    cross: &PackedCrossTerms<E>,
) -> Vec<PairingOutput<E>> {
    let params = &dk.params;
    assert_eq!(pd_slots.len(), params.ell);
    let u_count = params.num_subbatches;
    let pd_affine = E::G1::normalize_batch(pd_slots);
    cts.iter()
        .enumerate()
        .map(|(i, ct)| {
            let (u, s) = (i / params.ell, i % params.ell);
            let pd_term = E::pairing(pd_affine[s], dk.inner.powers_of_h[u_count - u].clone());
            ct.ct2 - (pd_term - cross.per_slot[s].values[u])
        })
        .collect()
}

/// Verify ciphertext proofs, then `predecrypt` + `finalize`.
pub fn decrypt<E: Pairing>(
    dk: &PackedDecryptionKey<E>,
    pd_slots: &[E::G1],
    cts: &[Ciphertext<E>],
    rng: &mut impl Rng,
) -> Vec<PairingOutput<E>> {
    assert!(
        verify_ciphertext_batch(cts, rng),
        "invalid ciphertext batch"
    );
    let cross = predecrypt(dk, cts);
    finalize(dk, pd_slots, cts, &cross)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bte::encryption::encrypt;
    use ark_bls12_381::Bls12_381;
    use ark_std::rand::rngs::StdRng;
    use ark_std::rand::seq::SliceRandom;
    use ark_std::rand::SeedableRng;
    use ark_std::test_rng;

    type E = Bls12_381;
    type Fr = <E as Pairing>::ScalarField;
    type G1 = <E as Pairing>::G1;

    struct Fixture {
        params: PackedParams<Fr>,
        tau: Fr,
        dk: PackedDecryptionKey<E>,
        sks: Vec<PackedSecretKey<E>>,
        ks: Vec<Fr>,
        msgs: Vec<PairingOutput<E>>,
        cts: Vec<Ciphertext<E>>,
    }

    fn fixture(params: PackedParams<Fr>, batch_len: usize) -> Fixture {
        let mut rng = StdRng::seed_from_u64(7);
        let tau = Fr::rand(&mut rng);
        let (ek, dk, sks) = setup_with_tau::<E>(&params, tau, &mut rng);
        let msgs: Vec<PairingOutput<E>> = (0..batch_len)
            .map(|_| PairingOutput::<E>::generator() * Fr::rand(&mut rng))
            .collect();
        let mut ks = Vec::with_capacity(batch_len);
        let cts = msgs
            .iter()
            .map(|m| {
                // `encrypt` draws k first; replay the RNG to learn it.
                let k = Fr::rand(&mut rng.clone());
                let ct = encrypt(&ek, m, &mut rng);
                assert_eq!(ct.ct1, (G1::generator() * k).into_affine());
                ks.push(k);
                ct
            })
            .collect();
        Fixture {
            params,
            tau,
            dk,
            sks,
            ks,
            msgs,
            cts,
        }
    }

    fn random_quorum(n: usize, size: usize, rng: &mut impl Rng) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..n).collect();
        idx.shuffle(rng);
        idx.truncate(size);
        idx
    }

    fn partials(fx: &Fixture, idx: &[usize]) -> Vec<PackedPartialDecryption<E>> {
        let mut rng = test_rng();
        idx.iter()
            .map(|&i| partial_decrypt(&fx.params, &fx.sks[i], &fx.cts, &mut rng).unwrap())
            .collect()
    }

    fn assert_roundtrip(fx: &Fixture, pd_slots: &[G1]) {
        let mut rng = test_rng();
        let out = decrypt(&fx.dk, pd_slots, &fx.cts, &mut rng);
        assert_eq!(out, fx.msgs);
    }

    #[test]
    fn slot_aggregates_are_simple_bte_aggregates() {
        // ell = 3, U = 4, B = 11: the last sub-batch has 2 of 3 slots.
        let fx = fixture(PackedParams::new(16, 3, 8, 3, 11), 11);
        let mut rng = test_rng();
        let idx = random_quorum(fx.params.n, fx.params.t, &mut rng);
        let pd = combine(&fx.params, &partials(&fx, &idx));
        for s in 0..fx.params.ell {
            let mut expected = Fr::zero();
            let mut tau_pow = fx.tau;
            for u in 0..fx.params.num_subbatches {
                if let Some(k) = fx.ks.get(u * fx.params.ell + s) {
                    expected += tau_pow * k;
                }
                tau_pow *= fx.tau;
            }
            assert_eq!(pd[s], G1::generator() * expected, "slot {s}");
        }
    }

    #[test]
    fn roundtrip_random_quorum() {
        // n = 128, f = 42, t = 85, ell = 22; B = 3*ell + 5.
        let b = 3 * 22 + 5;
        let fx = fixture(PackedParams::new(128, 42, 85, 22, b), b);
        let mut rng = test_rng();
        let idx = random_quorum(fx.params.n, fx.params.t, &mut rng);
        let pds = partials(&fx, &idx);
        for pd in &pds {
            assert!(verify(&fx.dk, pd, &fx.cts));
        }
        assert_roundtrip(&fx, &combine(&fx.params, &pds));
    }

    #[test]
    fn different_quorums_agree() {
        let fx = fixture(PackedParams::new(40, 9, 20, 6, 30), 30);
        let mut rng = test_rng();
        let a = combine(&fx.params, &partials(&fx, &random_quorum(40, 20, &mut rng)));
        let b = combine(&fx.params, &partials(&fx, &random_quorum(40, 25, &mut rng)));
        assert_eq!(a, b);
        assert_roundtrip(&fx, &a);
    }

    #[test]
    fn verify_rejects_bad_shares() {
        let fx = fixture(PackedParams::new(16, 3, 8, 3, 9), 9);
        let pds = partials(&fx, &[2, 5]);
        assert!(verify(&fx.dk, &pds[0], &fx.cts));

        let mut bad = pds[0].clone();
        bad.value += G1::generator();
        assert!(!verify(&fx.dk, &bad, &fx.cts));

        // A valid share attributed to the wrong party.
        let mut wrong_index = pds[0].clone();
        wrong_index.index = pds[1].index;
        assert!(!verify(&fx.dk, &wrong_index, &fx.cts));
    }

    #[test]
    fn ell_one_is_plain_simple_bte_shape() {
        let fx = fixture(PackedParams::new(10, 3, 4, 1, 7), 7);
        assert_eq!(fx.params.num_subbatches, 7);
        let mut rng = test_rng();
        let idx = random_quorum(10, 4, &mut rng);
        assert_roundtrip(&fx, &combine(&fx.params, &partials(&fx, &idx)));
    }

    #[test]
    fn batch_smaller_than_ell() {
        let fx = fixture(PackedParams::new(20, 4, 12, 4, 16), 3);
        let mut rng = test_rng();
        let idx = random_quorum(20, 12, &mut rng);
        let pds = partials(&fx, &idx);
        assert!(verify(&fx.dk, &pds[0], &fx.cts));
        assert_roundtrip(&fx, &combine(&fx.params, &pds));
    }

    #[test]
    fn rejects_bad_ciphertext_proof() {
        let mut fx = fixture(PackedParams::new(16, 3, 8, 3, 6), 6);
        fx.cts[4].proof.response += Fr::one();
        let mut rng = test_rng();
        assert!(partial_decrypt(&fx.params, &fx.sks[0], &fx.cts, &mut rng).is_none());
    }
}
