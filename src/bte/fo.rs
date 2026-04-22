//! Fujisaki-Okamoto transform for BTE with pairing-free batch verification.
//!
//! Encrypts byte-string messages. A helper performs the expensive FFT-based
//! decryption and broadcasts compact hints `(K_i, P_i)`.  Any verifier can
//! then check correctness and recover plaintexts using only MSMs and hashes --
//! **no pairings**.

use super::decryption::CrossTerms;
use super::{DecryptionKey, EncryptionKey, PartialDecryption, SecretKey};
use ark_ec::pairing::{Pairing, PairingOutput};
use ark_ec::{AffineRepr, CurveGroup, PrimeGroup, VariableBaseMSM};
use ark_ff::{BigInteger, PrimeField, Zero};
use ark_poly::EvaluationDomain;
use ark_serialize::CanonicalSerialize;
use ark_std::rand::Rng;
use core::ops::AddAssign;
use sha2::{Digest, Sha256};

// ---------------------------------------------------------------------------
// Hash functions (random-oracle model)
// ---------------------------------------------------------------------------

/// H_K : G_T -> {0,1}^128.  Maps a target-group element to a 16-byte key.
///
/// Serializes directly into the hasher to avoid intermediate allocation.
fn h_k<E: Pairing>(p: &PairingOutput<E>) -> [u8; 16] {
    let mut hasher = Sha256::new();
    hasher.update(b"simple-bte-fo-hk");
    p.serialize_compressed(&mut hasher)
        .expect("serialization should succeed");
    let hash = hasher.finalize();
    hash[..16].try_into().unwrap()
}

/// H_R : K x {0,1}* -> F.  Derives deterministic encryption randomness.
///
/// Produces 512 bits of hash output and reduces mod the scalar field order
/// for a near-uniform distribution.
fn h_r<E: Pairing>(key: &[u8; 16], msg: &[u8]) -> E::ScalarField {
    let len_bytes = (msg.len() as u64).to_le_bytes();
    let base = Sha256::new()
        .chain_update(b"simple-bte-fo-hr")
        .chain_update(key)
        .chain_update(&len_bytes)
        .chain_update(msg);
    let h0 = base.clone().chain_update([0]).finalize();
    let h1 = base.chain_update([1]).finalize();
    let mut bytes = [0u8; 64];
    bytes[..32].copy_from_slice(&h0);
    bytes[32..].copy_from_slice(&h1);
    E::ScalarField::from_le_bytes_mod_order(&bytes)
}

/// XOR the H_M(K) keystream into `input` and return the result.
///
/// This avoids materializing the full keystream in a temporary buffer.
fn h_m_xor(key: &[u8; 16], input: &[u8]) -> Vec<u8> {
    let mut out = input.to_vec();
    let base = Sha256::new()
        .chain_update(b"simple-bte-fo-hm")
        .chain_update(key);
    let mut ctr = 0u32;
    for chunk in out.chunks_mut(32) {
        let block = base.clone().chain_update(ctr.to_le_bytes()).finalize();
        for (dst, src) in chunk.iter_mut().zip(block.iter()) {
            *dst ^= *src;
        }
        ctr += 1;
    }
    out
}

fn xor_16(a: &[u8; 16], b: &[u8; 16]) -> [u8; 16] {
    core::array::from_fn(|i| a[i] ^ b[i])
}

// ---------------------------------------------------------------------------
// Small-scalar Pippenger MSM
// ---------------------------------------------------------------------------

/// Number of bits in the random batch-verification challenges.
/// 128 bits gives ~100-bit statistical soundness after accounting for the
/// birthday bound and is sufficient for all practical security levels.
const CHALLENGE_BITS: usize = 128;

fn ln_without_floats(a: usize) -> usize {
    let log2a = (usize::BITS - a.leading_zeros()) as usize;
    log2a * 69 / 100
}

/// Pippenger MSM with configurable scalar bit-length.
///
/// The default arkworks MSM processes all ~255 bits of each scalar because it
/// cannot assume they are small.  When the scalars are known to fit in
/// `scalar_bits` bits (e.g. 128-bit random challenges), passing that value
/// here halves the number of Pippenger windows and yields a ~2x speedup.
fn msm_small<G, B, F>(bases: &[B], scalars: &[F::BigInt], scalar_bits: usize) -> G
where
    G: Copy + Zero + AddAssign<G> + for<'a> AddAssign<&'a B>,
    F: PrimeField,
{
    let size = bases.len().min(scalars.len());
    if size == 0 || scalar_bits == 0 {
        return G::zero();
    }

    let c = if size < 32 {
        3
    } else {
        ln_without_floats(size) + 2
    };
    let num_buckets = (1 << c) - 1;
    let mut buckets = vec![G::zero(); num_buckets];
    let mut total = G::zero();
    let mut first = true;

    for w_start in (0..scalar_bits).step_by(c).rev() {
        if !first {
            for _ in 0..c {
                total += total;
            }
        }
        first = false;

        buckets.fill(G::zero());
        for (scalar, base) in scalars[..size].iter().zip(&bases[..size]) {
            if scalar.is_zero() {
                continue;
            }
            let mut s = *scalar;
            s >>= w_start as u32;
            let idx = (s.as_ref()[0] & ((1u64 << c) - 1)) as usize;
            if idx != 0 {
                buckets[idx - 1] += base;
            }
        }

        let mut running_sum = G::zero();
        let mut window_sum = G::zero();
        for &bucket in buckets.iter().rev() {
            running_sum += bucket;
            window_sum += running_sum;
        }
        total += window_sum;
    }

    total
}

// ---------------------------------------------------------------------------
// Helper: recover pairing keys
// ---------------------------------------------------------------------------

/// Recover the per-slot pairing keys `z_i = [k_i * tau^{B+1}]_T` from the
/// combined partial decryption and pre-computed cross-terms.
fn recover_pairing_keys<E: Pairing>(
    dk: &DecryptionKey<E>,
    pd: &E::G1,
    cross: &CrossTerms<E>,
) -> Vec<PairingOutput<E>> {
    let b = dk.batch_size;
    (0..b)
        .map(|i| {
            let pd_term = E::pairing(*pd, dk.powers_of_h[b - i].clone());
            pd_term - cross.values[i]
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// FO-transformed BTE ciphertext for a single byte-string message.
///
/// ```text
/// ct = ( [k]_1,  H_K(k * ek) xor K,  H_M(K) xor msg )
/// ```
#[derive(Clone, Debug)]
pub struct FoCiphertext<E: Pairing> {
    /// [k]_1 -- the same G1 element used by the underlying BTE.
    pub ct0: E::G1Affine,
    /// H_K(k * ek) xor K -- 16-byte encrypted symmetric key.
    pub ct1: [u8; 16],
    /// H_M(K) xor msg -- encrypted payload.
    pub ct2: Vec<u8>,
}

/// Hints broadcast by the helper after performing the expensive decryption.
#[derive(Clone, Debug)]
pub struct DecryptionHints<E: Pairing> {
    /// K_i -- the per-ciphertext symmetric key.
    pub keys: Vec<[u8; 16]>,
    /// P_i = [k_i * tau^{B+1}]_T -- the per-ciphertext pairing key.
    pub pairing_values: Vec<PairingOutput<E>>,
}

// ---------------------------------------------------------------------------
// Encryption
// ---------------------------------------------------------------------------

/// Encrypt a byte-string message under the FO-transformed BTE scheme.
///
/// Ciphertext overhead vs. plaintext: |G1| + 16 bytes
/// (48 + 16 = 64 bytes for BLS12-381 compressed).
pub fn encrypt<E: Pairing>(
    ek: &EncryptionKey<E>,
    msg: &[u8],
    rng: &mut impl Rng,
) -> FoCiphertext<E> {
    let mut key = [0u8; 16];
    rng.fill(&mut key);

    let k = h_r::<E>(&key, msg);
    let ct0 = (E::G1::generator() * k).into_affine();
    let p = ek.e * k;
    let ct1 = xor_16(&h_k::<E>(&p), &key);
    let ct2 = h_m_xor(&key, msg);

    FoCiphertext { ct0, ct1, ct2 }
}

// ---------------------------------------------------------------------------
// Partial decryption (validator side -- same as base BTE)
// ---------------------------------------------------------------------------

/// Compute a partial decryption share for a batch of FO ciphertexts.
///
/// Unlike the base BTE there is no Schnorr proof to check; the FO transform
/// provides integrity at verification time instead.
pub fn partial_decrypt<E: Pairing>(
    sk: &SecretKey<E>,
    cts: &[FoCiphertext<E>],
) -> PartialDecryption<E> {
    assert_eq!(
        cts.len(),
        sk.shares.len(),
        "batch size must match number of shares"
    );
    let bases: Vec<_> = cts.iter().map(|ct| ct.ct0).collect();
    let value = E::G1::msm(&bases, &sk.shares).unwrap();
    PartialDecryption {
        value,
        party_index: sk.party_index,
    }
}

/// Verify a partial decryption share against the public verification keys.
///
/// Checks e(pd_j, g_2) = sum_i e(ct_{i,0}, v^i_j).
pub fn verify_partial_decryption<E: Pairing>(
    dk: &DecryptionKey<E>,
    pd: &PartialDecryption<E>,
    cts: &[FoCiphertext<E>],
) -> bool {
    assert_eq!(cts.len(), dk.batch_size);
    let party_idx = pd.party_index - 1;
    let lhs = E::pairing(pd.value, E::G2::generator());
    let rhs = E::multi_pairing(
        cts.iter().map(|ct| ct.ct0),
        dk.verification_keys[party_idx].iter().cloned(),
    );
    lhs == rhs
}

/// Combine t partial decryption shares via Lagrange interpolation at x = 0.
///
/// Re-exported from the base module; works identically for FO ciphertexts.
pub use super::decryption::combine;

// ---------------------------------------------------------------------------
// Helper: expensive decryption + hint generation
// ---------------------------------------------------------------------------

/// Pre-compute cross-terms via FFT (pipelineable -- does **not** need `pd`).
///
/// Equivalent to `decryption::predecrypt_fft` but operates on FO ciphertexts.
pub fn predecrypt_fft<E: Pairing>(dk: &DecryptionKey<E>, cts: &[FoCiphertext<E>]) -> CrossTerms<E> {
    let b = dk.batch_size;
    let n = dk.fft_size;
    assert_eq!(cts.len(), b);

    let mut a_vec: Vec<E::G1> = cts.iter().map(|ct| ct.ct0.into_group()).collect();
    a_vec.resize(n, E::G1::zero());

    dk.fft_domain.fft_in_place(&mut a_vec);
    let a_hat = E::G1::normalize_batch(&a_vec);

    let mut c_hat: Vec<PairingOutput<E>> = a_hat
        .iter()
        .zip(dk.fft_h.iter())
        .map(|(a, b)| E::pairing(*a, b.clone()))
        .collect();

    dk.fft_domain.ifft_in_place(&mut c_hat);

    CrossTerms { values: c_hat }
}

/// Perform the expensive FFT decryption and produce compact hints.
///
/// Returns `(messages, hints)` where `hints` can be broadcast to any verifier
/// for cheap pairing-free batch verification.
///
/// Equivalent to calling `predecrypt_fft` then `helper_finalize`.
pub fn helper_decrypt<E: Pairing>(
    dk: &DecryptionKey<E>,
    pd: &E::G1,
    cts: &[FoCiphertext<E>],
) -> (Vec<Vec<u8>>, DecryptionHints<E>) {
    let cross = predecrypt_fft(dk, cts);
    helper_finalize(dk, pd, cts, &cross)
}

/// Finalize decryption given `pd` and pre-computed cross-terms.  Returns
/// `(messages, hints)`.
pub fn helper_finalize<E: Pairing>(
    dk: &DecryptionKey<E>,
    pd: &E::G1,
    cts: &[FoCiphertext<E>],
    cross: &CrossTerms<E>,
) -> (Vec<Vec<u8>>, DecryptionHints<E>) {
    let b = dk.batch_size;
    assert_eq!(cts.len(), b);

    let pairing_values = recover_pairing_keys(dk, pd, cross);

    let mut keys = Vec::with_capacity(b);
    let mut messages = Vec::with_capacity(b);
    for i in 0..b {
        let key = xor_16(&h_k::<E>(&pairing_values[i]), &cts[i].ct1);
        let msg = h_m_xor(&key, &cts[i].ct2);
        keys.push(key);
        messages.push(msg);
    }

    (
        messages,
        DecryptionHints {
            keys,
            pairing_values,
        },
    )
}

// ---------------------------------------------------------------------------
// Batch verification (verifier side -- no pairings!)
// ---------------------------------------------------------------------------

/// Verify decryption hints and recover all plaintexts.
///
/// Cost: one G1-MSM, one G_T-MSM, O(B) field multiplications, O(B) hashes.
/// **No pairings.**
///
/// Returns `Some(messages)` if all checks pass, `None` otherwise.
pub fn batch_verify<E: Pairing>(
    ek: &EncryptionKey<E>,
    cts: &[FoCiphertext<E>],
    hints: &DecryptionHints<E>,
    rng: &mut impl Rng,
) -> Option<Vec<Vec<u8>>> {
    let b = cts.len();
    assert_eq!(hints.keys.len(), b);
    assert_eq!(hints.pairing_values.len(), b);

    // Step 1: Per-ciphertext consistency -- ct_{i,1} == H_K(P_i) xor K_i.
    // Cheapest check; done first for early exit on trivially bad hints.
    for i in 0..b {
        if cts[i].ct1 != xor_16(&h_k::<E>(&hints.pairing_values[i]), &hints.keys[i]) {
            return None;
        }
    }

    // Step 2: For each i, recover msg_i and re-derive k_i.
    let mut messages = Vec::with_capacity(b);
    let mut scalars = Vec::with_capacity(b);
    for i in 0..b {
        let msg_i = h_m_xor(&hints.keys[i], &cts[i].ct2);
        let k_i = h_r::<E>(&hints.keys[i], &msg_i);
        messages.push(msg_i);
        scalars.push(k_i);
    }

    // Sample 128-bit random challenges.  Using CHALLENGE_BITS-bit scalars lets
    // msm_small skip roughly half the Pippenger windows vs full-size scalars.
    let mut r_bigints = Vec::with_capacity(b);
    let mut rk_sum = E::ScalarField::zero();
    for k_i in &scalars {
        let mut bytes = [0u8; 16];
        rng.fill(&mut bytes);
        let r_i = E::ScalarField::from_le_bytes_mod_order(&bytes);
        rk_sum += r_i * *k_i;
        r_bigints.push(r_i.into_bigint());
    }

    // Step 3a: G1 check -- MSM(ct0s, r) == rk_sum * g1
    let g1_bases: Vec<_> = cts.iter().map(|ct| ct.ct0).collect();
    let g1_lhs =
        msm_small::<E::G1, E::G1Affine, E::ScalarField>(&g1_bases, &r_bigints, CHALLENGE_BITS);
    if g1_lhs != E::G1::generator() * rk_sum {
        return None;
    }

    // Step 3b: G_T check -- MSM(P_i, r) == rk_sum * ek
    let gt_lhs = msm_small::<PairingOutput<E>, PairingOutput<E>, E::ScalarField>(
        &hints.pairing_values,
        &r_bigints,
        CHALLENGE_BITS,
    );
    if gt_lhs != ek.e * rk_sum {
        return None;
    }

    Some(messages)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bte::crs::setup;
    use ark_bls12_381::Bls12_381;
    use ark_std::test_rng;

    type E = Bls12_381;

    #[test]
    fn fo_roundtrip() {
        let mut rng = test_rng();
        let batch_size = 8;
        let num_parties = 8;
        let threshold = 5;

        let (ek, dk, sks) = setup::<E>(batch_size, num_parties, threshold, &mut rng);

        let messages: Vec<Vec<u8>> = (0..batch_size)
            .map(|i| format!("message number {i}").into_bytes())
            .collect();

        let cts: Vec<_> = messages.iter().map(|m| encrypt(&ek, m, &mut rng)).collect();

        let pds: Vec<_> = sks[..threshold]
            .iter()
            .map(|sk| partial_decrypt(sk, &cts))
            .collect();

        for pd in &pds {
            assert!(
                verify_partial_decryption(&dk, pd, &cts),
                "share verification failed for party {}",
                pd.party_index,
            );
        }

        let pd = combine::<E>(&pds);

        let (helper_msgs, hints) = helper_decrypt(&dk, &pd, &cts);
        for i in 0..batch_size {
            assert_eq!(
                helper_msgs[i], messages[i],
                "helper decrypt mismatch at {i}"
            );
        }

        let verified = batch_verify(&ek, &cts, &hints, &mut rng);
        let verified_msgs = verified.expect("batch verification should succeed");
        for i in 0..batch_size {
            assert_eq!(
                verified_msgs[i], messages[i],
                "verified message mismatch at {i}"
            );
        }
    }

    #[test]
    fn fo_roundtrip_pipelined() {
        let mut rng = test_rng();
        let batch_size = 16;
        let num_parties = 10;
        let threshold = 7;

        let (ek, dk, sks) = setup::<E>(batch_size, num_parties, threshold, &mut rng);

        let messages: Vec<Vec<u8>> = (0..batch_size).map(|i| vec![i as u8; 100]).collect();

        let cts: Vec<_> = messages.iter().map(|m| encrypt(&ek, m, &mut rng)).collect();

        let pds: Vec<_> = sks[..threshold]
            .iter()
            .map(|sk| partial_decrypt(sk, &cts))
            .collect();

        let pd = combine::<E>(&pds);

        let cross = predecrypt_fft(&dk, &cts);
        let (helper_msgs, hints) = helper_finalize(&dk, &pd, &cts, &cross);
        for i in 0..batch_size {
            assert_eq!(helper_msgs[i], messages[i]);
        }

        let verified_msgs =
            batch_verify(&ek, &cts, &hints, &mut rng).expect("batch verification should succeed");
        for i in 0..batch_size {
            assert_eq!(verified_msgs[i], messages[i]);
        }
    }

    #[test]
    fn fo_rejects_bad_key_hint() {
        let mut rng = test_rng();
        let batch_size = 4;
        let num_parties = 4;
        let threshold = 3;

        let (ek, dk, sks) = setup::<E>(batch_size, num_parties, threshold, &mut rng);

        let messages: Vec<Vec<u8>> = (0..batch_size)
            .map(|i| format!("msg {i}").into_bytes())
            .collect();

        let cts: Vec<_> = messages.iter().map(|m| encrypt(&ek, m, &mut rng)).collect();

        let pds: Vec<_> = sks[..threshold]
            .iter()
            .map(|sk| partial_decrypt(sk, &cts))
            .collect();

        let pd = combine::<E>(&pds);
        let (_, mut hints) = helper_decrypt(&dk, &pd, &cts);

        hints.keys[0][0] ^= 0xff;

        assert!(
            batch_verify(&ek, &cts, &hints, &mut rng).is_none(),
            "should reject corrupted K hint"
        );
    }

    #[test]
    fn fo_rejects_bad_pairing_hint() {
        let mut rng = test_rng();
        let batch_size = 4;
        let num_parties = 4;
        let threshold = 3;

        let (ek, dk, sks) = setup::<E>(batch_size, num_parties, threshold, &mut rng);

        let messages: Vec<Vec<u8>> = (0..batch_size)
            .map(|i| format!("msg {i}").into_bytes())
            .collect();

        let cts: Vec<_> = messages.iter().map(|m| encrypt(&ek, m, &mut rng)).collect();

        let pds: Vec<_> = sks[..threshold]
            .iter()
            .map(|sk| partial_decrypt(sk, &cts))
            .collect();

        let pd = combine::<E>(&pds);
        let (_, mut hints) = helper_decrypt(&dk, &pd, &cts);

        hints.pairing_values[1] += PairingOutput::<E>::generator();

        assert!(
            batch_verify(&ek, &cts, &hints, &mut rng).is_none(),
            "should reject corrupted P hint"
        );
    }

    #[test]
    fn fo_rejects_bad_partial_decryption() {
        let mut rng = test_rng();
        let batch_size = 4;
        let num_parties = 4;
        let threshold = 3;

        let (ek, dk, sks) = setup::<E>(batch_size, num_parties, threshold, &mut rng);

        let messages: Vec<Vec<u8>> = (0..batch_size)
            .map(|i| format!("msg {i}").into_bytes())
            .collect();

        let cts: Vec<_> = messages.iter().map(|m| encrypt(&ek, m, &mut rng)).collect();

        let mut bad_pd = partial_decrypt(&sks[0], &cts);
        bad_pd.value += <E as Pairing>::G1::generator();

        assert!(
            !verify_partial_decryption(&dk, &bad_pd, &cts),
            "should reject corrupted partial decryption"
        );
    }

    #[test]
    fn fo_empty_messages() {
        let mut rng = test_rng();
        let batch_size = 4;
        let num_parties = 4;
        let threshold = 3;

        let (ek, dk, sks) = setup::<E>(batch_size, num_parties, threshold, &mut rng);

        let messages: Vec<Vec<u8>> = vec![vec![]; batch_size];

        let cts: Vec<_> = messages.iter().map(|m| encrypt(&ek, m, &mut rng)).collect();

        let pds: Vec<_> = sks[..threshold]
            .iter()
            .map(|sk| partial_decrypt(sk, &cts))
            .collect();
        let pd = combine::<E>(&pds);

        let (_, hints) = helper_decrypt(&dk, &pd, &cts);
        let verified =
            batch_verify(&ek, &cts, &hints, &mut rng).expect("empty messages should verify");
        for i in 0..batch_size {
            assert!(verified[i].is_empty());
        }
    }

    #[test]
    fn fo_variable_length_messages() {
        let mut rng = test_rng();
        let batch_size = 8;
        let num_parties = 8;
        let threshold = 5;

        let (ek, dk, sks) = setup::<E>(batch_size, num_parties, threshold, &mut rng);

        let messages: Vec<Vec<u8>> = (0..batch_size).map(|i| vec![0xab; i * 50]).collect();

        let cts: Vec<_> = messages.iter().map(|m| encrypt(&ek, m, &mut rng)).collect();

        let pds: Vec<_> = sks[..threshold]
            .iter()
            .map(|sk| partial_decrypt(sk, &cts))
            .collect();
        let pd = combine::<E>(&pds);

        let (_, hints) = helper_decrypt(&dk, &pd, &cts);
        let verified = batch_verify(&ek, &cts, &hints, &mut rng)
            .expect("variable-length messages should verify");
        for i in 0..batch_size {
            assert_eq!(verified[i], messages[i]);
        }
    }
}
