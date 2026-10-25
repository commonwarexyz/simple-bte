use super::{schnorr_challenge, Ciphertext, DecryptionKey, PartialDecryption, SecretKey};
use ark_ec::pairing::{Pairing, PairingOutput};
use ark_ec::{AffineRepr, CurveGroup, PrimeGroup, VariableBaseMSM};
use ark_ff::{batch_inversion, One, Zero};
use ark_poly::EvaluationDomain;
use ark_std::{rand::Rng, UniformRand};

/// Verify all Schnorr proofs attached to a ciphertext batch with one
/// randomized linear-combination check.
pub fn verify_ciphertext_batch<E: Pairing>(cts: &[Ciphertext<E>], rng: &mut impl Rng) -> bool {
    if cts.is_empty() {
        return true;
    }

    let mut lhs_scalar = E::ScalarField::zero();
    let mut rhs_bases = Vec::with_capacity(2 * cts.len());
    let mut rhs_scalars = Vec::with_capacity(2 * cts.len());

    for ct in cts {
        // Reject the trivial key k = 0: an attacker can otherwise craft a valid
        // proof with ct1 = identity and ct2 = m, publishing plaintext without
        // any threshold consensus. This is a safety requirement, not a
        // performance optimization: the Schnorr transcript alone accepts ct1 = O.
        if ct.ct1.is_zero() {
            return false;
        }
        let weight = E::ScalarField::rand(rng);
        let challenge = schnorr_challenge::<E>(&ct.ct1, &ct.ct2, &ct.proof.commitment);
        lhs_scalar += weight * ct.proof.response;
        rhs_bases.push(ct.proof.commitment);
        rhs_scalars.push(weight);
        rhs_bases.push(ct.ct1);
        rhs_scalars.push(weight * challenge);
    }

    let lhs = E::G1::generator() * lhs_scalar;
    let rhs = E::G1::msm(&rhs_bases, &rhs_scalars).unwrap();
    lhs == rhs
}

/// Compute a partial decryption share for a batch of ciphertexts.
///
/// pd_j = sum_{i in [B]} sigma^i_j * ct_{i,1}
pub fn partial_decrypt<E: Pairing>(
    sk: &SecretKey<E>,
    cts: &[Ciphertext<E>],
    rng: &mut impl Rng,
) -> Option<PartialDecryption<E>> {
    assert_eq!(
        cts.len(),
        sk.shares.len(),
        "batch size must match number of shares"
    );
    if !verify_ciphertext_batch(cts, rng) {
        return None;
    }
    let bases: Vec<_> = cts.iter().map(|ct| ct.ct1).collect();
    let value = E::G1::msm(&bases, &sk.shares).unwrap();
    Some(PartialDecryption {
        value,
        party_index: sk.party_index,
    })
}

/// Verify a partial decryption share against the public verification keys.
///
/// Assumes `verify_ciphertext_batch(cts)` has already succeeded.
///
/// Checks: e(pd_j, g_2) = sum_{i in [B]} e(ct_{i,1}, v^i_j)
///
/// Never panics on attacker-controlled input: malformed shares or misleading
/// batch sizes yield `false` instead of an out-of-bounds panic (DoS).
pub fn verify<E: Pairing>(
    dk: &DecryptionKey<E>,
    pd: &PartialDecryption<E>,
    cts: &[Ciphertext<E>],
) -> bool {
    if cts.len() != dk.batch_size {
        return false;
    }
    // The share index is 1-based and must address a registered party;
    // `party_index - 1` underflows for index 0 and reads out of bounds for
    // any index above the number of parties.
    if pd.party_index == 0 || pd.party_index > dk.num_parties {
        return false;
    }
    let party_idx = pd.party_index - 1; // 0-based for array access

    let lhs = E::pairing(pd.value, E::G2::generator());
    let rhs = E::multi_pairing(
        cts.iter().map(|ct| ct.ct1),
        dk.verification_keys[party_idx].iter().cloned(),
    );

    lhs == rhs
}

/// Combine t partial decryption shares via Lagrange interpolation at x = 0.
///
/// pd = sum_{j in T} lambda_j * pd_j   where  lambda_j = prod_{l in T, l!=j} l/(l-j)
///
/// Returns `None` when the input set is empty or contains duplicate party
/// indices; in either case the Lagrange weights below are undefined (zero
/// denominators) and any resulting share would be meaningless.
pub fn combine<E: Pairing>(partial_decs: &[PartialDecryption<E>]) -> Option<E::G1> {
    let t = partial_decs.len();
    if t == 0 {
        return None;
    }
    // A duplicated party index collapses two columns of the interpolation
    // denominator to zero; produce an error instead of a wrong combined key.
    for i in 0..t {
        for j in (i + 1)..t {
            if partial_decs[i].party_index == partial_decs[j].party_index {
                return None;
            }
        }
    }
    let indices: Vec<E::ScalarField> = partial_decs
        .iter()
        .map(|pd| E::ScalarField::from(pd.party_index as u64))
        .collect();

    // Collect all (x_j - x_i) differences and batch-invert in one shot
    // (1 field inversion + 3(t^2-t-1) multiplications instead of t(t-1) inversions).
    let mut diffs: Vec<E::ScalarField> = Vec::with_capacity(t * (t - 1));
    for i in 0..t {
        for j in 0..t {
            if i != j {
                diffs.push(indices[j] - indices[i]);
            }
        }
    }
    batch_inversion(&mut diffs);

    let mut lambdas: Vec<E::ScalarField> = Vec::with_capacity(t);
    let mut idx = 0;
    for i in 0..t {
        let mut lambda = E::ScalarField::one();
        for j in 0..t {
            if i != j {
                lambda *= indices[j] * diffs[idx];
                idx += 1;
            }
        }
        lambdas.push(lambda);
    }

    let base_points: Vec<_> = partial_decs.iter().map(|pd| pd.value).collect();
    let bases = E::G1::normalize_batch(&base_points);
    let combined = E::G1::msm(&bases, &lambdas).unwrap();
    Some(combined)
}

/// Decrypt a batch of ciphertexts given the combined pre-decryption key.
///
/// Assumes the caller wants ciphertext proof verification as part of decryption.
///
/// For each i in [B] (0-based):
///   z_i = e(pd, h_{B-i}) - sum_{l!=i} e(ct_l.ct1, h_{l-i+B+1})
///   m_i = ct_i.ct2 - z_i
///
/// Runs in O(B^2) pairings (uses multi_pairing per row).
///
/// Returns `None` when the batch size does not match the decryption key or
/// when any ciphertext Schnorr proof fails: a malformed batch is a caller
/// error, not a panic condition.
pub fn decrypt<E: Pairing>(
    dk: &DecryptionKey<E>,
    pd: &E::G1,
    cts: &[Ciphertext<E>],
    rng: &mut impl Rng,
) -> Option<Vec<PairingOutput<E>>> {
    let b = dk.batch_size;
    if cts.len() != b {
        return None;
    }
    if !verify_ciphertext_batch(cts, rng) {
        return None;
    }

    let rows: Vec<PairingOutput<E>> = (0..b)
        .map(|i| {
            let pd_term = E::pairing(*pd, dk.powers_of_h[b - i].clone());
            let cross = E::multi_pairing(
                (0..b).filter(|&l| l != i).map(|l| cts[l].ct1),
                (0..b)
                    .filter(|&l| l != i)
                    .map(|l| dk.powers_of_h[l + b + 1 - i].clone()),
            );

            let z_i = pd_term - cross;
            cts[i].ct2 - z_i
        })
        .collect();
    Some(rows)
}

/// Cross-terms computed during the pre-decryption phase. This intermediate
/// result depends only on the ciphertexts and dk, not on the combined partial
/// decryption, so it can be computed while PartialDec/Verify/Combine are still
/// in flight.
pub struct CrossTerms<E: Pairing> {
    pub values: Vec<PairingOutput<E>>,
}

/// **Pre-decryption phase** (pipelineable): compute the cross-term vector C[i]
/// using FFT convolution.  Costs O(B log B) group ops (G1-FFT + G_T-iFFT) plus
/// 2B pairings in the frequency domain.  Does *not* require `pd`.
pub fn predecrypt_fft<E: Pairing>(dk: &DecryptionKey<E>, cts: &[Ciphertext<E>]) -> CrossTerms<E> {
    let b = dk.batch_size;
    let n = dk.fft_size;
    assert_eq!(cts.len(), b);

    // 1. Build a_vec = [ct_0.ct1, ..., ct_{B-1}.ct1, 0, ..., 0] in G1.
    let mut a_vec: Vec<E::G1> = cts.iter().map(|ct| ct.ct1.into_group()).collect();
    a_vec.resize(n, E::G1::zero());

    // 2. FFT(a_vec) in G1.
    dk.fft_domain.fft_in_place(&mut a_vec);
    let a_hat = E::G1::normalize_batch(&a_vec);

    // 3. Pointwise pairing: c_hat[k] = e(a_hat[k], fft_h[k]).  (2B pairings)
    let mut c_hat: Vec<PairingOutput<E>> = a_hat
        .iter()
        .zip(dk.fft_h.iter())
        .map(|(a, b)| E::pairing(*a, b.clone()))
        .collect();

    // 4. iFFT(c_hat) in G_T -> C[i] = cross-term for row i.
    dk.fft_domain.ifft_in_place(&mut c_hat);

    CrossTerms { values: c_hat }
}

/// **Finalization phase**: given the combined partial decryption `pd` and the
/// pre-computed cross-terms, recover each plaintext.  Costs B pairings + B
/// group ops in G_T.
pub fn finalize_decrypt<E: Pairing>(
    dk: &DecryptionKey<E>,
    pd: &E::G1,
    cts: &[Ciphertext<E>],
    cross: &CrossTerms<E>,
) -> Vec<PairingOutput<E>> {
    let b = dk.batch_size;
    assert_eq!(cts.len(), b);

    (0..b)
        .map(|i| {
            let pd_term = E::pairing(*pd, dk.powers_of_h[b - i].clone());
            let z_i = pd_term - cross.values[i];
            cts[i].ct2 - z_i
        })
        .collect()
}

/// FFT-accelerated decryption: same result as `decrypt` but O(B log B) group
/// operations + O(B) pairings instead of O(B^2) pairings.
///
/// Equivalent to calling `predecrypt_fft` then `finalize_decrypt`.
///
/// Returns `None` under the same conditions as `decrypt`.
pub fn decrypt_fft<E: Pairing>(
    dk: &DecryptionKey<E>,
    pd: &E::G1,
    cts: &[Ciphertext<E>],
    rng: &mut impl Rng,
) -> Option<Vec<PairingOutput<E>>> {
    if cts.len() != dk.batch_size || !verify_ciphertext_batch(cts, rng) {
        return None;
    }

    let cross = predecrypt_fft(dk, cts);
    Some(finalize_decrypt(dk, pd, cts, &cross))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bte::crs::setup;
    use crate::bte::encryption::encrypt;
    use ark_bls12_381::Bls12_381;
    use ark_ec::pairing::PairingOutput;
    use ark_std::{test_rng, UniformRand};

    type E = Bls12_381;

    #[test]
    fn test_bte_roundtrip() {
        let mut rng = test_rng();
        let batch_size = 8;
        let num_parties = 8;
        let threshold = 5;

        let (ek, dk, sks) = setup::<E>(batch_size, num_parties, threshold, &mut rng);

        let messages: Vec<PairingOutput<E>> = (0..batch_size)
            .map(|_| PairingOutput::<E>::generator() * <E as Pairing>::ScalarField::rand(&mut rng))
            .collect();

        let cts: Vec<_> = messages.iter().map(|m| encrypt(&ek, m, &mut rng)).collect();
        assert!(verify_ciphertext_batch(&cts, &mut rng));

        // Partial decryptions from the first `threshold` parties.
        let pds: Vec<_> = sks[..threshold]
            .iter()
            .map(|sk| partial_decrypt(sk, &cts, &mut rng).expect("valid ciphertext proofs"))
            .collect();

        for pd in &pds {
            assert!(
                verify(&dk, pd, &cts),
                "verify failed for party {}",
                pd.party_index
            );
        }

        let pd = combine::<E>(&pds).expect("valid distinct shares");

        // Naive O(B^2) decrypt.
        let recovered = decrypt(&dk, &pd, &cts, &mut rng).expect("valid ciphertext batch");
        for i in 0..batch_size {
            assert_eq!(recovered[i], messages[i], "naive decrypt mismatch at {i}");
        }

        // FFT O(B log B) decrypt.
        let recovered_fft = decrypt_fft(&dk, &pd, &cts, &mut rng).expect("valid ciphertext batch");
        for i in 0..batch_size {
            assert_eq!(recovered_fft[i], messages[i], "FFT decrypt mismatch at {i}");
        }

        // Pipelined: predecrypt_fft + finalize_decrypt.
        let cross = predecrypt_fft(&dk, &cts);
        let recovered_pipelined = finalize_decrypt(&dk, &pd, &cts, &cross);
        for i in 0..batch_size {
            assert_eq!(
                recovered_pipelined[i], messages[i],
                "pipelined decrypt mismatch at {i}"
            );
        }
    }

    #[test]
    fn test_fft_matches_naive() {
        let mut rng = test_rng();
        let batch_size = 16;
        let num_parties = 10;
        let threshold = 7;

        let (ek, dk, sks) = setup::<E>(batch_size, num_parties, threshold, &mut rng);

        let messages: Vec<PairingOutput<E>> = (0..batch_size)
            .map(|_| PairingOutput::<E>::generator() * <E as Pairing>::ScalarField::rand(&mut rng))
            .collect();

        let cts: Vec<_> = messages.iter().map(|m| encrypt(&ek, m, &mut rng)).collect();
        assert!(verify_ciphertext_batch(&cts, &mut rng));

        let pds: Vec<_> = sks[..threshold]
            .iter()
            .map(|sk| partial_decrypt(sk, &cts, &mut rng).expect("valid ciphertext proofs"))
            .collect();

        let pd = combine::<E>(&pds).expect("valid distinct shares");

        let naive = decrypt(&dk, &pd, &cts, &mut rng).expect("valid ciphertext batch");
        let fft = decrypt_fft(&dk, &pd, &cts, &mut rng).expect("valid ciphertext batch");

        for i in 0..batch_size {
            assert_eq!(naive[i], fft[i], "naive vs FFT mismatch at {i}");
        }
    }

    #[test]
    fn test_verify_rejects_bad_share() {
        let mut rng = test_rng();
        let batch_size = 4;
        let num_parties = 4;
        let threshold = 3;

        let (ek, dk, sks) = setup::<E>(batch_size, num_parties, threshold, &mut rng);

        let messages: Vec<PairingOutput<E>> = (0..batch_size)
            .map(|_| PairingOutput::<E>::generator() * <E as Pairing>::ScalarField::rand(&mut rng))
            .collect();

        let cts: Vec<_> = messages.iter().map(|m| encrypt(&ek, m, &mut rng)).collect();
        assert!(verify_ciphertext_batch(&cts, &mut rng));

        let mut bad_pd = partial_decrypt(&sks[0], &cts, &mut rng).expect("valid ciphertext proofs");
        // Corrupt the partial decryption.
        bad_pd.value += <E as Pairing>::G1::generator();

        assert!(!verify(&dk, &bad_pd, &cts));
    }

    #[test]
    fn test_verify_rejects_bad_proof() {
        let mut rng = test_rng();
        let batch_size = 4;
        let num_parties = 4;
        let threshold = 3;

        let (ek, _dk, sks) = setup::<E>(batch_size, num_parties, threshold, &mut rng);
        let messages: Vec<PairingOutput<E>> = (0..batch_size)
            .map(|_| PairingOutput::<E>::generator() * <E as Pairing>::ScalarField::rand(&mut rng))
            .collect();

        let mut cts: Vec<_> = messages.iter().map(|m| encrypt(&ek, m, &mut rng)).collect();
        cts[0].proof.response += <E as Pairing>::ScalarField::one();

        assert!(!verify_ciphertext_batch(&cts, &mut rng));
        assert!(partial_decrypt(&sks[0], &cts, &mut rng).is_none());
    }

    #[test]
    fn test_verify_rejects_out_of_range_party_index() {
        let mut rng = test_rng();
        let (ek, dk, sks) = setup::<E>(4, 4, 3, &mut rng);

        let messages: Vec<PairingOutput<E>> = (0..4)
            .map(|_| PairingOutput::<E>::generator() * <E as Pairing>::ScalarField::rand(&mut rng))
            .collect();
        let cts: Vec<_> = messages.iter().map(|m| encrypt(&ek, m, &mut rng)).collect();

        let real_pd = partial_decrypt(&sks[0], &cts, &mut rng).expect("valid ciphertext proofs");

        // party_index = 0 would underflow to usize::MAX in the old code.
        let zero_pd = PartialDecryption {
            value: real_pd.value,
            party_index: 0,
        };
        assert!(!verify(&dk, &zero_pd, &cts));

        // party_index beyond the registered party count read out of bounds.
        let bogus_pd = PartialDecryption {
            value: real_pd.value,
            party_index: 5,
        };
        assert!(!verify(&dk, &bogus_pd, &cts));

        // Mismatched batch length must verify false, not panic.
        assert!(!verify(&dk, &real_pd, &cts[..3]));
    }

    #[test]
    fn test_combine_rejects_empty_and_duplicate_shares() {
        let mut rng = test_rng();
        let (ek, _dk, sks) = setup::<E>(4, 4, 3, &mut rng);

        let messages: Vec<PairingOutput<E>> = (0..4)
            .map(|_| PairingOutput::<E>::generator() * <E as Pairing>::ScalarField::rand(&mut rng))
            .collect();
        let cts: Vec<_> = messages.iter().map(|m| encrypt(&ek, m, &mut rng)).collect();

        let pds: Vec<_> = sks[..3]
            .iter()
            .map(|sk| partial_decrypt(sk, &cts, &mut rng).expect("valid ciphertext proofs"))
            .collect();

        // Empty input has no defined interpolation.
        assert!(combine::<E>(&[]).is_none());

        // Duplicated party indices collapse Lagrange denominators to zero.
        let duplicated = [pds[0].clone(), pds[0].clone()];
        assert!(combine::<E>(&duplicated).is_none());

        // The honest path must still succeed.
        assert!(combine::<E>(&pds).is_some());
    }

    #[test]
    fn test_verify_ciphertext_batch_rejects_trivial_key() {
        let mut rng = test_rng();
        let (ek, _dk, _sks) = setup::<E>(4, 4, 3, &mut rng);

        // Forge a ciphertext with the trivial key k = 0: ct1 = identity and
        // ct2 = m, with a Schnorr commitment/response pair that satisfies the
        // verification equation and would pass without the identity check.
        let message = PairingOutput::<E>::generator() * <E as Pairing>::ScalarField::rand(&mut rng);
        let r = <E as Pairing>::ScalarField::rand(&mut rng);
        let trivial_ct = Ciphertext::<E> {
            ct1: <E as Pairing>::G1Affine::identity(),
            ct2: message,
            proof: crate::bte::SchnorrProof {
                commitment: (<E as Pairing>::G1::generator() * r).into_affine(),
                response: r,
            },
        };
        assert!(!verify_ciphertext_batch(&[trivial_ct], &mut rng));
        assert!(encrypt(&ek, &message, &mut rng).ct1 != <E as Pairing>::G1Affine::identity());
    }

    #[test]
    fn test_decrypt_rejects_invalid_batch() {
        let mut rng = test_rng();
        let (ek, dk, sks) = setup::<E>(4, 4, 3, &mut rng);

        let messages: Vec<PairingOutput<E>> = (0..4)
            .map(|_| PairingOutput::<E>::generator() * <E as Pairing>::ScalarField::rand(&mut rng))
            .collect();
        let cts: Vec<_> = messages.iter().map(|m| encrypt(&ek, m, &mut rng)).collect();

        let pds: Vec<_> = sks[..3]
            .iter()
            .map(|sk| partial_decrypt(sk, &cts, &mut rng).expect("valid ciphertext proofs"))
            .collect();
        let pd = combine::<E>(&pds).expect("valid distinct shares");

        // Corrupt the first ciphertext proof: decryption must fail without panicking.
        let mut bad_cts = cts.clone();
        bad_cts[0].proof.response += <E as Pairing>::ScalarField::one();
        assert!(decrypt(&dk, &pd, &bad_cts, &mut rng).is_none());
        assert!(decrypt_fft(&dk, &pd, &bad_cts, &mut rng).is_none());

        // Wrong batch length must fail without panicking.
        assert!(decrypt(&dk, &pd, &cts[..2], &mut rng).is_none());
    }
}
