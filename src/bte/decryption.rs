use super::{
    schnorr_challenge, Ciphertext, DecryptionKey, PartialDecryption, SecretKey,
};
use ark_ec::pairing::{Pairing, PairingOutput};
use ark_ec::{AffineRepr, CurveGroup, PrimeGroup, VariableBaseMSM};
use ark_ff::{batch_inversion, One, Zero};
use ark_poly::EvaluationDomain;
use ark_std::{rand::Rng, UniformRand};

/// Verify all Schnorr proofs attached to a ciphertext batch with one
/// randomized linear-combination check.
pub fn verify_ciphertext_batch<E: Pairing>(
    cts: &[Ciphertext<E>],
    rng: &mut impl Rng,
) -> bool {
    if cts.is_empty() {
        return true;
    }

    let mut lhs_scalar = E::ScalarField::zero();
    let mut rhs_bases = Vec::with_capacity(2 * cts.len());
    let mut rhs_scalars = Vec::with_capacity(2 * cts.len());

    for ct in cts {
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
pub fn verify<E: Pairing>(
    dk: &DecryptionKey<E>,
    pd: &PartialDecryption<E>,
    cts: &[Ciphertext<E>],
) -> bool {
    assert_eq!(cts.len(), dk.batch_size);
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
pub fn combine<E: Pairing>(partial_decs: &[PartialDecryption<E>]) -> E::G1 {
    let t = partial_decs.len();
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
    E::G1::msm(&bases, &lambdas).unwrap()
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
pub fn decrypt<E: Pairing>(
    dk: &DecryptionKey<E>,
    pd: &E::G1,
    cts: &[Ciphertext<E>],
    rng: &mut impl Rng,
) -> Vec<PairingOutput<E>> {
    let b = dk.batch_size;
    assert_eq!(cts.len(), b);
    assert!(verify_ciphertext_batch(cts, rng), "invalid ciphertext batch");

    (0..b)
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
        .collect()
}

/// FFT-accelerated decryption: same result as `decrypt` but O(B log B) group
/// operations + O(B) pairings instead of O(B^2) pairings.
///
/// Uses the precomputed FFT of the circular h-vector stored in `dk.fft_h`.
pub fn decrypt_fft<E: Pairing>(
    dk: &DecryptionKey<E>,
    pd: &E::G1,
    cts: &[Ciphertext<E>],
    rng: &mut impl Rng,
) -> Vec<PairingOutput<E>> {
    let b = dk.batch_size;
    let n = dk.fft_size;
    assert_eq!(cts.len(), b);
    assert!(verify_ciphertext_batch(cts, rng), "invalid ciphertext batch");

    // 1. Build a_vec = [ct_0.ct1, ..., ct_{B-1}.ct1, 0, ..., 0] in G1.
    let mut a_vec: Vec<E::G1> = cts.iter().map(|ct| ct.ct1.into_group()).collect();
    a_vec.resize(n, E::G1::zero());

    // 2. FFT(a_vec) in G1.
    dk.fft_domain.fft_in_place(&mut a_vec);
    let a_hat = E::G1::normalize_batch(&a_vec);

    // 3. Pointwise pairing: c_hat[k] = e(a_hat[k], fft_h[k]).
    let mut c_hat: Vec<PairingOutput<E>> = a_hat
        .iter()
        .zip(dk.fft_h.iter())
        .map(|(a, b)| E::pairing(*a, b.clone()))
        .collect();

    // 4. iFFT(c_hat) in G_T -> C[i] = cross-term for row i.
    dk.fft_domain.ifft_in_place(&mut c_hat);

    // 5. z[i] = e(pd, h_{B-i}) - C[i],  m[i] = ct[i].ct2 - z[i].
    (0..b)
        .map(|i| {
            let pd_term = E::pairing(*pd, dk.powers_of_h[b - i].clone());
            let z_i = pd_term - c_hat[i];
            cts[i].ct2 - z_i
        })
        .collect()
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
            .map(|_| {
                PairingOutput::<E>::generator()
                    * <E as Pairing>::ScalarField::rand(&mut rng)
            })
            .collect();

        let cts: Vec<_> = messages.iter().map(|m| encrypt(&ek, m, &mut rng)).collect();
        assert!(verify_ciphertext_batch(&cts, &mut rng));

        // Partial decryptions from the first `threshold` parties.
        let pds: Vec<_> = sks[..threshold]
            .iter()
            .map(|sk| partial_decrypt(sk, &cts, &mut rng).expect("valid ciphertext proofs"))
            .collect();

        for pd in &pds {
            assert!(verify(&dk, pd, &cts), "verify failed for party {}", pd.party_index);
        }

        let pd = combine::<E>(&pds);

        // Naive O(B^2) decrypt.
        let recovered = decrypt(&dk, &pd, &cts, &mut rng);
        for i in 0..batch_size {
            assert_eq!(recovered[i], messages[i], "naive decrypt mismatch at {i}");
        }

        // FFT O(B log B) decrypt.
        let recovered_fft = decrypt_fft(&dk, &pd, &cts, &mut rng);
        for i in 0..batch_size {
            assert_eq!(recovered_fft[i], messages[i], "FFT decrypt mismatch at {i}");
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
            .map(|_| {
                PairingOutput::<E>::generator()
                    * <E as Pairing>::ScalarField::rand(&mut rng)
            })
            .collect();

        let cts: Vec<_> = messages.iter().map(|m| encrypt(&ek, m, &mut rng)).collect();
        assert!(verify_ciphertext_batch(&cts, &mut rng));

        let pds: Vec<_> = sks[..threshold]
            .iter()
            .map(|sk| partial_decrypt(sk, &cts, &mut rng).expect("valid ciphertext proofs"))
            .collect();

        let pd = combine::<E>(&pds);

        let naive = decrypt(&dk, &pd, &cts, &mut rng);
        let fft = decrypt_fft(&dk, &pd, &cts, &mut rng);

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
            .map(|_| {
                PairingOutput::<E>::generator()
                    * <E as Pairing>::ScalarField::rand(&mut rng)
            })
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
            .map(|_| {
                PairingOutput::<E>::generator()
                    * <E as Pairing>::ScalarField::rand(&mut rng)
            })
            .collect();

        let mut cts: Vec<_> = messages.iter().map(|m| encrypt(&ek, m, &mut rng)).collect();
        cts[0].proof.response += <E as Pairing>::ScalarField::one();

        assert!(!verify_ciphertext_batch(&cts, &mut rng));
        assert!(partial_decrypt(&sks[0], &cts, &mut rng).is_none());
    }
}
