use super::{DecryptionKey, EncryptionKey, SecretKey};
use ark_ec::pairing::Pairing;
use ark_ec::{AffineRepr, CurveGroup, PrimeGroup, ScalarMul};
use ark_ff::{Field, One, Zero};
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use ark_std::rand::Rng;
use ark_std::UniformRand;

/// Trusted setup: produces encryption key, decryption key, and per-party secret keys.
///
/// Implements Setup(1^lambda, 1^B, 1^N, 1^t) from the paper.
pub fn setup<E: Pairing>(
    batch_size: usize,
    num_parties: usize,
    threshold: usize,
    rng: &mut impl Rng,
) -> (EncryptionKey<E>, DecryptionKey<E>, Vec<SecretKey<E>>) {
    assert!(
        threshold >= 1 && threshold <= num_parties,
        "threshold must be in [1, N]"
    );
    let b = batch_size;

    // -- powers of tau ---------------------------------------------------
    let tau = E::ScalarField::rand(rng);
    let mut tau_powers = Vec::with_capacity(2 * b + 1);
    tau_powers.push(E::ScalarField::one());
    for i in 1..=2 * b {
        tau_powers.push(tau_powers[i - 1] * tau);
    }

    // Group elements: zero out the B+1 slot so h_{B+1} = 0.
    let mut tau_for_groups = tau_powers.clone();
    tau_for_groups[b + 1] = E::ScalarField::zero();

    let g_affine = E::G1::generator().batch_mul(&tau_for_groups);
    let h_affine = E::G2::generator().batch_mul(&tau_for_groups);
    let powers_of_h: Vec<E::G2Prepared> = h_affine.iter().cloned().map(Into::into).collect();

    // [tau^{B+1}]_T = e([tau^B]_1, [tau]_2)
    let e = E::pairing(g_affine[b], h_affine[1]);

    // -- Shamir secret sharing --------------------------------------------
    // For each slot i in {1,...,B} we share tau^i among N parties with threshold t.
    let mut party_shares = vec![Vec::with_capacity(b); num_parties];
    let mut verification_keys = vec![Vec::with_capacity(b); num_parties];

    for slot in 0..b {
        let secret = tau_powers[slot + 1]; // tau^{slot+1}

        // Random degree-(t-1) polynomial f with f(0) = secret.
        let mut coeffs = vec![secret];
        for _ in 1..threshold {
            coeffs.push(E::ScalarField::rand(rng));
        }

        for j in 1..=num_parties {
            let x = E::ScalarField::from(j as u64);
            let share = eval_poly::<E::ScalarField>(&coeffs, &x);
            party_shares[j - 1].push(share);
            verification_keys[j - 1].push((E::G2::generator() * share).into_affine().into());
        }
    }

    // -- per-party secret keys ---------------------------------------------
    let secret_keys: Vec<SecretKey<E>> = (0..num_parties)
        .map(|j| SecretKey {
            shares: party_shares[j].clone(),
            party_index: j + 1,
        })
        .collect();

    // -- FFT precomputation for decrypt_fft ---------------------------------
    //
    // We precompute FFT(b_tilde) where b_tilde is the flipped circular
    // h-vector so that the standard circular convolution yields the
    // cross-correlation needed by the decryption cross-terms:
    //
    //   C[i] = sum_{j!=i} e(ct[j].ct1, h_{B+1+(j-i)})
    //
    // b_tilde[0]   = h_{B+1} = 0             (self-term vanishes)
    // b_tilde[k]   = h_{B+1-k}   k=1..B-1    (j < i lags)
    // b_tilde[N-j] = h_{B+1+j}   j=1..B-1    (j > i lags)
    // remaining     = 0                       (dead-zone)
    let fft_size = (2 * b).next_power_of_two();
    let fft_domain = Radix2EvaluationDomain::<E::ScalarField>::new(fft_size)
        .expect("fft_size must be within the field's 2-adicity");

    let mut b_vec: Vec<E::G2> = vec![E::G2::zero(); fft_size];
    for k in 0..b {
        b_vec[k] = h_affine[b + 1 - k].into_group(); // k=0 -> h[B+1] = 0
    }
    for j in 1..b {
        b_vec[fft_size - j] = h_affine[b + 1 + j].into_group();
    }
    fft_domain.fft_in_place(&mut b_vec);
    let fft_h: Vec<E::G2Prepared> = b_vec
        .iter()
        .map(|h| E::G2Affine::from(*h))
        .map(Into::into)
        .collect();

    let dk = DecryptionKey {
        batch_size: b,
        num_parties,
        threshold,
        powers_of_h,
        verification_keys,
        fft_size,
        fft_domain,
        fft_h,
    };

    (EncryptionKey { e }, dk, secret_keys)
}

/// Evaluate polynomial `coeffs[0] + coeffs[1]*x + ...` at `x`.
fn eval_poly<F: Field + Copy>(coeffs: &[F], x: &F) -> F {
    coeffs
        .iter()
        .rev()
        .fold(F::zero(), |acc, coeff| acc * x + *coeff)
}
