use super::{schnorr_challenge, Ciphertext, EncryptionKey, SchnorrProof};
use ark_ec::pairing::{Pairing, PairingOutput};
use ark_ec::{CurveGroup, PrimeGroup};
use ark_std::rand::Rng;
use ark_std::UniformRand;

/// Encrypt a message m in G_T under the encryption key.
///
/// ct = ([k]_1,  m + k*[tau^{B+1}]_T)
pub fn encrypt<E: Pairing>(
    ek: &EncryptionKey<E>,
    message: &PairingOutput<E>,
    rng: &mut impl Rng,
) -> Ciphertext<E> {
    let k = E::ScalarField::rand(rng);
    let ct1 = (E::G1::generator() * k).into_affine();
    let ct2 = *message + ek.e * k;

    let r = E::ScalarField::rand(rng);
    let commitment = (E::G1::generator() * r).into_affine();
    let challenge = schnorr_challenge::<E>(&ct1, &ct2, &commitment);
    let response = r + challenge * k;

    Ciphertext {
        ct1,
        ct2,
        proof: SchnorrProof {
            commitment,
            response,
        },
    }
}
