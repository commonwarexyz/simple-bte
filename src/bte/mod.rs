use ark_ec::pairing::{Pairing, PairingOutput};
use ark_ff::PrimeField;
use ark_poly::Radix2EvaluationDomain;
use ark_serialize::CanonicalSerialize;
use merlin::Transcript;

pub mod crs;
pub mod decryption;
pub mod encryption;
pub mod fo;

/// Encryption key: published for anyone to encrypt.
#[derive(Clone, Debug)]
pub struct EncryptionKey<E: Pairing> {
    /// [tau^{B+1}]_T
    pub e: PairingOutput<E>,
}

/// Decryption key: public parameters for verification, combining, and decryption.
#[derive(Clone, Debug)]
pub struct DecryptionKey<E: Pairing> {
    pub batch_size: usize,
    pub num_parties: usize,
    pub threshold: usize,
    /// powers_of_h[j] is the prepared form of [tau^j]_2 for j = 0, ..., 2B.
    /// Slot B+1 is the identity.
    pub powers_of_h: Vec<E::G2Prepared>,
    /// verification_keys\[party\]\[slot\] is the prepared form of
    /// [sigma^{slot+1}_{party+1}]_2 where party in 0..N and slot in 0..B.
    pub verification_keys: Vec<Vec<E::G2Prepared>>,
    /// FFT size N = next_power_of_two(2 * batch_size).
    pub fft_size: usize,
    /// Precomputed FFT domain for FFT-accelerated decryption.
    pub fft_domain: Radix2EvaluationDomain<E::ScalarField>,
    /// Precomputed prepared FFT of the circular h-vector.
    pub fft_h: Vec<E::G2Prepared>,
}

/// Secret key for a single party.
#[derive(Clone, Debug)]
pub struct SecretKey<E: Pairing> {
    /// shares[i] = sigma^{i+1}_j -- the Shamir share of tau^{i+1} held by this party.
    pub shares: Vec<E::ScalarField>,
    /// 1-based party index.
    pub party_index: usize,
}

/// A single BTE ciphertext.
#[derive(Clone, Debug)]
pub struct Ciphertext<E: Pairing> {
    /// ct_1 = \[k\]_1
    pub ct1: E::G1Affine,
    /// ct_2 = m + k * [tau^{B+1}]_T
    pub ct2: PairingOutput<E>,
    /// Schnorr proof of knowledge of k such that ct1 = [k]_1.
    pub proof: SchnorrProof<E>,
}

/// Partial decryption share from a single party.
#[derive(Clone, Debug)]
pub struct PartialDecryption<E: Pairing> {
    /// pd_j = sum_{i in [B]} sigma^i_j * ct_{i,1}
    pub value: E::G1,
    /// 1-based party index.
    pub party_index: usize,
}

/// Non-compressed Fiat-Shamir Schnorr proof `(A, z)` for `ct1 = [k]_1`.
#[derive(Clone, Debug)]
pub struct SchnorrProof<E: Pairing> {
    /// First message `A = [r]_1`.
    pub commitment: E::G1Affine,
    /// Response `z = r + c*k`.
    pub response: E::ScalarField,
}

pub(crate) fn schnorr_challenge<E: Pairing>(
    ct1: &E::G1Affine,
    ct2: &PairingOutput<E>,
    commitment: &E::G1Affine,
) -> E::ScalarField {
    let mut transcript = Transcript::new(b"simple-bte-schnorr");
    append_canonical(&mut transcript, b"ct1", ct1);
    append_canonical(&mut transcript, b"ct2", ct2);
    append_canonical(&mut transcript, b"commitment", commitment);
    transcript_challenge::<E>(&mut transcript, b"challenge")
}

pub(crate) fn append_canonical<T: CanonicalSerialize>(
    transcript: &mut Transcript,
    label: &'static [u8],
    value: &T,
) {
    let mut bytes = Vec::new();
    value
        .serialize_compressed(&mut bytes)
        .expect("canonical serialization should succeed");
    transcript.append_message(label, &bytes);
}

fn transcript_challenge<E: Pairing>(transcript: &mut Transcript, label: &'static [u8]) -> E::ScalarField {
    let mut bytes = [0u8; 64];
    transcript.challenge_bytes(label, &mut bytes);
    E::ScalarField::from_le_bytes_mod_order(&bytes)
}
