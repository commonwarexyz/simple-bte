//! Simple BTE vs packed Simple BTE (Remark 2), n = 128, f = 42, t = 85.
//!
//! The original scheme's combine / predecrypt / finalize were measured in
//! `pairing-free/benches/compare.rs` (unchanged code); here we time the packed
//! variant end to end and one-party share verification for both.

use ark_bls12_381::Bls12_381;
use ark_ec::pairing::{Pairing, PairingOutput};
use ark_ec::{CurveGroup, PrimeGroup, VariableBaseMSM};
use ark_std::rand::seq::SliceRandom;
use ark_std::{test_rng, UniformRand};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use simple_batched_threshold_encryption::bte::{
    decryption, encryption::encrypt, packed, Ciphertext, PartialDecryption,
};
use std::time::Duration;

type E = Bls12_381;
type Fr = <E as Pairing>::ScalarField;

const N: usize = 128;
const F: usize = 42;
const T: usize = 85;
const BATCH_SIZES: [usize; 5] = [8, 32, 128, 512, 2048];
const ELLS: [usize; 2] = [16, 22];

struct Context {
    params: packed::PackedParams<Fr>,
    dk: packed::PackedDecryptionKey<E>,
    cts: Vec<Ciphertext<E>>,
    pds: Vec<packed::PackedPartialDecryption<E>>,
    pd_slots: Vec<<E as Pairing>::G1>,
}

fn context(ell: usize, b: usize) -> Context {
    let mut rng = test_rng();
    let params = packed::PackedParams::<Fr>::new(N, F, T, ell, b);
    let (ek, dk, sks) = packed::setup::<E>(&params, &mut rng);
    let msgs: Vec<PairingOutput<E>> = (0..b)
        .map(|_| PairingOutput::<E>::generator() * Fr::rand(&mut rng))
        .collect();
    let cts: Vec<_> = msgs.iter().map(|m| encrypt(&ek, m, &mut rng)).collect();

    let mut idx: Vec<usize> = (0..N).collect();
    idx.shuffle(&mut rng);
    idx.truncate(T);
    let pds: Vec<_> = idx
        .iter()
        .map(|&i| packed::partial_decrypt(&params, &sks[i], &cts, &mut rng).unwrap())
        .collect();
    let pd_slots = packed::combine(&params, &pds);
    assert_eq!(packed::decrypt(&dk, &pd_slots, &cts, &mut rng), msgs);
    assert!(packed::verify(&dk, &pds[0], &cts));

    Context {
        params,
        dk,
        cts,
        pds,
        pd_slots,
    }
}

fn bench_packed(c: &mut Criterion) {
    let mut group = c.benchmark_group("packed");
    group.sample_size(10);

    for &ell in &ELLS {
        for &b in &BATCH_SIZES {
            let ctx = context(ell, b);
            let id = |name: &str| BenchmarkId::new(format!("l{ell}/{name}"), b);

            group.bench_function(id("combine"), |bench| {
                bench.iter(|| packed::combine(&ctx.params, &ctx.pds))
            });
            group.bench_function(id("predecrypt"), |bench| {
                bench.iter(|| packed::predecrypt(&ctx.dk, &ctx.cts))
            });
            let cross = packed::predecrypt(&ctx.dk, &ctx.cts);
            group.bench_function(id("finalize"), |bench| {
                bench.iter(|| packed::finalize(&ctx.dk, &ctx.pd_slots, &ctx.cts, &cross))
            });
            group.bench_function(id("verify"), |bench| {
                bench.iter(|| packed::verify(&ctx.dk, &ctx.pds[0], &ctx.cts))
            });
        }
    }
    group.finish();
}

/// Original `decryption::verify` for one party: B + 1 pairings. Only that
/// party's B verification keys are materialized.
fn bench_original_verify(c: &mut Criterion) {
    let mut group = c.benchmark_group("original");
    group.sample_size(10);

    for &b in &BATCH_SIZES {
        let mut rng = test_rng();
        // Only the encryption key and the batch-size-B powers are needed here,
        // so a 2-party committee avoids materializing n * B verification keys.
        let params = packed::PackedParams::<Fr>::new(2, 0, 1, 1, b);
        let (ek, packed_dk, _) = packed::setup::<E>(&params, &mut rng);
        let cts: Vec<_> = (0..b)
            .map(|_| {
                let m = PairingOutput::<E>::generator() * Fr::rand(&mut rng);
                encrypt(&ek, &m, &mut rng)
            })
            .collect();

        let shares: Vec<Fr> = (0..b).map(|_| Fr::rand(&mut rng)).collect();
        let mut dk = packed_dk.inner.clone();
        dk.batch_size = b;
        dk.verification_keys = vec![shares
            .iter()
            .map(|s| (<E as Pairing>::G2::generator() * s).into_affine().into())
            .collect()];
        let bases: Vec<_> = cts.iter().map(|ct| ct.ct1).collect();
        let pd = PartialDecryption::<E> {
            value: <E as Pairing>::G1::msm(&bases, &shares).unwrap(),
            party_index: 1,
        };
        assert!(decryption::verify(&dk, &pd, &cts));

        group.bench_with_input(BenchmarkId::new("verify", b), &b, |bench, _| {
            bench.iter(|| decryption::verify(&dk, &pd, &cts))
        });
    }
    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3));
    targets = bench_packed, bench_original_verify
);
criterion_main!(benches);
