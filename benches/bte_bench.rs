use ark_bls12_381::Bls12_381;
use ark_ec::pairing::{Pairing, PairingOutput};
use ark_ec::PrimeGroup;
use ark_std::{test_rng, UniformRand};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use simple_batched_threshold_encryption::bte::{
    crs::setup,
    decryption::{combine, decrypt_fft, finalize_decrypt, partial_decrypt, predecrypt_fft, verify},
    encryption::encrypt,
    Ciphertext, DecryptionKey, EncryptionKey, PartialDecryption, SecretKey,
};
use std::time::Duration;

type E = Bls12_381;
type Fr = <E as Pairing>::ScalarField;

struct BenchContext {
    ek: EncryptionKey<E>,
    dk: DecryptionKey<E>,
    sks: Vec<SecretKey<E>>,
    messages: Vec<PairingOutput<E>>,
    cts: Vec<Ciphertext<E>>,
    pds: Vec<PartialDecryption<E>>,
    combined_pd: <E as Pairing>::G1,
}

fn make_context(batch_size: usize, num_parties: usize, threshold: usize) -> BenchContext {
    let mut rng = test_rng();

    let (ek, dk, sks) = setup::<E>(batch_size, num_parties, threshold, &mut rng);

    let messages: Vec<PairingOutput<E>> = (0..batch_size)
        .map(|_| PairingOutput::<E>::generator() * Fr::rand(&mut rng))
        .collect();

    let cts: Vec<_> = messages.iter().map(|m| encrypt(&ek, m, &mut rng)).collect();

    assert!(simple_batched_threshold_encryption::bte::decryption::verify_ciphertext_batch(
        &cts,
        &mut rng,
    ));

    let pds: Vec<_> = sks[..threshold]
        .iter()
        .map(|sk| partial_decrypt(sk, &cts, &mut rng).expect("valid ciphertext proofs"))
        .collect();

    let combined_pd = combine::<E>(&pds);

    BenchContext {
        ek,
        dk,
        sks,
        messages,
        cts,
        pds,
        combined_pd,
    }
}

fn bench_encrypt(c: &mut Criterion) {
    let mut group = c.benchmark_group("encrypt");
    group.sample_size(10);
    let ctx = make_context(8, 100, 50);
    let mut rng = test_rng();

    group.bench_function("single_ct", |bench| {
        bench.iter(|| encrypt(&ctx.ek, &ctx.messages[0], &mut rng));
    });
    group.finish();
}

fn bench_partial_decrypt(c: &mut Criterion) {
    let mut group = c.benchmark_group("partial_decrypt");
    group.sample_size(10);

    for &b in &[8, 32, 128, 512, 2048] {
        let ctx = make_context(b, 100, 50);
        group.bench_with_input(BenchmarkId::from_parameter(b), &b, |bench, _| {
            let mut rng = test_rng();
            bench.iter(|| partial_decrypt(&ctx.sks[0], &ctx.cts, &mut rng).expect("valid ciphertext proofs"));
        });
    }
    group.finish();
}

fn bench_verify(c: &mut Criterion) {
    let mut group = c.benchmark_group("verify");
    group.sample_size(10);

    for &b in &[8, 32, 128, 512, 2048] {
        let ctx = make_context(b, 100, 50);
        group.bench_with_input(BenchmarkId::from_parameter(b), &b, |bench, _| {
            bench.iter(|| verify(&ctx.dk, &ctx.pds[0], &ctx.cts));
        });
    }
    group.finish();
}

fn bench_combine(c: &mut Criterion) {
    let mut group = c.benchmark_group("combine");
    group.sample_size(10);

    for &b in &[8, 32, 128, 512, 2048] {
        let ctx = make_context(b, 100, 50);
        group.bench_with_input(BenchmarkId::from_parameter(b), &b, |bench, _| {
            bench.iter(|| combine::<E>(&ctx.pds));
        });
    }
    group.finish();
}

fn bench_decrypt_naive_vs_fft(c: &mut Criterion) {
    let mut group = c.benchmark_group("decrypt");
    group.sample_size(10);

    for &b in &[8, 32, 128, 512, 2048] {
        let ctx = make_context(b, 100, 50);

        // group.bench_with_input(BenchmarkId::new("naive", b), &b, |bench, _| {
        //     let mut rng = test_rng();
        //     bench.iter(|| decrypt(&ctx.dk, &ctx.combined_pd, &ctx.cts, &mut rng));
        // });

        group.bench_with_input(BenchmarkId::new("fft", b), &b, |bench, _| {
            let mut rng = test_rng();
            bench.iter(|| decrypt_fft(&ctx.dk, &ctx.combined_pd, &ctx.cts, &mut rng));
        });

        group.bench_with_input(BenchmarkId::new("predecrypt", b), &b, |bench, _| {
            bench.iter(|| predecrypt_fft(&ctx.dk, &ctx.cts));
        });

        group.bench_with_input(BenchmarkId::new("finalize", b), &b, |bench, _| {
            let cross = predecrypt_fft(&ctx.dk, &ctx.cts);
            bench.iter(|| finalize_decrypt(&ctx.dk, &ctx.combined_pd, &ctx.cts, &cross));
        });
    }
    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default().measurement_time(Duration::from_secs(5));
    targets =
        bench_encrypt,
        bench_partial_decrypt,
        bench_verify,
        bench_combine,
        bench_decrypt_naive_vs_fft,
);
criterion_main!(benches);
