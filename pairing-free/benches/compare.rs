//! Combine + Dec: pairing-free BTE (Pallas) vs Simple BTE (BLS12-381).
//!
//! n = 128, f = 42, t = 2f + 1 = 85, ell = 22 for the pairing-free scheme.
//! Simple BTE uses the same quorum size t = 85.
//!
//! Partial decryptions are assumed correct (optimistic path): verification is
//! not timed for either scheme. Ciphertext proof verification is also excluded
//! from both.

use ark_bls12_381::Bls12_381;
use ark_ec::pairing::{Pairing, PairingOutput};
use ark_ec::short_weierstrass::{Affine, Projective};
use ark_ec::{AffineRepr, CurveGroup, PrimeGroup, ScalarMul, VariableBaseMSM};
use ark_ff::{One, UniformRand, Zero};
use ark_pallas::PallasConfig;
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use ark_std::rand::seq::SliceRandom;
use ark_std::test_rng;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use pairing_free_bte::params::Params;
use pairing_free_bte::scheme::{
    Ciphertext, CombineContext, PartialDecryption, PreparedBatch, QuorumKey, Scalar, combine,
    combine_fft, combine_lagrange, decrypt, encrypt, partial_decrypt, prepare_batch, setup,
};
use simple_batched_threshold_encryption::bte as sbte;
use std::time::Duration;

const N: usize = 128;
const BATCH_SIZES: [usize; 5] = [8, 32, 128, 512, 2048];

// ---------------------------------------------------------------------------
// Pairing-free
// ---------------------------------------------------------------------------

type P = PallasConfig;

struct PfContext {
    params: Params<Scalar<P>>,
    ctx: CombineContext<P>,
    cts: Vec<Ciphertext<P>>,
    batch: PreparedBatch<P>,
    quorum_t: Vec<usize>,
    pds_t: Vec<PartialDecryption<P>>,
    quorum_all: Vec<usize>,
    pds_all: Vec<PartialDecryption<P>>,
    combined: Vec<Projective<P>>,
}

fn pf_context(b: usize) -> PfContext {
    let mut rng = test_rng();
    let params = Params::<Scalar<P>>::bft(N);
    let (ek, sks) = setup::<P>(&params, &mut rng);
    let msgs: Vec<Affine<P>> = (0..b)
        .map(|_| Projective::<P>::rand(&mut rng).into_affine())
        .collect();
    let cts: Vec<_> = msgs.iter().map(|m| encrypt(&ek, m, &mut rng)).collect();
    let batch = prepare_batch(&cts, &mut rng);

    let pds_all: Vec<_> = sks
        .iter()
        .map(|sk| partial_decrypt(&params, sk, &batch))
        .collect();
    let quorum_all: Vec<usize> = (0..N).collect();
    let mut quorum_t = quorum_all.clone();
    quorum_t.shuffle(&mut rng);
    quorum_t.truncate(params.t);
    let pds_t: Vec<_> = quorum_t.iter().map(|&i| pds_all[i].clone()).collect();

    let ctx = CombineContext::new(&params);
    let qk = QuorumKey::new(&params, &quorum_t);
    let combined = combine_fft(&params, &ctx, &qk, &pds_t, b);
    assert_eq!(combined, combine_lagrange(&params, &ctx, &qk, &pds_t, b));
    assert_eq!(combined, combine(&params, &ctx, &qk, &pds_t, b));
    let qk_all = QuorumKey::new(&params, &quorum_all);
    assert_eq!(combined, combine_fft(&params, &ctx, &qk_all, &pds_all, b));
    for (m, o) in msgs.iter().zip(decrypt(&batch, &cts, &combined)) {
        assert_eq!(o.unwrap(), *m);
    }

    PfContext {
        params,
        ctx,
        cts,
        batch,
        quorum_t,
        pds_t,
        quorum_all,
        pds_all,
        combined,
    }
}

// ---------------------------------------------------------------------------
// Simple BTE
// ---------------------------------------------------------------------------

type E = Bls12_381;
type Fr = <E as Pairing>::ScalarField;

struct SbteContext {
    dk: sbte::DecryptionKey<E>,
    cts: Vec<sbte::Ciphertext<E>>,
    pds: Vec<sbte::PartialDecryption<E>>,
    combined: <E as Pairing>::G1,
}

/// `sbte::crs::setup` minus the n*B prepared G2 verification keys, which are
/// only needed for share verification and do not fit in memory at
/// B = 2048, n = 128. Shares are generated only for the t combining parties.
fn sbte_context(b: usize, t: usize) -> SbteContext {
    let mut rng = test_rng();

    let tau = Fr::rand(&mut rng);
    let mut tau_powers = Vec::with_capacity(2 * b + 1);
    tau_powers.push(Fr::one());
    for i in 1..=2 * b {
        tau_powers.push(tau_powers[i - 1] * tau);
    }
    let mut tau_for_groups = tau_powers.clone();
    tau_for_groups[b + 1] = Fr::zero();
    let h_affine = <E as Pairing>::G2::generator().batch_mul(&tau_for_groups);
    let powers_of_h = h_affine.iter().cloned().map(Into::into).collect();
    let g_b = (<E as Pairing>::G1::generator() * tau_powers[b]).into_affine();
    let ek = sbte::EncryptionKey::<E> {
        e: E::pairing(g_b, h_affine[1]),
    };

    let fft_size = (2 * b).next_power_of_two();
    let fft_domain = Radix2EvaluationDomain::<Fr>::new(fft_size).unwrap();
    let mut b_vec = vec![<E as Pairing>::G2::zero(); fft_size];
    for k in 0..b {
        b_vec[k] = h_affine[b + 1 - k].into_group();
    }
    for j in 1..b {
        b_vec[fft_size - j] = h_affine[b + 1 + j].into_group();
    }
    fft_domain.fft_in_place(&mut b_vec);
    let fft_h = <E as Pairing>::G2::normalize_batch(&b_vec)
        .into_iter()
        .map(Into::into)
        .collect();

    let dk = sbte::DecryptionKey::<E> {
        batch_size: b,
        num_parties: N,
        threshold: t,
        powers_of_h_affine: h_affine,
        powers_of_h,
        verification_keys: vec![],
        fft_size,
        fft_domain,
        fft_h,
    };

    let msgs: Vec<PairingOutput<E>> = (0..b)
        .map(|_| PairingOutput::<E>::generator() * Fr::rand(&mut rng))
        .collect();
    let cts: Vec<_> = msgs
        .iter()
        .map(|m| sbte::encryption::encrypt(&ek, m, &mut rng))
        .collect();

    // Shamir shares of tau^{slot+1} for parties 1..=t.
    let mut shares = vec![Vec::with_capacity(b); t];
    for slot in 0..b {
        let mut coeffs = vec![tau_powers[slot + 1]];
        coeffs.extend((1..t).map(|_| Fr::rand(&mut rng)));
        for (j, s) in shares.iter_mut().enumerate() {
            let x = Fr::from((j + 1) as u64);
            s.push(coeffs.iter().rev().fold(Fr::zero(), |acc, c| acc * x + c));
        }
    }
    let ct1s: Vec<_> = cts.iter().map(|ct| ct.ct1).collect();
    let pds: Vec<_> = shares
        .iter()
        .enumerate()
        .map(|(j, s)| sbte::PartialDecryption::<E> {
            value: <E as Pairing>::G1::msm(&ct1s, s).unwrap(),
            party_index: j + 1,
        })
        .collect();

    let combined = sbte::decryption::combine::<E>(&pds);
    let cross = sbte::decryption::predecrypt_fft(&dk, &cts);
    let out = sbte::decryption::finalize_decrypt(&dk, &combined, &cts, &cross);
    assert_eq!(out, msgs);

    SbteContext {
        dk,
        cts,
        pds,
        combined,
    }
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

fn bench_pairing_free(c: &mut Criterion) {
    let mut group = c.benchmark_group("pf");
    group.sample_size(10);

    for &b in &BATCH_SIZES {
        let pc = pf_context(b);

        // Combine includes the per-quorum precomputation, since the quorum is
        // only known once the partial decryptions arrive.
        group.bench_with_input(BenchmarkId::new("combine_fft_t", b), &b, |bench, _| {
            bench.iter(|| {
                let qk = QuorumKey::new(&pc.params, &pc.quorum_t);
                combine_fft(&pc.params, &pc.ctx, &qk, &pc.pds_t, b)
            })
        });
        group.bench_with_input(BenchmarkId::new("combine_hybrid_t", b), &b, |bench, _| {
            bench.iter(|| {
                let qk = QuorumKey::new(&pc.params, &pc.quorum_t);
                combine(&pc.params, &pc.ctx, &qk, &pc.pds_t, b)
            })
        });
        group.bench_with_input(BenchmarkId::new("combine_fft_all", b), &b, |bench, _| {
            bench.iter(|| {
                let qk = QuorumKey::new(&pc.params, &pc.quorum_all);
                combine_fft(&pc.params, &pc.ctx, &qk, &pc.pds_all, b)
            })
        });
        group.bench_with_input(BenchmarkId::new("combine_lagrange_t", b), &b, |bench, _| {
            bench.iter(|| {
                let qk = QuorumKey::new(&pc.params, &pc.quorum_t);
                combine_lagrange(&pc.params, &pc.ctx, &qk, &pc.pds_t, b)
            })
        });
        group.bench_with_input(BenchmarkId::new("dec", b), &b, |bench, _| {
            bench.iter(|| decrypt(&pc.batch, &pc.cts, &pc.combined))
        });
    }

    let pc = pf_context(8);
    group.bench_function("quorum_key_t", |bench| {
        bench.iter(|| QuorumKey::<P>::new(&pc.params, &pc.quorum_t))
    });
    group.finish();
}

fn bench_simple_bte(c: &mut Criterion) {
    // Building the B = 2048 context takes a while; allow skipping it when only
    // re-measuring the pairing-free side.
    if std::env::var_os("SKIP_SBTE").is_some() {
        return;
    }
    let mut group = c.benchmark_group("sbte");
    group.sample_size(10);
    let t = Params::<Scalar<P>>::bft(N).t;

    for &b in &BATCH_SIZES {
        let sc = sbte_context(b, t);

        group.bench_with_input(BenchmarkId::new("combine", b), &b, |bench, _| {
            bench.iter(|| sbte::decryption::combine::<E>(&sc.pds))
        });
        group.bench_with_input(BenchmarkId::new("predecrypt", b), &b, |bench, _| {
            bench.iter(|| sbte::decryption::predecrypt_fft(&sc.dk, &sc.cts))
        });
        let cross = sbte::decryption::predecrypt_fft(&sc.dk, &sc.cts);
        group.bench_with_input(BenchmarkId::new("finalize", b), &b, |bench, _| {
            bench.iter(|| sbte::decryption::finalize_decrypt(&sc.dk, &sc.combined, &sc.cts, &cross))
        });
    }
    group.finish();
}

criterion_group!(
    name = compare;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3));
    targets = bench_pairing_free, bench_simple_bte
);
criterion_main!(compare);
