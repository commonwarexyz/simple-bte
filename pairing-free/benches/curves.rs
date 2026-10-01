//! Pallas vs Vesta on the operations that dominate Combine.

use ark_ec::scalar_mul::glv::GLVConfig;
use ark_ec::short_weierstrass::{Affine, Projective};
use ark_ec::{CurveGroup, VariableBaseMSM};
use ark_ff::{FftField, UniformRand};
use ark_std::test_rng;
use criterion::{Criterion, criterion_group, criterion_main};
use pairing_free_bte::fft::GroupFft;
use pairing_free_bte::glv::{GlvScalar, glv_mul};
use std::time::Duration;

fn bench_curve<P: GLVConfig>(c: &mut Criterion, name: &str) {
    let mut rng = test_rng();
    let mut group = c.benchmark_group(format!("curve/{name}"));

    let p = Projective::<P>::rand(&mut rng);
    let k = P::ScalarField::rand(&mut rng);
    let gk = GlvScalar::new::<P>(k);
    let q = Projective::<P>::rand(&mut rng);
    group.bench_function("add", |b| b.iter(|| p + q));
    group.bench_function("double", |b| b.iter(|| p + p));
    group.bench_function("mul_default", |b| b.iter(|| p * k));
    group.bench_function("mul_glv_precomputed", |b| b.iter(|| glv_mul(&p, &gk)));

    let bases: Vec<Affine<P>> = (0..85)
        .map(|_| Projective::<P>::rand(&mut rng).into_affine())
        .collect();
    let scalars: Vec<P::ScalarField> = (0..85).map(|_| P::ScalarField::rand(&mut rng)).collect();
    group.bench_function("msm_85", |b| {
        b.iter(|| Projective::<P>::msm(&bases, &scalars).unwrap())
    });

    let root = P::ScalarField::get_root_of_unity(128).unwrap();
    let fft = GroupFft::new::<P>(128, root);
    let input: Vec<Projective<P>> = (0..128).map(|_| Projective::<P>::rand(&mut rng)).collect();
    group.bench_function("group_fft_128", |b| {
        b.iter(|| {
            let mut a = input.clone();
            fft.apply(&mut a);
            a
        })
    });
    group.finish();
}

fn benches(c: &mut Criterion) {
    bench_curve::<ark_pallas::PallasConfig>(c, "pallas");
    bench_curve::<ark_vesta::VestaConfig>(c, "vesta");
}

criterion_group!(
    name = curves;
    config = Criterion::default()
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2));
    targets = benches
);
criterion_main!(curves);
