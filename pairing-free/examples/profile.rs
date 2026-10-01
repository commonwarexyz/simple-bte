use ark_ec::short_weierstrass::Projective;
use ark_ec::{AdditiveGroup, CurveGroup};
use ark_ff::{FftField, UniformRand};
use ark_pallas::{Fr, PallasConfig as P};
use ark_std::test_rng;
use pairing_free_bte::fft::GroupFft;
use pairing_free_bte::glv::{GlvScalar, glv_mul};
use std::hint::black_box;
use std::time::Instant;

fn time<T>(name: &str, iters: usize, mut f: impl FnMut() -> T) {
    for _ in 0..iters / 10 + 1 {
        black_box(f());
    }
    let t = Instant::now();
    for _ in 0..iters {
        black_box(f());
    }
    println!(
        "{name:40} {:>10.3} µs",
        t.elapsed().as_secs_f64() * 1e6 / iters as f64
    );
}

fn main() {
    let mut rng = test_rng();
    let pts: Vec<Projective<P>> = (0..256)
        .map(|_| Projective::<P>::rand(&mut rng).double().double())
        .collect(); // Z != 1
    let pts_aff_z: Vec<Projective<P>> = pts.iter().map(|p| p.into_affine().into()).collect();
    let ks: Vec<GlvScalar> = (0..256)
        .map(|_| GlvScalar::new::<P>(Fr::rand(&mut rng)))
        .collect();
    let (a, b) = (pts[0], pts[1]);
    time("proj add (distinct)", 100000, || {
        black_box(a) + black_box(b)
    });
    time("proj double", 100000, || black_box(a).double());
    let mut i = 0;
    time("glv_mul same k, same p (Z!=1)", 20000, || {
        glv_mul(&pts[0], &ks[0])
    });
    time("glv_mul varied k, varied p (Z!=1)", 20000, || {
        i = (i + 1) & 255;
        glv_mul(&pts[i], &ks[(i * 7) & 255])
    });
    time("glv_mul varied k, varied p (Z=1)", 20000, || {
        i = (i + 1) & 255;
        glv_mul(&pts_aff_z[i], &ks[(i * 7) & 255])
    });
    let k = Fr::rand(&mut rng);
    time("default mul (full scalar)", 20000, || black_box(pts[3]) * k);
    let root = Fr::get_root_of_unity(128).unwrap();
    let fft = GroupFft::new::<P>(128, root);
    time("group fft 128 (incl clone)", 200, || {
        let mut v = pts[..128].to_vec();
        fft.apply(&mut v);
        v
    });
}
