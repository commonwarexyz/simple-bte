//! Radix-2 FFT over curve points with precomputed GLV twiddles.

use crate::glv::{glv_mul_batch, GlvScalar};
use ark_ec::scalar_mul::glv::GLVConfig;
use ark_ec::short_weierstrass::Projective;
use ark_ff::{Field, Zero};

/// Twiddle factors `root^k` for `k < size / 2`, pre-decomposed for GLV.
/// Index 0 (the identity) is never multiplied.
#[derive(Clone, Debug)]
pub struct GroupFft {
    size: usize,
    twiddles: Vec<GlvScalar>,
}

impl GroupFft {
    /// `root` must be a primitive `size`-th root of unity.
    pub fn new<P: GLVConfig>(size: usize, root: P::ScalarField) -> Self {
        assert!(size.is_power_of_two());
        debug_assert_eq!(root.pow([size as u64]), P::ScalarField::ONE);
        let mut twiddles = Vec::with_capacity(size / 2);
        let mut w = P::ScalarField::ONE;
        for _ in 0..size / 2 {
            twiddles.push(GlvScalar::new::<P>(w));
            w *= root;
        }
        Self { size, twiddles }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    /// In-place, unnormalized DFT: `a[j] <- sum_k a[k] * root^{jk}`.
    pub fn apply<P: GLVConfig>(&self, a: &mut [Projective<P>]) {
        let n = self.size;
        assert_eq!(a.len(), n);
        bit_reverse_permute(a);

        let mut half = 1;
        let mut idx = Vec::with_capacity(n / 2);
        let mut pts = Vec::with_capacity(n / 2);
        let mut tws = Vec::with_capacity(n / 2);
        while half < n {
            let stride = n / (2 * half);
            // Twiddle multiplications of the whole stage in one batch.
            idx.clear();
            pts.clear();
            tws.clear();
            for start in (0..n).step_by(2 * half) {
                for j in 1..half {
                    let hi = start + j + half;
                    if !a[hi].is_zero() {
                        idx.push(hi);
                        pts.push(a[hi]);
                        tws.push(&self.twiddles[j * stride]);
                    }
                }
            }
            for (i, v) in idx.iter().zip(glv_mul_batch(&pts, &tws)) {
                a[*i] = v;
            }
            for start in (0..n).step_by(2 * half) {
                for j in 0..half {
                    let lo = start + j;
                    let hi = lo + half;
                    let (u, v) = (a[lo], a[hi]);
                    a[lo] = u + v;
                    a[hi] = u - v;
                }
            }
            half *= 2;
        }
    }
}

fn bit_reverse_permute<T>(a: &mut [T]) {
    let n = a.len();
    let bits = n.trailing_zeros();
    if bits == 0 {
        return;
    }
    for i in 0..n {
        let j = i.reverse_bits() >> (usize::BITS - bits);
        if i < j {
            a.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::FftField;
    use ark_pallas::{Fr, PallasConfig, Projective as G};
    use ark_std::{test_rng, UniformRand};

    #[test]
    fn matches_naive_dft() {
        let mut rng = test_rng();
        for size in [1usize, 2, 8, 32] {
            let root = Fr::get_root_of_unity(size as u64).unwrap();
            let fft = GroupFft::new::<PallasConfig>(size, root);
            let input: Vec<G> = (0..size).map(|_| G::rand(&mut rng)).collect();
            let mut out = input.clone();
            fft.apply(&mut out);
            for j in 0..size {
                let expected: G = (0..size)
                    .map(|k| input[k] * root.pow([(j * k) as u64]))
                    .sum();
                assert_eq!(out[j], expected, "size {size}, index {j}");
            }
        }
    }
}
