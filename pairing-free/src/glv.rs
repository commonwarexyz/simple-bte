//! GLV scalar multiplication with a precomputed scalar decomposition.
//!
//! The FFT twiddles and the per-quorum combine constants are fixed scalars
//! that are applied to many group elements, so we decompose each of them once
//! into `k = k1 + lambda * k2` (|k1|, |k2| ~ sqrt(p)) and store the two halves
//! in width-`W` NAF form. A multiplication is then a Straus walk over ~128 bits
//! with one table of odd multiples of `P` and a second table obtained for free
//! by applying the endomorphism to the first.

use ark_ec::scalar_mul::glv::GLVConfig;
use ark_ec::short_weierstrass::{Affine, Projective};
use ark_ec::{AdditiveGroup, CurveGroup};
use ark_ff::{BigInteger, PrimeField, Zero};

/// NAF window width. Digits are odd and lie in (-2^{W-1}, 2^{W-1}).
const W: usize = 4;
/// Number of odd multiples {P, 3P, ..., (2^{W-1} - 1)P} kept in the table.
const TABLE_SIZE: usize = 1 << (W - 2);

/// A scalar decomposed for GLV multiplication. Signs of the two halves are
/// folded into the NAF digits.
#[derive(Clone, Debug)]
pub struct GlvScalar {
    naf1: Vec<i8>,
    naf2: Vec<i8>,
}

impl GlvScalar {
    pub fn new<P: GLVConfig>(k: P::ScalarField) -> Self {
        let ((sgn1, k1), (sgn2, k2)) = P::scalar_decomposition(k);
        Self {
            naf1: signed_naf(k1, sgn1),
            naf2: signed_naf(k2, sgn2),
        }
    }

    pub fn is_zero(&self) -> bool {
        self.naf1.is_empty() && self.naf2.is_empty()
    }
}

fn signed_naf<F: PrimeField>(k: F, positive: bool) -> Vec<i8> {
    let mut naf: Vec<i8> = k
        .into_bigint()
        .find_wnaf(W)
        .expect("valid window")
        .into_iter()
        .map(|d| if positive { d as i8 } else { -(d as i8) })
        .collect();
    while naf.last() == Some(&0) {
        naf.pop();
    }
    naf
}

/// Compute `k * p`.
pub fn glv_mul<P: GLVConfig>(p: &Projective<P>, k: &GlvScalar) -> Projective<P> {
    if p.is_zero() || k.is_zero() {
        return Projective::zero();
    }

    let mut t1 = [*p; TABLE_SIZE];
    let double = p.double();
    for i in 1..TABLE_SIZE {
        t1[i] = t1[i - 1] + double;
    }
    let t2: [Projective<P>; TABLE_SIZE] = core::array::from_fn(|i| P::endomorphism(&t1[i]));

    let len = k.naf1.len().max(k.naf2.len());
    let mut res = Projective::<P>::zero();
    for i in (0..len).rev() {
        res.double_in_place();
        add_digit(&mut res, &t1, k.naf1.get(i).copied().unwrap_or(0));
        add_digit(&mut res, &t2, k.naf2.get(i).copied().unwrap_or(0));
    }
    res
}

#[inline(always)]
fn add_digit<P: GLVConfig>(res: &mut Projective<P>, table: &[Projective<P>], d: i8) {
    if d > 0 {
        *res += table[(d as usize) >> 1];
    } else if d < 0 {
        *res -= table[((-d) as usize) >> 1];
    }
}

#[inline(always)]
fn add_digit_affine<P: GLVConfig>(res: &mut Projective<P>, table: &[Affine<P>], d: i8) {
    if d > 0 {
        *res += table[(d as usize) >> 1];
    } else if d < 0 {
        *res -= table[((-d) as usize) >> 1];
    }
}

/// Compute `scalars[i] * points[i]` for all `i`. The tables of odd multiples
/// of every point are normalized to affine with a single batched inversion,
/// so the walk uses mixed additions.
pub fn glv_mul_batch<P: GLVConfig>(
    points: &[Projective<P>],
    scalars: &[&GlvScalar],
) -> Vec<Projective<P>> {
    assert_eq!(points.len(), scalars.len());
    let mut tables = Vec::with_capacity(points.len() * TABLE_SIZE);
    for p in points {
        let double = p.double();
        let mut cur = *p;
        tables.push(cur);
        for _ in 1..TABLE_SIZE {
            cur += double;
            tables.push(cur);
        }
    }
    let tables = Projective::<P>::normalize_batch(&tables);

    points
        .iter()
        .zip(scalars)
        .zip(tables.chunks_exact(TABLE_SIZE))
        .map(|((p, k), t1)| {
            if p.is_zero() || k.is_zero() {
                return Projective::zero();
            }
            let t2: [Affine<P>; TABLE_SIZE] =
                core::array::from_fn(|i| P::endomorphism_affine(&t1[i]));
            let len = k.naf1.len().max(k.naf2.len());
            let mut res = Projective::<P>::zero();
            for i in (0..len).rev() {
                res.double_in_place();
                add_digit_affine(&mut res, t1, k.naf1.get(i).copied().unwrap_or(0));
                add_digit_affine(&mut res, &t2, k.naf2.get(i).copied().unwrap_or(0));
            }
            res
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ec::PrimeGroup;
    use ark_pallas::{Fr, PallasConfig, Projective as G};
    use ark_std::{UniformRand, test_rng};

    #[test]
    fn glv_matches_default_mul() {
        let mut rng = test_rng();
        for _ in 0..200 {
            let p = G::rand(&mut rng);
            let k = Fr::rand(&mut rng);
            assert_eq!(glv_mul(&p, &GlvScalar::new::<PallasConfig>(k)), p * k);
        }
        let points: Vec<G> = (0..50).map(|_| G::rand(&mut rng)).collect();
        let ks: Vec<Fr> = (0..50).map(|_| Fr::rand(&mut rng)).collect();
        let gs: Vec<GlvScalar> = ks
            .iter()
            .map(|&k| GlvScalar::new::<PallasConfig>(k))
            .collect();
        let mut pts = points.clone();
        pts[3] = G::zero();
        let refs: Vec<&GlvScalar> = gs.iter().collect();
        let out = glv_mul_batch(&pts, &refs);
        for i in 0..50 {
            assert_eq!(out[i], pts[i] * ks[i]);
        }
        let p = G::generator();
        for k in [Fr::zero(), Fr::from(1u64), -Fr::from(1u64), Fr::from(7u64)] {
            assert_eq!(glv_mul(&p, &GlvScalar::new::<PallasConfig>(k)), p * k);
        }
    }
}
