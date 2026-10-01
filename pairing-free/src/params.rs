//! Thresholds and evaluation domains.
//!
//! * Party `k` (0-based) sits at `omega^k`, where `omega` generates the
//!   subgroup `H_N` of size `N = next_pow2(n)`.
//! * Slot `s` of a sub-batch sits at `c * nu^s`, where `nu` generates `H_m`
//!   for `m = next_pow2(ell)`. Only the first `ell` points of the coset are
//!   used. The offset `c` is chosen so that `c^m = h` is a small integer: the
//!   combine step folds a degree-`< N` polynomial modulo `X^m - h`, and a small
//!   `h` makes that fold cost a handful of doublings instead of `N` full scalar
//!   multiplications.

use ark_ff::{BigInteger, FftField, Field, PrimeField};

#[derive(Clone, Debug)]
pub struct Params<F: FftField> {
    /// Number of parties.
    pub n: usize,
    /// Corruption threshold.
    pub f: usize,
    /// Reconstruction threshold (quorum size).
    pub t: usize,
    /// Packing factor: ciphertexts per sub-batch.
    pub ell: usize,
    /// Size of the party domain `H_N`.
    pub big_n: usize,
    /// Size of the slot coset `c * H_m`.
    pub m: usize,
    /// Generator of `H_N`.
    pub omega: F,
    /// Generator of `H_m`.
    pub nu: F,
    /// Coset offset `c`.
    pub offset: F,
    /// `h = c^m`, a small integer.
    pub offset_pow_m: F,
    /// `party_points[k] = omega^k` for `k < N`; parties use the first `n`.
    pub party_points: Vec<F>,
    /// `slot_points[s] = c * nu^s` for `s < m`; slots use the first `ell`.
    pub slot_points: Vec<F>,
}

impl<F: PrimeField> Params<F> {
    /// The setting of the paper: `t >= f + 2*ell - 1`, `t <= n`.
    pub fn new(n: usize, f: usize, t: usize, ell: usize) -> Self {
        assert!(ell >= 1, "ell must be positive");
        assert!(t <= n, "quorum larger than the committee");
        assert!(t >= f + 2 * ell - 1, "need t >= f + 2*ell - 1");

        let big_n = n.next_power_of_two();
        let m = ell.next_power_of_two();
        let omega = F::get_root_of_unity(big_n as u64).expect("2-adicity too small for n");
        let nu = F::get_root_of_unity(m as u64).expect("2-adicity too small for ell");
        let (offset, offset_pow_m) = small_offset::<F>(m);

        let party_points = powers(F::one(), omega, big_n);
        let slot_points = powers(offset, nu, m);

        // Slots must avoid the party domain: d^N != 1 for every slot d.
        for d in &slot_points {
            assert!(!d.pow([big_n as u64]).is_one(), "slot coset meets H_N");
        }

        Self {
            n,
            f,
            t,
            ell,
            big_n,
            m,
            omega,
            nu,
            offset,
            offset_pow_m,
            party_points,
            slot_points,
        }
    }

    /// The BFT setting `n >= 3f + 1`, `t = 2f + 1`, and the largest packing
    /// factor it allows, `ell = floor((t - f + 1) / 2)`.
    pub fn bft(n: usize) -> Self {
        let f = (n - 1) / 3;
        let t = 2 * f + 1;
        let ell = (t - f + 1) / 2;
        Self::new(n, f, t, ell)
    }

    pub fn num_subbatches(&self, batch_len: usize) -> usize {
        batch_len.div_ceil(self.ell)
    }

    /// Lagrange basis over the `ell` slot points, evaluated at `x`:
    /// returns `L_s(x)` for `s < ell`.
    pub fn slot_lagrange_at(&self, x: F) -> Vec<F> {
        let d = &self.slot_points[..self.ell];
        let mut denoms: Vec<F> = Vec::with_capacity(self.ell * 2);
        for s in 0..self.ell {
            let mut w = F::one();
            for k in 0..self.ell {
                if k != s {
                    w *= d[s] - d[k];
                }
            }
            denoms.push(w * (x - d[s]));
        }
        ark_ff::batch_inversion(&mut denoms);
        let z: F = d.iter().map(|ds| x - ds).product();
        denoms.into_iter().map(|inv| z * inv).collect()
    }
}

fn powers<F: Field>(start: F, step: F, len: usize) -> Vec<F> {
    let mut out = Vec::with_capacity(len);
    let mut cur = start;
    for _ in 0..len {
        out.push(cur);
        cur *= step;
    }
    out
}

/// Find the smallest integer `h >= 2` that is an `m`-th power, and return an
/// `m`-th root `c` of it together with `h`. `m` is a power of two not exceeding
/// the 2-adicity, so `c` is obtained by repeated square roots: if `x` is a
/// `2^a`-th power then both of its square roots are `2^{a-1}`-th powers, since
/// `-1` is a `2^{s-1}`-th power.
fn small_offset<F: PrimeField>(m: usize) -> (F, F) {
    assert!(m.is_power_of_two());
    let exp = {
        let mut e = F::MODULUS;
        e.sub_with_borrow(&F::BigInt::from(1u64));
        e >> m.trailing_zeros()
    };
    let mut h = 2u64;
    loop {
        let hf = F::from(h);
        if hf.pow(exp).is_one() {
            let mut c = hf;
            for _ in 0..m.trailing_zeros() {
                c = c.sqrt().expect("m-th power has a square root");
            }
            assert_eq!(c.pow([m as u64]), hf);
            assert!(!c.is_zero());
            return (c, hf);
        }
        h += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::Zero;
    use ark_pallas::Fr;

    #[test]
    fn bft_128() {
        let p = Params::<Fr>::bft(128);
        assert_eq!((p.f, p.t, p.ell, p.big_n, p.m), (42, 85, 22, 128, 32));
        assert_eq!(p.offset.pow([p.m as u64]), p.offset_pow_m);
    }

    #[test]
    fn slot_lagrange_interpolates() {
        let p = Params::<Fr>::bft(64);
        let coeffs: Vec<Fr> = (0..p.ell as u64).map(|i| Fr::from(i * 7 + 3)).collect();
        let eval = |x: Fr| coeffs.iter().rev().fold(Fr::zero(), |a, c| a * x + c);
        let x = p.party_points[5];
        let l = p.slot_lagrange_at(x);
        let interp: Fr = (0..p.ell).map(|s| l[s] * eval(p.slot_points[s])).sum();
        assert_eq!(interp, eval(x));
    }
}
