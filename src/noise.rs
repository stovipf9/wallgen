//! Gradient (Perlin-style) noise on a square lattice — deliberately *not* value noise
//! (scalar lattice values interpolated directly), which is what made the original
//! canvas demo's Flow field look blocky. Gradient noise interpolates dot products of
//! unit gradient vectors with the offset from each corner, which keeps derivatives smooth —
//! that smoothness is what the curl field in `flow.rs` depends on.

use std::f64::consts::TAU;

pub struct GradientNoise {
    seed: u64,
}

impl GradientNoise {
    /// Fully determined by `seed`: same seed must always produce the same field.
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    fn dot(p: (f64, f64), q: (f64, f64)) -> f64 {
        p.0 * q.0 + p.1 * q.1
    }

    /// Single-octave gradient noise, roughly in [-1, 1].
    pub fn sample(&self, x: f64, y: f64) -> f64 {
        let x0 = x.floor() as i64;
        let y0 = y.floor() as i64;
        let u = x - x0 as f64;
        let v = y - y0 as f64;
        let u_fade = Self::fade(u);

        // each corner contributes its own 1st-order approximation
        // zero at the corner itself, sloped by its gradient
        Self::lerp(
            Self::lerp(
                Self::dot(self.grad(x0, y0), (u, v)),
                Self::dot(self.grad(x0 + 1, y0), (u - 1.0, v)),
                u_fade,
            ),
            Self::lerp(
                Self::dot(self.grad(x0, y0 + 1), (u, v - 1.0)),
                Self::dot(self.grad(x0 + 1, y0 + 1), (u - 1.0, v - 1.0)),
                u_fade,
            ),
            Self::fade(v),
        )
    }

    /// Fractal Brownian motion: sum of `octaves` layers, each doubling frequency and halving
    /// amplitude relative to the last, normalized by the total amplitude used (so the result
    /// stays roughly in noise range regardless of octave count — `fbm(x, y, 1) == sample(x, y)`).
    pub fn fbm(&self, x: f64, y: f64, octaves: u32) -> f64 {
        assert!(
            octaves > 0,
            "given octaves is {}, and should be positive",
            octaves
        );

        let ratio_by_octave: f64 = 0.5;
        let mut result = 0.0;
        let mut total_amplitude = 0.0;

        for octave in 1..=octaves {
            let amplitude = ratio_by_octave.powi(octave as i32 - 1);
            let frequency = 1.0 / ratio_by_octave.powi(octave as i32 - 1);

            result += amplitude * self.sample(x * frequency, y * frequency);
            total_amplitude += amplitude;
        }

        result / total_amplitude
    }

    /// smoothing function on [0, 1]
    fn fade(t: f64) -> f64 {
        6.0 * t.powi(5) - 15.0 * t.powi(4) + 10.0 * t.powi(3)
    }

    /// linear interpolation
    fn lerp(a: f64, b: f64, t: f64) -> f64 {
        a + t * (b - a)
    }

    /// splitmix64's finalizer
    /// statistically-tested constants, not derived from first principles
    fn mix(z: u64) -> u64 {
        const C1: u64 = 0xbf58476d1ce4e5b9;
        const C2: u64 = 0x94d049bb133111eb;
        let mut ret = z;
        ret = (ret ^ (ret >> 30)).wrapping_mul(C1);
        ret = (ret ^ (ret >> 27)).wrapping_mul(C2);
        ret ^ (ret >> 31)
    }

    fn grad(&self, xi: i64, yi: i64) -> (f64, f64) {
        // xi and yi are folded in one at a time (mix, combine, mix again) rather than
        // combined into a single expression first, to dodge three separate pitfalls:
        // 1. and/or can collapse via absorption
        // 2. any commutative op -- even xor with `!`, which is just xor by a constant --
        //    can't tell xi from yi apart
        // 3. xor-ing two independently-mixed values can cancel outright if they happen to
        //    coincide (a ^ a == 0)
        let mut code = self.seed;
        code = Self::mix(code ^ xi as u64);
        code = Self::mix(code ^ yi as u64);

        let theta = code as f64 / u64::MAX as f64 * TAU;
        (theta.cos(), theta.sin())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_zero_at_every_integer_lattice_point() {
        // the offset vector to a lattice point's own corner is (0,0), so its dot-product
        // contribution is 0 — true for any seed, any gradient assignment.
        let n = GradientNoise::new(42);
        for (x, y) in [(0.0, 0.0), (3.0, 0.0), (0.0, 5.0), (-2.0, 4.0)] {
            let v = n.sample(x, y);
            assert!(v.abs() < 1e-9, "sample({x},{y}) should be ~0, got {v}");
        }
    }

    #[test]
    fn stays_within_expected_bounds() {
        let n = GradientNoise::new(7);
        let mut x = 0.0f64;
        for _ in 0..2000 {
            x += 0.37;
            let v = n.sample(x, x * 1.3);
            assert!((-1.5..=1.5).contains(&v), "sample out of bounds: {v}");
        }
    }

    #[test]
    fn same_seed_is_fully_deterministic() {
        let a = GradientNoise::new(123);
        let b = GradientNoise::new(123);
        assert_eq!(a.sample(1.23, 4.56), b.sample(1.23, 4.56));
    }

    #[test]
    fn different_seeds_usually_disagree() {
        let a = GradientNoise::new(1);
        let b = GradientNoise::new(2);
        assert_ne!(a.sample(1.23, 4.56), b.sample(1.23, 4.56));
    }

    #[test]
    #[should_panic]
    fn zero_octaves_is_invalid() {
        GradientNoise::new(1).fbm(1.0, 2.0, 0);
    }

    #[test]
    fn fbm_with_one_octave_matches_plain_sample() {
        let n = GradientNoise::new(9);
        assert_eq!(n.fbm(2.5, -1.1, 1), n.sample(2.5, -1.1));
    }

    #[test]
    fn fbm_adds_higher_frequency_detail_so_it_differs_from_one_octave() {
        let n = GradientNoise::new(9);
        assert_ne!(n.fbm(2.5, -1.1, 4), n.fbm(2.5, -1.1, 1));
    }

    #[test]
    fn fade_is_0_at_0_and_1_at_1() {
        let f0 = GradientNoise::fade(0.0);
        let f1 = GradientNoise::fade(1.0);
        assert!(f0.abs() < 1e-15);
        assert!((f1 - 1.0).abs() < 1e-15);
    }

    #[test]
    fn fade_is_increasing() {
        let values: Vec<_> = (0..=7)
            .map(|i| GradientNoise::fade(i as f64 / 7.0))
            .collect();

        for (i, value) in values.iter().take(values.len() - 1).enumerate() {
            assert!(*value < values[i + 1]);
        }
    }

    #[test]
    fn first_derivative_of_fade_is_nearly_0_at_the_low_boundary() {
        let eps = 1e-8;
        let g0 = GradientNoise::fade(eps) / eps;
        assert!(g0.abs() < 1e-14);
    }

    #[test]
    fn fade_is_point_symmetric_about_its_midpoint() {
        // fade(t) + fade(1-t) == 1 for every t. Combined with the low-boundary derivative
        // check above, this certifies the derivative also vanishes at the high boundary —
        // computing it directly the same way (fade(1.0) - fade(1.0 - eps)) / eps is a
        // subtraction of two values within 1e-16 of each other, so it's dominated by
        // floating-point cancellation noise (~1e-16/eps) rather than the true O(eps^2)
        // analytic answer.
        for t in [0.0, 0.1, 0.37, 0.5, 0.8, 0.999, 1.0] {
            let sum = GradientNoise::fade(t) + GradientNoise::fade(1.0 - t);
            assert!(
                (sum - 1.0).abs() < 1e-12,
                "fade({t}) + fade({}) = {sum}, expected 1",
                1.0 - t
            );
        }
    }

    #[test]
    fn lerp_is_affine_interpolation_between_two_values() {
        let a = 1.23;
        let b = 4.56;
        let l0 = GradientNoise::lerp(a, b, 0.0);
        let l_half = GradientNoise::lerp(a, b, 0.5);
        let l1 = GradientNoise::lerp(a, b, 1.0);
        assert!((a - l0).abs() < 1e-15);
        assert!(((a + b) / 2.0 - l_half).abs() < 1e-15);
        assert!((b - l1).abs() < 1e-15);
    }

    #[test]
    fn grad_is_deterministic() {
        let g = GradientNoise::new(9);

        assert_eq!(g.grad(123, 456), g.grad(123, 456));
    }

    #[test]
    fn grad_based_on_different_seeds_usually_disagree() {
        let a = GradientNoise::new(1);
        let b = GradientNoise::new(2);
        assert_ne!(a.grad(123, 456), b.grad(123, 456));
    }

    #[test]
    fn grad_is_a_unit_vector() {
        let grad = GradientNoise::new(3).grad(123, 456);
        let magnitude = (grad.0.powi(2) + grad.1.powi(2)).sqrt();
        assert!((magnitude - 1.0).abs() < 1e-15);
    }

    #[test]
    fn grad_is_usually_not_diagonally_symmetric() {
        let n = GradientNoise::new(4);
        assert_ne!(n.grad(123, 456), n.grad(456, 123));
    }
}
