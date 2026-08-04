//! Curl-noise dye advection — the pure math, no drawing. Both functions take closures so
//! they can be tested against simple analytic fields instead of the full noise field.

/// Divergence-free velocity derived from an arbitrary scalar potential, via central finite
/// differences and a 90-degree rotation of the gradient: v = (dPsi/dy, -dPsi/dx),
/// normalized to unit length and scaled by `amount`.
///
/// If the potential is flat at this point (zero gradient), returns `(0.0, 0.0)` instead of
/// dividing by zero.
pub fn curl_velocity(
    potential: impl Fn(f64, f64) -> f64,
    x: f64,
    y: f64,
    eps: f64,
    amount: f64,
) -> (f64, f64) {
    assert_ne!(eps, 0.0, "too small eps={eps}");

    let dp_dx = (potential(x + eps, y) - potential(x - eps, y)) / (2.0 * eps);
    let dp_dy = (potential(x, y + eps) - potential(x, y - eps)) / (2.0 * eps);

    let norm = (dp_dx.powi(2) + dp_dy.powi(2)).sqrt();

    if norm == 0.0 {
        (0.0, 0.0)
    } else {
        (dp_dy / norm * amount, -dp_dx / norm * amount)
    }
}

/// Advect a point through `vel` for up to `steps` iterations using RK2 (midpoint method).
/// Each returned point is guaranteed to lie within `[bounds_min, bounds_max]` on both axes —
/// as soon as a step would leave the bounds, that point is dropped and advection stops.
/// The first element of the result is always `start`.
pub fn advect_rk2(
    start: (f64, f64),
    mut vel: impl FnMut(f64, f64) -> (f64, f64),
    steps: u32,
    bounds_min: (f64, f64),
    bounds_max: (f64, f64),
) -> Vec<(f64, f64)> {
    let step_size = 1.0;
    let mut points = vec![start];

    for _ in 0..steps {
        let (last_x, last_y) = *points.last().unwrap();
        let (k1_x, k1_y) = vel(last_x, last_y);
        let (k2_x, k2_y) = vel(
            last_x + k1_x * step_size / 2.0,
            last_y + k1_y * step_size / 2.0,
        );
        let (next_x, next_y) = (last_x + k2_x * step_size, last_y + k2_y * step_size);

        if !(bounds_min.0..=bounds_max.0).contains(&next_x)
            || !(bounds_min.1..=bounds_max.1).contains(&next_y)
        {
            break;
        }

        points.push((next_x, next_y));
    }

    points
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn magnitude_always_equals_amount_for_a_nonflat_potential() {
        let potential = |x: f64, y: f64| x * y;
        let (vx, vy) = curl_velocity(potential, 2.0, 3.0, 0.01, 2.5);
        let mag = (vx * vx + vy * vy).sqrt();
        assert!(approx(mag, 2.5, 1e-3), "expected magnitude ~2.5, got {mag}");
    }

    #[test]
    fn velocity_is_perpendicular_to_the_potentials_gradient() {
        // potential = x*y has analytic gradient (dPsi/dx, dPsi/dy) = (y, x) at any point.
        // curl of a scalar field is always perpendicular to the field's own gradient —
        // that perpendicularity IS the divergence-free property this function exists for.
        let potential = |x: f64, y: f64| x * y;
        let (vx, vy) = curl_velocity(potential, 2.0, 3.0, 0.01, 1.0);
        let (gx, gy) = (3.0, 2.0); // (y, x) at (x=2, y=3)
        let dot = vx * gx + vy * gy;
        assert!(
            dot.abs() < 1e-2,
            "velocity should be ~perpendicular to gradient, dot={dot}"
        );
    }

    #[test]
    fn flat_potential_yields_zero_velocity_without_panicking() {
        let flat = |_x: f64, _y: f64| 5.0;
        let (vx, vy) = curl_velocity(flat, 0.0, 0.0, 0.01, 3.0);
        assert!(vx.is_finite() && vy.is_finite());
        assert_eq!((vx, vy), (0.0, 0.0));
    }

    #[test]
    fn straight_line_field_advances_by_a_fixed_step_each_iteration() {
        let vel = |_x: f64, _y: f64| (1.0, 0.0);
        let pts = advect_rk2((0.0, 0.0), vel, 5, (-1000.0, -1000.0), (1000.0, 1000.0));
        assert_eq!(pts.len(), 6); // start + 5 steps
        assert_eq!(pts[0], (0.0, 0.0));
        assert_eq!(pts[5], (5.0, 0.0));
    }

    #[test]
    fn stops_early_and_excludes_any_point_outside_bounds() {
        let vel = |_x: f64, _y: f64| (10.0, 0.0);
        let pts = advect_rk2((0.0, 0.0), vel, 20, (-5.0, -5.0), (5.0, 5.0));
        assert!(pts.len() < 21, "should stop before using all 20 steps");
        assert!(pts
            .iter()
            .all(|&(x, y)| (-5.0..=5.0).contains(&x) && (-5.0..=5.0).contains(&y)));
    }

    #[test]
    fn rotational_field_keeps_points_near_a_constant_radius() {
        // vel(x,y) = (-y, x) * k is a pure rotation about the origin — divergence-free,
        // constant angular speed. RK2 should hold the radius far closer to constant than
        // plain Euler would over this many steps; this test fails if you swap in Euler.
        let vel = |x: f64, y: f64| (-y * 0.05, x * 0.05);
        let pts = advect_rk2((1.0, 0.0), vel, 100, (-10.0, -10.0), (10.0, 10.0));
        let r0 = 1.0;
        for &(x, y) in &pts {
            let r = (x * x + y * y).sqrt();
            assert!(
                (r - r0).abs() < 0.05,
                "radius drifted to {r}, expected ~{r0}"
            );
        }
    }
}
