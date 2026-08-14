//! Curl-noise dye advection — the pure math, no drawing. Both functions take closures so
//! they can be tested against simple analytic fields instead of the full noise field.

/// Unit-length, divergence-free direction derived from an arbitrary scalar potential, via central
/// finite differences and a 90-degree rotation of the gradient: v = (dPsi/dy, -dPsi/dx),
/// normalized.
///
/// The magnitude is discarded on purpose. What carries the meaning is the direction: it is
/// perpendicular to the gradient, and therefore tangent to the potential's isolines — that
/// perpendicularity is the divergence-free property this function exists for. How far to travel
/// along the direction is a separate decision, and belongs to `advect_rk2`'s `step_length`.
///
/// `eps` is the central-difference step, in the same units as `x` and `y`, and is not free to
/// pick: too large and the finer octaves of an fbm potential are attenuated — past a point,
/// sign-flipped — by the difference's sinc response; too small and cancellation between two
/// nearly equal potential values destroys the gradient. Both limits depend on how much detail the
/// potential carries, which only the caller knows, so the caller derives it and the reasoning is
/// written where it is derived.
///
/// If the potential is flat at this point (zero gradient), returns `(0.0, 0.0)` instead of
/// dividing by zero — to the caller, a point that never moves.
pub fn curl_velocity(potential: impl Fn(f64, f64) -> f64, x: f64, y: f64, eps: f64) -> (f64, f64) {
    assert_ne!(eps, 0.0, "too small eps={eps}");

    let dp_dx = (potential(x + eps, y) - potential(x - eps, y)) / (2.0 * eps);
    let dp_dy = (potential(x, y + eps) - potential(x, y - eps)) / (2.0 * eps);

    let norm = (dp_dx.powi(2) + dp_dy.powi(2)).sqrt();

    if norm == 0.0 {
        (0.0, 0.0)
    } else {
        (dp_dy / norm, -dp_dx / norm)
    }
}

/// Advect a point through the direction field `dir` for up to `steps` iterations using RK2
/// (midpoint method), covering `step_length` per iteration.
///
/// `dir` is required to return a unit vector. Nothing here enforces it, and the arithmetic works
/// either way, but `step_length` only means "the distance one step covers" while it holds: the
/// real displacement is `|dir| * step_length`, so a non-unit field quietly splits one quantity
/// across two parameters. `curl_velocity` normalizes for precisely this reason.
///
/// The first element of the result is always `start`, unexamined — pass one outside the bounds and
/// it comes back. Every point after it lies within `[bounds_min, bounds_max]` on both axes: as soon
/// as a step would leave, that point is dropped and advection stops. So the bounds constrain where
/// this function goes, not where it may be told to begin.
///
/// Those two are the only ways it ends. A streamline of a curl field follows a level set of the
/// potential, and in a bounded region a level set is either an arc that reaches the boundary or a
/// closed loop — so a path that does not leave spends the rest of its budget retracing one loop,
/// drifting a little onto neighboring level sets as it goes. Whether that is wanted is the
/// caller's question rather than this function's, and `steps` is where it gets answered.
///
/// A `dir` that returns `(0, 0)` leaves the point where it is, and advection still spends the
/// whole budget stacking every point on the same spot. Nothing guards against it: `curl_velocity`
/// returns that only for an exactly zero gradient, which the fbm potentials this is used with do
/// not produce, and a synthetic field that does is the caller's to handle.
pub fn advect_rk2(
    start: (f64, f64),
    mut dir: impl FnMut(f64, f64) -> (f64, f64),
    step_length: f64,
    steps: u32,
    bounds_min: (f64, f64),
    bounds_max: (f64, f64),
) -> Vec<(f64, f64)> {
    let mut points = vec![start];

    for _ in 0..steps {
        let (last_x, last_y) = *points.last().unwrap();
        let (k1_x, k1_y) = dir(last_x, last_y);
        let (k2_x, k2_y) = dir(
            last_x + k1_x * step_length / 2.0,
            last_y + k1_y * step_length / 2.0,
        );
        let (next_x, next_y) = (last_x + k2_x * step_length, last_y + k2_y * step_length);

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
    fn magnitude_is_always_one_for_a_nonflat_potential() {
        let potential = |x: f64, y: f64| x * y;
        let (vx, vy) = curl_velocity(potential, 2.0, 3.0, 0.01);
        let mag = (vx * vx + vy * vy).sqrt();
        assert!(approx(mag, 1.0, 1e-3), "expected magnitude ~1.0, got {mag}");
    }

    #[test]
    fn velocity_is_perpendicular_to_the_potentials_gradient() {
        // potential = x*y has analytic gradient (dPsi/dx, dPsi/dy) = (y, x) at any point.
        // curl of a scalar field is always perpendicular to the field's own gradient —
        // that perpendicularity IS the divergence-free property this function exists for.
        let potential = |x: f64, y: f64| x * y;
        let (vx, vy) = curl_velocity(potential, 2.0, 3.0, 0.01);
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
        let (vx, vy) = curl_velocity(flat, 0.0, 0.0, 0.01);
        assert!(vx.is_finite() && vy.is_finite());
        assert_eq!((vx, vy), (0.0, 0.0));
    }

    #[test]
    fn straight_line_field_advances_by_a_fixed_step_each_iteration() {
        let dir = |_x: f64, _y: f64| (1.0, 0.0);
        let pts = advect_rk2(
            (0.0, 0.0),
            dir,
            1.0,
            5,
            (-1000.0, -1000.0),
            (1000.0, 1000.0),
        );
        assert_eq!(pts.len(), 6); // start + 5 steps
        assert_eq!(pts[0], (0.0, 0.0));
        assert_eq!(pts[5], (5.0, 0.0));
    }

    #[test]
    fn stops_early_and_excludes_any_point_outside_bounds() {
        let dir = |_x: f64, _y: f64| (1.0, 0.0);
        // (0,0) -> (2,0) -> (4,0), and the step after that would land on (6,0), out of bounds.
        // The step length is picked so the run actually stops partway: make it large enough to
        // leave on the first step and this test asserts nothing at all.
        let pts = advect_rk2((0.0, 0.0), dir, 2.0, 20, (-5.0, -5.0), (5.0, 5.0));
        assert_eq!(
            pts.len(),
            3,
            "should stop partway, not consume all 20 steps"
        );
        assert!(pts
            .iter()
            .all(|&(x, y)| (-5.0..=5.0).contains(&x) && (-5.0..=5.0).contains(&y)));
    }

    #[test]
    fn rotational_field_keeps_points_near_a_constant_radius() {
        // dir(x,y) = (-y, x)/r is the unit tangential field about the origin — divergence-free,
        // and a point following it traces a circle at constant linear speed. (Angular speed is
        // step_length/r here, constant only because the radius holds; the field itself does not
        // fix it.) RK2 holds the radius far closer to constant than plain Euler would over this
        // many steps — Euler's drift is first order in the step and accumulates outward every
        // iteration, so it leaves the tolerance asserted here well before the budget runs out. This
        // test does fail if you swap in Euler, which is the point of the tolerance being tight.
        let dir = |x: f64, y: f64| {
            let r = (x * x + y * y).sqrt();
            if r < 1e-10 {
                (0.0, 0.0)
            } else {
                (-y / r, x / r)
            }
        };
        let pts = advect_rk2((1.0, 0.0), dir, 0.05, 100, (-10.0, -10.0), (10.0, 10.0));
        let r0 = 1.0;
        for &(x, y) in &pts {
            let r = (x * x + y * y).sqrt();
            assert!(
                (r - r0).abs() < 0.05,
                "radius drifted to {r}, expected ~{r0}"
            );
        }
    }

    /// Unit tangential field again, but given a budget many laps long. The path is a closed loop
    /// of circumference 2*PI at step 0.05, so one lap is about 126 steps and the remaining 9,874
    /// redraw the same circle. Retracing is the contract, not an accident — anything that stopped
    /// on closure would fail here — and the radius check makes it a retrace of the same orbit
    /// rather than a slow spiral off it.
    #[test]
    fn a_closed_orbit_retraces_for_the_whole_budget() {
        let dir = |x: f64, y: f64| {
            let r = (x * x + y * y).sqrt();
            if r < 1e-10 {
                (0.0, 0.0)
            } else {
                (-y / r, x / r)
            }
        };
        let step = 0.05;
        let steps = 10_000;
        let pts = advect_rk2((1.0, 0.0), dir, step, steps, (-10.0, -10.0), (10.0, 10.0));
        assert_eq!(pts.len(), steps as usize + 1, "the budget should be spent");

        let laps = step * (pts.len() - 1) as f64 / std::f64::consts::TAU;
        assert!(laps > 50.0, "expected many laps, got {laps}");

        // and it is still the same circle at the end, not a spiral off it
        let (x, y) = pts[pts.len() - 1];
        let r = (x * x + y * y).sqrt();
        assert!(
            (r - 1.0).abs() < 0.05,
            "radius drifted to {r} over {laps} laps"
        );
    }

    /// A field with no direction anywhere leaves the point where it is, and advection still spends
    /// the whole budget — every returned point is `start`. Nothing guards against it because the
    /// curl field never produces an exactly zero gradient; this only records what a synthetic one
    /// gets, so a caller that builds one knows not to expect a stop.
    #[test]
    fn a_point_that_cannot_move_spends_its_budget_standing_still() {
        let dir = |_x: f64, _y: f64| (0.0, 0.0);
        let pts = advect_rk2((0.0, 0.0), dir, 1.0, 1_000, (-10.0, -10.0), (10.0, 10.0));
        assert_eq!(pts.len(), 1_001);
        assert!(pts.iter().all(|&p| p == (0.0, 0.0)));
    }
}
