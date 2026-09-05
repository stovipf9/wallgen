//! Curl-noise dye advection — the pure math, no drawing. Both functions take closures so
//! they can be tested against simple analytic fields instead of the full noise field.

use std::f64::consts::SQRT_2;

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

/// How an advection stopped, handed back alongside the path rather than in place of it.
///
/// Deliberately not a `Result`. None of the three is a failure, and the points are worth drawing in
/// every case — a filament that closed traced a whole orbit before it did. An `Err` carries no
/// points, so returning one would have settled the caller's policy inside this function: the orbit
/// could only ever be discarded, never drawn once and then left.
///
/// Which ending is worth acting on belongs to the caller, and nothing here prefers one: drawing a
/// closed orbit once, discarding it, or retracing it for the rest of the budget the way `advect_rk2`
/// does, are all expressible from the same return.
///
/// What earns the enum its place is the detection rather than the report. Reaching `Closed` is what
/// ends the run, so a caller that never reads it still gets an orbit traced once instead of laps
/// stacked on laps. Reading it is optional; the stopping is not.
#[derive(Debug, PartialEq)]
pub enum Ending {
    /// The path came back to where it began, so its level set is a loop that fits inside the
    /// bounds. Only `advect_rk2_projected` can report this, and only because it holds the path on
    /// one level set — see its doc for why the test says nothing without that.
    Closed,
    /// A step would have left `[bounds_min, bounds_max]`, so it was not taken. Every point in the
    /// path is inside the bounds, this one included.
    LeftBounds,
    /// `steps` iterations went by without either of the above. The level set the path is on must
    /// still close or reach the boundary eventually — in a bounded region there is no third
    /// option — but not within the steps it was given.
    StepsSpent,
}

/// `advect_rk2` again, but held on the level set it started from, and reporting how it stopped.
///
/// It takes the potential where `advect_rk2` takes a direction, and that is the whole design rather
/// than a convenience. The correction below is only meaningful while the direction being integrated
/// is tangent to the level sets of the scalar being projected onto; taking both a `dir` and a
/// `potential` would leave that as a precondition the caller can break, and breaking it fails
/// quietly — the integration walks one field while the projection hauls the point back onto another.
/// Deriving the direction here from the same potential makes it true by construction. `eps` comes
/// along for the same reason and is the caller's to derive, exactly as `curl_velocity` describes.
///
/// A curl streamline *is* an isoline of the potential: unnormalized, `(dPsi/dy, -dPsi/dx)` is a
/// Hamiltonian system with `Psi` for its Hamiltonian, and `curl_velocity`'s normalization only
/// changes how fast the point travels, not which curve it travels along. `Psi` is therefore
/// conserved along the true path and every departure from it is the integrator's, so each step ends
/// by pulling the point back onto `Psi = level`. That costs 13 potential evaluations against
/// `advect_rk2`'s 8, and reaches a fidelity that shrinking the step alone needs several times as
/// much work to match.
///
/// What it removes is the error *across* the isoline. The error *along* it is untouched, since the
/// correction is orthogonal to the flow by the same perpendicularity that makes the field
/// divergence-free — and for a drawn polyline that is the whole of it, because a point that lags
/// along its own curve moves no ink.
///
/// Where `grad Psi` is exactly zero the correction is skipped rather than treated as an error. That
/// is not a new failure mode: `curl_velocity` returns `(0, 0)` on the same condition, so the point
/// stops moving and the run becomes the standing-still case `advect_rk2` already documents.
///
/// Coming back to where it began ends the run. No check on direction is needed, because an isoline
/// cannot cross itself: a return to the start *is* the closure and not a pass near some unrelated
/// part of the field. That argument needs the projection — without it the path has drifted onto a
/// neighbouring level set by the time it comes round, and "did it come back?" has no clean answer.
pub fn advect_rk2_projected(
    start: (f64, f64),
    potential: impl Fn(f64, f64) -> f64,
    eps: f64,
    step_length: f64,
    steps: u32,
    bounds_min: (f64, f64),
    bounds_max: (f64, f64),
) -> (Vec<(f64, f64)>, Ending) {
    let (mut points, mut ending) = (vec![start], Ending::StepsSpent);

    let dir = |x: f64, y: f64| curl_velocity(&potential, x, y, eps);
    let level = potential(start.0, start.1);
    for _ in 0..steps {
        // RK2 step
        let (last_x, last_y) = *points.last().unwrap();
        let (k1_x, k1_y) = dir(last_x, last_y);
        let (k2_x, k2_y) = dir(
            last_x + k1_x * step_length / 2.0,
            last_y + k1_y * step_length / 2.0,
        );
        let (mut next_x, mut next_y) = (last_x + k2_x * step_length, last_y + k2_y * step_length);

        // projection
        let gradient = (
            (potential(next_x + eps, next_y) - potential(next_x - eps, next_y)) / (2.0 * eps),
            (potential(next_x, next_y + eps) - potential(next_x, next_y - eps)) / (2.0 * eps),
        );
        let grad_norm = gradient.0.hypot(gradient.1);
        if grad_norm > 0.0 {
            // The nearest point on `Psi = level` to the RK2 result, as a constrained minimization:
            // stationarity puts the correction along `grad Psi`, since any other direction travels
            // further for the same change in `Psi`. One Newton step from zero is enough — what is
            // being corrected is the local truncation error, so the linearization is excellent
            // there and a second iteration buys almost nothing.
            let mut pull_back = (potential(next_x, next_y) - level) / grad_norm.powi(2);

            // A step is tangent to the isoline, so the first-order term in `Psi(next) - level`
            // cancels and what is left is second order: the sagitta of the isoline over one step,
            // `kappa * h^2 / 2`, with `kappa` the curve's own curvature. The numerator is set by
            // the Hessian, which does not vanish where the gradient does — at a nondegenerate
            // critical point the gradient is zero and the Hessian is not — so the correction grows
            // as `grad_norm` shrinks rather than shrinking with it.
            //
            // Newton moves along the normal, so a correction as long as the radius of curvature
            // `1 / kappa` arrives at the centre of the osculating circle, where the normal is
            // undefined and past which the point crosses to the far side of an extremum or onto
            // the other branch of a saddle. Substituting the sagitta into `|c| < 1 / kappa` gives
            // `|c| < h / SQRT_2`, and the bound belongs against that ceiling rather than inside
            // it: looser overshoots the centre of curvature, tighter throttles the ordinary
            // corrections everywhere else.
            let trust_radius = step_length / SQRT_2;

            // Clamped, not dropped. Only the length is untrustworthy — the direction is the
            // minimum-displacement one whatever the magnitude — so dropping would discard a
            // correction that was right about where to go, and leave the step less corrected than
            // no bound at all would.
            if (pull_back * grad_norm).abs() > trust_radius {
                pull_back = pull_back.signum() * trust_radius / grad_norm;
            }

            next_x -= pull_back * gradient.0;
            next_y -= pull_back * gradient.1;
        }

        if !(bounds_min.0..=bounds_max.0).contains(&next_x)
            || !(bounds_min.1..=bounds_max.1).contains(&next_y)
        {
            ending = Ending::LeftBounds;
            break;
        }

        // Live from the third point: a polyline needs three vertices to enclose anything, so an
        // orbit this can represent at all is three steps around or more, and before that there is
        // no closure to find. Waiting instead until the path has left the radius would tie "close
        // enough to count as back" to "far enough along to be asking", and leave an orbit smaller
        // than the radius unable to arm the test at all.
        //
        // Taken from the current segment rather than its endpoint, so a crossing that falls between
        // two samples still counts.
        //
        // The radius has no derivation, and is held only loosely from either side: below by the
        // residual the projection leaves, which moves with the field and is not known here, above by
        // the third point's own distance from the start, past which every run fires at once. Between
        // them it trades closures found against how far short of its own start a closed path ends,
        // since the run stops as soon as the segment is inside the radius rather than once the point
        // has reached it.
        if points.len() > 2
            && distance_to_segment(
                start,
                *points.last().expect("the loop pushes one before it can get here"),
                (next_x, next_y),
            ) < step_length / 2.0
        {
            points.push((next_x, next_y));
            ending = Ending::Closed;
            break;
        }

        points.push((next_x, next_y));
    }

    (points, ending)
}

pub(crate) fn distance_to_segment(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let length_squared = dx * dx + dy * dy;
    if length_squared == 0.0 {
        return (p.0 - a.0).hypot(p.1 - a.1);
    }
    let t = (((p.0 - a.0) * dx + (p.1 - a.1) * dy) / length_squared).clamp(0.0, 1.0);
    (p.0 - (a.0 + t * dx)).hypot(p.1 - (a.1 + t * dy))
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

    /// `-(x^2 + y^2) / 2`, whose isolines are circles about the origin and whose curl is the unit
    /// tangential field the tests above use. A central difference of a quadratic is exact, so `eps`
    /// carries no error into any of the three tests that take it.
    fn circular_potential(x: f64, y: f64) -> f64 {
        -(x * x + y * y) / 2.0
    }

    /// The circumference is known here — `2 * PI * r` — so the closure can be checked against the
    /// arc actually walked rather than against itself. The run stops while the last segment is
    /// still inside the radius, so it ends up to one radius short; the tolerance is that plus room
    /// for RK2 over six hundred steps.
    #[test]
    fn a_circular_isoline_closes_after_one_circumference() {
        const R: f64 = 100.0;
        const STEP: f64 = 1.0;

        let (points, ending) = advect_rk2_projected(
            (R, 0.0),
            circular_potential,
            1e-3,
            STEP,
            2_000,
            (-1_000.0, -1_000.0),
            (1_000.0, 1_000.0),
        );

        assert_eq!(ending, Ending::Closed);
        let walked = (points.len() - 1) as f64 * STEP;
        let circumference = std::f64::consts::TAU * R;
        assert!(
            approx(walked, circumference, 2.0 * STEP),
            "walked {walked} for a circumference of {circumference}"
        );
    }

    /// The projection is what makes that closure mean anything, so it gets its own check: the
    /// potential at every point of the path is the potential at the start. Without the pull back
    /// the radius would creep outward and the path would never be on one isoline to return to.
    #[test]
    fn the_path_stays_on_the_isoline_it_started_from() {
        let start = (100.0, 0.0);
        let level = circular_potential(start.0, start.1);
        let (points, _) = advect_rk2_projected(
            start,
            circular_potential,
            1e-3,
            1.0,
            500,
            (-1_000.0, -1_000.0),
            (1_000.0, 1_000.0),
        );

        let worst = points
            .iter()
            .map(|&(x, y)| (circular_potential(x, y) - level).abs())
            .fold(0.0f64, f64::max);
        // `level` is -5000 here, so this is a relative departure of 2e-9
        assert!(worst < 1e-5, "potential departed by {worst}");
    }

    /// `y`, whose isolines are horizontal lines: open, so there is nothing to close. A test that
    /// reported closure on these would be reporting the path passing near its own start, which on a
    /// curve that cannot cross itself is the one thing it must not confuse the closure with.
    #[test]
    fn an_open_isoline_leaves_the_bounds_instead_of_closing() {
        let (points, ending) = advect_rk2_projected(
            (0.0, 5.0),
            |_x: f64, y: f64| y,
            1e-3,
            1.0,
            1_000,
            (-10.0, -10.0),
            (10.0, 10.0),
        );

        assert_eq!(ending, Ending::LeftBounds);
        assert!(points.iter().all(|&(_, y)| approx(y, 5.0, 1e-9)));
    }
}
