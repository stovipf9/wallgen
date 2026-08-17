//! Rendering to actual pixels — not test-driven; correctness here means "does it look right",
//! checked by eye against the baked PNG, not by unit test (see `lib.rs`).

use std::{
    f32::consts::TAU,
    f64::consts::PI,
    mem::swap,
    ops::{Range, RangeInclusive},
};

use rand::{seq::SliceRandom, Rng};
use tiny_skia::{BlendMode, Color, Paint, PathBuilder, Pixmap, Point, Rect, Stroke, Transform};

use crate::{
    dreamcore::{digits, eyes, glyph_word, icon_shape, IconShape},
    flow::{advect_rk2, curl_velocity},
    noise::GradientNoise,
    palette::{Palette, Rgb},
};

/// Features the streamline potential fits across the short side. Under one across the frame by
/// design — the wash's range starts where this one ends — so the streamlines follow a field far
/// larger than the canvas and read as fragments of it rather than as a repeating texture. At
/// module scope so the tests can sweep the scales the layer can actually reach.
const VORTEX_PER_SCREEN_RANGE: RangeInclusive<f64> = 0.15..=1.0;

/// Curl-noise streamlines advected over a domain-warped fbm potential, drawn as tapered,
/// additively-blended filaments — the opposite compositional philosophy from `render_dreamcore`,
/// unified with it only through sharing the same 16-color palette. `render_dreamcore` and
/// `render_flow` are deliberately separate, style-selectable outputs (a rotating set), not layers
/// meant to be composited into one image.
///
/// - Background wash: `warped_potential(x,y) = fbm(x + warp_strength*wx, y + warp_strength*wy)`,
///   where `wx`/`wy` are fbm over two *independent* noise fields — independent by construction,
///   not by offsetting a single field, because an offset scaled by the warp strength shrinks with
///   it and collapses the displacement onto the diagonal at low strengths. Mapped between the
///   background and one of the two shades nearest it — kept subtle so the streamlines read as the
///   foreground.
/// - Streamline curl: uses the *unwarped* fbm as its potential (this one is sampled millions of
///   times across all the RK2 steps of every streamline, so it stays cheap — the warp cost is
///   paid only once per background pixel, not per streamline step).
/// - Each of `streamline_count` streamlines is seeded at a random point, advected via
///   `advect_rk2`, and drawn as a 3-stage tapered filament (thin/low-alpha tip → thick/full-alpha
///   middle → thin/medium-alpha tail), using `BlendMode::Screen` so overlapping filaments brighten
///   each other rather than occlude.
/// - One accent color for the whole image rather than one per filament: every filament here is
///   tracing the same potential, and a single hue is what says so. `render_dreamcore` picks per
///   fragment for the matching reason — its fragments have nothing to do with one another.
pub fn render_flow(
    palette: &Palette,
    width: u32,
    height: u32,
    streamline_count: usize,
    rng: &mut impl Rng,
) -> Pixmap {
    let mut pixmap = Pixmap::new(width, height).unwrap();

    let min_wh = width.min(height) as f64;

    let wash_color = *[palette.base01, palette.base02]
        .choose(rng)
        .expect("wash_color cannot be empty");
    // The wash gets its own feature scale, in the same unit as the streamlines' — how many
    // features fit across the short side — so the two are directly comparable: the streamline
    // field fits at most one feature across the frame, where the wash starts at one and runs all
    // the way up to a feature per pixel.
    //
    // They used to share one `scale`, and then the domain warp was the only thing giving the wash
    // any texture at all: its local stretch worked out to exactly `wash_vortex / vortex`, so the
    // warp was doing a scale conversion rather than a distortion. That single overloading caused
    // both of the problems that took the longest to pin down — the wash flattening into a ramp as
    // the vortex count was lowered (with under one feature across the frame there was nothing to
    // see), and `warp_strength` having no determinable value (it was answering two questions at
    // once). Separating the scales leaves the warp as a distortion only.
    //
    // Both ends belong to `octave_range`, restated in features per screen: under one across the
    // frame the coarsest octave no longer fits, and at `min_wh` of them a cell is down to one
    // pixel. Nothing between the two is privileged, so the draw is log-uniform over the span.
    //
    // The top is open rather than closed, because a cell of exactly one pixel is the sampling
    // limit itself rather than a point inside it. `gen_range` over a half-open float range rejects
    // any sample that lands on the end, so nothing has to be clamped afterwards.
    let wash_scale = rng.gen_range(1f64.ln()..min_wh.ln()).exp() / min_wh;
    let wash_octave_range = octave_range(wash_scale, min_wh);
    let wash_octaves = rng.gen_range(wash_octave_range.clone());
    let warp_octaves = rng.gen_range(*wash_octave_range.start()..=wash_octaves);

    let warp_noise = GradientNoise::new(rng.next_u64());
    let warp_x = GradientNoise::new(rng.next_u64());
    let warp_y = GradientNoise::new(rng.next_u64());

    let (dq_max, dq_typ) = max_warp_step(
        &warp_x,
        &warp_y,
        wash_scale,
        warp_octaves,
        width as usize,
        height as usize,
    );

    // Both ends of the window are derived, so there is no magnitude to choose and the draw is over
    // the whole of it, the same shape `wash_octave_range` takes.
    //
    // A unit uniform mapped into the window, rather than `gen_range(lo..=hi)`, because `lo` and
    // `hi` come from measurements over the pixel grid and so depend on the resolution. Drawing
    // from them directly would shift the whole random sequence when the output size changes, and
    // one shifted draw is enough to make a different picture (#5). This way the randomness
    // consumed is resolution-independent and only the value it maps to adapts. The `wash_octaves`
    // and `warp_octaves` draws are still the old shape; #5 covers converting them.
    let (lo, hi) = warp_strength_window(wash_scale, dq_max, dq_typ);
    let u = rng.gen_range(0.0..=1.0);
    let warp_strength = if lo == 0.0 {
        0.0
    } else {
        lo * (hi / lo).powf(u)
    };

    for (i_pixel, pixel) in pixmap.pixels_mut().iter_mut().enumerate() {
        let x = (i_pixel % width as usize) as f64 * wash_scale;
        let y = (i_pixel / width as usize) as f64 * wash_scale;
        let warped_potential =
            (1.0 + warp_noise.fbm(
                x + warp_strength * warp_x.fbm(x, y, warp_octaves),
                y + warp_strength * warp_y.fbm(x, y, warp_octaves),
                wash_octaves,
            )) / 2.0;
        *pixel = Color::from_rgba8(
            ((1.0 - warped_potential) * palette.base00.0 as f64
                + warped_potential * wash_color.0 as f64) as u8,
            ((1.0 - warped_potential) * palette.base00.1 as f64
                + warped_potential * wash_color.1 as f64) as u8,
            ((1.0 - warped_potential) * palette.base00.2 as f64
                + warped_potential * wash_color.2 as f64) as u8,
            255,
        )
        .premultiply()
        .to_color_u8();
    }

    let scale = rng
        .gen_range(VORTEX_PER_SCREEN_RANGE.start().ln()..=VORTEX_PER_SCREEN_RANGE.end().ln())
        .exp()
        / min_wh;
    let stream_octave_range = octave_range(scale, min_wh);
    let octaves = rng.gen_range(stream_octave_range.clone());

    let noise = GradientNoise::new(rng.next_u64());
    let potential = |x: f64, y: f64| noise.fbm(x * scale, y * scale, octaves);
    let eps = curl_eps(scale, octaves);
    let dir = |x, y| curl_velocity(potential, x, y, eps);
    let step_length = streamline_step_length(scale, octaves);
    let mut stream_color = to_color(
        *palette.all()[8..]
            .choose(rng)
            .expect("stream color should be determined"),
    );
    for _ in 0..streamline_count {
        // Arc length a filament may run, in pixels. Both ends are properties of the frame itself
        // rather than numbers to pick, so there is no constant here left to tune.
        //
        // - Lower, the short side: the shortest run that can cross the frame at all. Under it a
        //   filament cannot span the composition in any direction, whatever it is drawn on.
        // - Upper, the frame's perimeter. `advect_rk2` ends only at the frame edge or here, so a
        //   filament that lands on a closed level set retraces it until this runs out. A convex
        //   curve enclosed by a convex region is never longer than that region's own boundary, so
        //   the perimeter is one lap around the largest ring the frame can hold, and past it even
        //   that ring has started a second lap.
        //
        // Retracing is the intent, not the accident it looks like: under `BlendMode::Screen` every
        // lap adds, and the drift onto neighboring level sets fills the orbit in, so a small closed
        // orbit reads as a lit ring at an extremum of the field. Stopping on closure instead was
        // built and compared side by side — it leaves those orbits as thin single outlines and
        // touches nothing else — and that was not the picture wanted.
        //
        // Level sets are not convex, and a convoluted one outruns that bound and so closes late or
        // never. The upper end is the ring worth completing, not every ring.
        //
        // Both ends move with the output size, unlike the ranges #5 is about, and here that is the
        // point rather than a violation: the draw is a length measured against a frame, so one seed
        // gives a filament the same fraction of the frame at every resolution — and takes the same
        // single value out of the stream to do it.
        let stream_points = advect_rk2(
            (
                rng.gen_range(0.0..width as f64),
                rng.gen_range(0.0..height as f64),
            ),
            dir,
            step_length,
            (rng.gen_range(min_wh..=2.0 * (width + height) as f64) / step_length) as u32,
            (0.0, 0.0),
            (width as f64, height as f64),
        );
        let mut paint = Paint {
            blend_mode: BlendMode::Screen,
            anti_alias: true,
            ..Default::default()
        };

        // Width at either end of a filament, in pixels. From one, where a stroke stops being a
        // width and becomes a coverage, up to where a filament would start reading as a shape with
        // an interior rather than as a line.
        const TIP_WIDTH_RANGE: Range<f32> = 1.0..2.0;
        // How much wider the middle is than the ends. Over one so a filament has a body at all, and
        // low enough that the taper stays a taper — a filament much thicker in the middle than at
        // its ends reads as a leaf, which is a shape, not a trace of a flow.
        const WIDTH_RATIO_RANGE: Range<f32> = 1.7..2.5;
        // Alpha the head fades in from. Includes zero, so some filaments have no visible start at
        // all — the field they trace has no start either.
        const TIP_ALPHA_RANGE: Range<f32> = 0.0..0.25;
        // Alpha at the middle, and at the tail. The tail is left half lit where the head fades in
        // from near nothing, which is what gives a filament a direction; both are fixed rather than
        // drawn because it is the *contrast* between the two ends that carries that, and drawing
        // them would let a filament come out with none.
        const MID_ALPHA: f32 = 1.0;
        const TAIL_ALPHA: f32 = 0.5;

        let tip_width = rng.gen_range(TIP_WIDTH_RANGE);
        let mid_width = tip_width * rng.gen_range(WIDTH_RATIO_RANGE);
        let tip_alpha = rng.gen_range(TIP_ALPHA_RANGE);

        for (i_stream_line, stream_segment) in stream_points
            .iter()
            .take(stream_points.len() - 1)
            .zip(stream_points[1..].iter())
            .enumerate()
        {
            let mut pb = PathBuilder::new();
            pb.move_to(stream_segment.0 .0 as f32, stream_segment.0 .1 as f32);
            pb.line_to(stream_segment.1 .0 as f32, stream_segment.1 .1 as f32);

            let t = i_stream_line as f32 / (stream_points.len() - 1) as f32;
            // Head and tail differ: the tail stays half-lit where the head fades in from near
            // nothing, so a filament reads as having a direction.
            stream_color.set_alpha(taper(tip_alpha, MID_ALPHA, TAIL_ALPHA, t));
            paint.set_color(stream_color);
            pixmap.stroke_path(
                &pb.finish().unwrap(),
                &paint,
                &Stroke {
                    // Symmetric, unlike the alpha: both ends taper to the same `tip_width`.
                    width: taper(tip_width, mid_width, tip_width, t),
                    ..Default::default()
                },
                Transform::identity(),
                None,
            );
        }
    }

    pixmap
}

/// Octave counts worth summing at a per-pixel `scale`, as an inclusive range. Both ends are
/// limits rather than choices, and they close on each other as `scale` rises.
///
/// `fbm` sums octaves `1..=n`, and octave `k` has a lattice cell of `1 / (scale * 2^(k-1))`
/// pixels, so a count is bounded on both sides:
///
/// - `coarsest` — the first count with an octave small enough to fit inside the frame. Below it
///   every octave in the sum spans the whole image as a ramp rather than as texture, and it is
///   the coarsest one that carries the largest amplitude share.
/// - `finest` — the last count whose finest octave still has a cell of at least one pixel.
///   Gradient noise is zero at every lattice node with its extremum mid-cell, so a cell is half a
///   period, and one more octave puts the period under two pixels — past there the sum is a
///   picture of the sampling rather than of the field.
///
/// `finest.max(coarsest)` never actually fires, and is kept only so the range cannot come out
/// empty by inspection. What already prevents that is the cast: `log2` of a `scale` past 1 is
/// negative, a negative float saturates to 0 on the way to `u32`, and `finest` lands on 1 — which
/// is where `coarsest` already is unless a lattice cell is wider than the whole canvas. Removing
/// the cast, or reaching for a wrapping one, is what would open the hole this looks like it plugs.
fn octave_range(scale: f64, min_wh: f64) -> RangeInclusive<u32> {
    let dead = (1.0 / (scale * min_wh)).log2().max(0.0).ceil() as u32;
    let coarsest = dead + 1;
    let finest = (1.0 / scale).log2().floor() as u32 + 1;

    coarsest..=finest.max(coarsest)
}

/// How far apart the domain warp moves the sample points of two adjacent pixels, at unit warp
/// strength — the largest such gap over the whole canvas, and the mean. Both are in lattice units,
/// and both scale linearly with the strength, so one pass settles the warp's whole usable window.
///
/// The two are not interchangeable, which is why both come back:
///
/// - The **maximum** bounds the strength from above. Gradient noise is exactly zero at every
///   lattice node with its extremum mid-cell, so the coarsest octave has a period of exactly two
///   lattice units, and Nyquist puts the sampling step at one. Past that the field is not
///   represented in the output, and it is the worst pixel that decides — hence an extremum.
///
///   The coarsest octave, where `octave_range` takes its ceiling from the finest. The two are
///   bounding different things and the ends follow from that. There, the question is how many
///   octaves may be summed before one of them is finer than the grid, which the last one added
///   settles. Here, the sum is already fixed and the question is how far the sample points may be
///   moved before the picture stops being of the field — and moving them by a lattice unit is
///   already enough to lose octave 1, whatever the finer ones do. Bounding on the finest here
///   would forbid a warp that the eye reads as the whole point of the layer.
/// - The **mean** bounds it from below. What has to be large enough there is how differently two
///   neighboring features are displaced, which is a typical property of the field, not a worst
///   case.
///
/// The mean rather than the median only because it costs nothing: a running sum is exact and O(1),
/// where a median wants every difference kept (33MB at 1080p, 133MB at 4K) or an estimator with a
/// sample count to argue about. Nothing downstream distinguishes them — the lower end of the
/// window needs a value in the body of the distribution, not a particular quantile of it — so the
/// one that needs no memory wins by default.
///
/// Every pixel is visited rather than a sample of them, because visiting every pixel is possible
/// and estimation is then just a worse answer to the same question. It costs 2 fbm per pixel; the
/// row buffer is what keeps it there, since each neighbor's value is the next pixel's own. If that
/// ever needs to come down — WASM, say — sampling is the continuous fallback rather than a
/// different design: check `n` scattered pixels instead of all of them and `1/(n+1)` becomes the
/// share of the canvas allowed past Nyquist. The upper tail is tight enough that it would barely
/// move the answer.
fn max_warp_step(
    warp_x: &GradientNoise,
    warp_y: &GradientNoise,
    scale: f64,
    warp_octaves: u32,
    width: usize,
    height: usize,
) -> (f64, f64) {
    let (mut prev_q, mut curr_q) = (vec![(0.0, 0.0); width], vec![(0.0, 0.0); width]);
    let mut dq_max = 0f64;
    let mut sum = 0f64;
    let mut count = 0usize;
    for row in 0..height {
        for col in 0..width {
            let (x, y) = (col as f64 * scale, row as f64 * scale);
            curr_q[col] = (
                warp_x.fbm(x, y, warp_octaves),
                warp_y.fbm(x, y, warp_octaves),
            );
            if col > 0 {
                let dq =
                    (curr_q[col].0 - curr_q[col - 1].0).hypot(curr_q[col].1 - curr_q[col - 1].1);
                dq_max = dq_max.max(dq);
                sum += dq;
                count += 1;
            }
            if row > 0 {
                let dq = (curr_q[col].0 - prev_q[col].0).hypot(curr_q[col].1 - prev_q[col].1);
                dq_max = dq_max.max(dq);
                sum += dq;
                count += 1;
            }
        }
        swap(&mut prev_q, &mut curr_q);
    }
    if count == 0 {
        (0.0, 0.0)
    } else {
        (dq_max, sum / count as f64)
    }
}

/// Displacement the domain warp is allowed, in lattice units — so a value reads directly as "how
/// many features does this shove the field by", and needs no scaling to follow the resolution.
/// Both ends are limits rather than choices, taken from `max_warp_step`'s two measurements.
///
/// - Lower, from the mean. A warp only bends the field if neighboring features are displaced by
///   *different* amounts. Two points one feature apart differ by `k * |grad q|`, so below
///   `1 / |grad q|` the displacement is near-uniform at the feature scale and the map is a local
///   translation, which changes nothing the seed does not already vary.
/// - Upper, from the maximum. Past the point where adjacent pixels sample more than one lattice
///   unit apart, the coarsest octave is aliased and the output stops representing the field at
///   all. This holds whatever it looks like: beyond it the picture is of the aliasing, not of the
///   fbm.
///
///   Two pixels are already `wash_scale` apart before any warping, and the warp displaces them by
///   a further `k * |grad q|` in whatever direction the field points — so the budget the warp has
///   to spend is `1 - wash_scale`, not the whole lattice unit. Dividing the whole unit is what
///   this used to do, and at the fine end of the wash range it let the worst pixel pass Nyquist by
///   almost 40%, worst exactly where the window has collapsed and this end is the value taken.
///   The subtraction also makes the two limits agree: a wash of one lattice unit per pixel is the
///   wash's own Nyquist, and there the warp is allowed nothing at all.
///
/// The two ends can meet, and the range then collapses to the upper one alone. That is not an edge
/// case — it happens for a wash finer than `dq_typ / (dq_max + dq_typ)` lattice units per pixel,
/// which the fine end of the wash range reaches. There the wash is finer than the warp can bend,
/// so the warp shrinks to sub-feature jitter and the layer is close to plain fbm. Keeping the
/// upper end stays continuous with the draws just under that point, where the window is merely
/// narrow rather than empty.
///
/// A field with no gradient at all — both measurements zero — collapses to zero instead, since
/// warping a constant field is a no-op at any strength and the upper end would be infinite. So
/// does a wash at or past its own Nyquist, which leaves the warp no budget.
fn warp_strength_window(wash_scale: f64, dq_max: f64, dq_typ: f64) -> (f64, f64) {
    if dq_max == 0.0 || dq_typ == 0.0 {
        return (0.0, 0.0);
    }

    let hi = (1.0 - wash_scale).max(0.0) / dq_max;

    ((wash_scale / dq_typ).min(hi), hi)
}

/// Central-difference step for the gradient inside `curl_velocity`, in pixels — the trailing
/// `/ scale` converts it from lattice units.
///
/// Derived rather than tuned, because two error sources move in opposite directions with the step
/// h, so their sum has a minimum:
///
/// - Truncation. A central difference returns the true derivative times sinc(k*h) — an identity,
///   not an approximation — so a component at wavenumber k loses (k*h)^2/6. Worst at the finest
///   octave in use, k_max = PI * 2^(octaves-1), which comes from the fbm's lacunarity of 2 and
///   gradient noise's dominant wavelength of about two lattice cells.
/// - Round-off. Psi(x+h) and Psi(x-h) share more leading digits the smaller h gets, and the
///   subtraction discards exactly those, leaving an error of roughly f64::EPSILON / h.
///
/// Minimizing the sum gives h_opt = (3 * f64::EPSILON / k_max^2)^(1/3). It has to follow the
/// `octaves` actually drawn rather than be a constant: a constant could only be pinned to one end
/// of `octave_range`, and would go quietly wrong the moment that range moved. The cube root also
/// makes the result insensitive to how crudely the error model is estimated — being 100x off on
/// the round-off term moves h_opt by 4.6x — so no fudge factor is needed here, and none should be
/// added.
///
/// Too large is what this used to be (0.01..0.1 lattice units, drawn at random). Past k*h = PI the
/// sinc goes negative, so the finer octaves steered the flow the wrong way and only about three
/// octaves reached the velocity at all. Both symptoms are gone now: the streamlines read as a
/// plain contour map — curl velocity is tangent to the potential's isolines by construction, and a
/// three-octave fbm has smooth nested ones — and cancellation between sign-flipped octaves
/// manufactured near-zero gradients, fake fixed points where a streamline circles within a few
/// pixels for all its remaining steps and Screen blending burns it into a white dot.
///
/// Do not clamp it small instead: below roughly 1e-16 lattice units Psi(x+h) and Psi(x-h) round to
/// the same f64, the difference is exactly zero, and `curl_velocity` returns (0, 0) — the
/// streamline stops dead.
fn curl_eps(scale: f64, octaves: u32) -> f64 {
    (3.0 * f64::EPSILON / (PI * 2f64.powi(octaves as i32 - 1)).powi(2)).cbrt() / scale
}

/// Arc length one RK2 step covers, in pixels — `curl_velocity` returns a unit vector, so the step
/// length is exactly this.
///
/// A fixed pixel count tracks neither the resolution nor the field's detail. A chord error goes as
/// curvature times the step squared, and the curvature the streamlines are drawn on spans the whole
/// of `2^octaves / L` as derived here, so holding the step fixed hands that entire span to the
/// error — orders of magnitude across the parameter range, over-resolving at one end of it and
/// under-resolving at the other.
///
/// The shape comes from the sagitta: a chord across an arc of curvature k deviates by about
/// k*h^2/8, so h goes as the square root of the deviation accepted. Curvature is set by the
/// *finest* octave in use — the fbm's persistence of 0.5 against its lacunarity of 2 multiply to
/// 1, so every octave contributes equally to the gradient and the second derivative is dominated
/// by the last one — making k proportional to 2^octaves / L, where L = 1/scale is the pixels per
/// lattice unit. Hence sqrt(SCALE_PX * L / 2^octaves).
///
/// `SCALE_PX` is a pixel figure for how finely a streamline is resolved, but not the chord error
/// itself, only proportional to it. The missing factor is the level-set curvature of this
/// particular fbm, which has no closed form and is not a single number either — it is a
/// distribution, so which point of it to take is already a choice. It only ever appears multiplied
/// by the error accepted, so the two cannot be identified separately and tuning `SCALE_PX` by eye
/// absorbs whichever point that is, exactly.
///
/// Unlike `curl_eps` there is no optimum to find — both error sources shrink together as the step
/// shrinks, so this is cost against quality and some chosen number is unavoidable. A forgiving one:
/// the step goes as the square root of `SCALE_PX`, so a decade of it buys a factor of about three,
/// while `octave_range` spans roughly `log2(min_wh)` counts and the step goes as
/// `2^(-octaves/2)` — so the octave count alone moves it by around the square root of the short
/// side, which at any usable canvas is the larger term by far.
///
/// The figure is absolute, deliberately not scaled by the stroke width: a lateral deviation
/// displaces both edges of a stroke equally whatever its width, and thickness hides structure
/// *inside* a line, not displacement *of* the line.
///
/// It also assumes there is a well-defined curve to approximate, which holds only under
/// `octave_range`'s ceiling — above it the radius of curvature drops below a pixel and a smaller
/// step chases detail the saddle points amplify into a different macro path.
fn streamline_step_length(scale: f64, octaves: u32) -> f64 {
    const SCALE_PX: f64 = 4.0;

    (SCALE_PX / scale / 2f64.powi(octaves as i32)).sqrt()
}

fn to_color(rgb: Rgb) -> Color {
    Color::from_rgba8(rgb.0, rgb.1, rgb.2, 255)
}

/// Ramp through three control points — `tip` at `t = 0`, `mid` at `t = 0.5`, `tail` at `t = 1` —
/// interpolated linearly on each half.
///
/// Piecewise linear rather than the single quadratic through the same three points, because a
/// parabola is not bounded by its control points: with `tip = 0`, `mid = 1`, `tail = 0.5` it
/// reaches 1.02 at `t = 0.58`. That put the real peak off-centre — `mid` was a point the curve
/// passed through, not its maximum — and clipped the alpha flat against 1.0. Here `mid` is the
/// maximum, at `t = 0.5`, by construction, and the result never leaves `[min, max]` of the three.
///
/// The kink at `t = 0.5` is not visible: a streamline is drawn as hundreds of segments, so the
/// change per segment is far below one step of either width or alpha.
fn taper(tip: f32, mid: f32, tail: f32, t: f32) -> f32 {
    if t < 0.5 {
        tip + (mid - tip) * 2.0 * t
    } else {
        mid + (tail - mid) * (2.0 * t - 1.0)
    }
}

enum RefPointKind {
    LeftUpper,
    Center,
}

/// Scatter `fragment_count` Dreamcore fragments (`eyes` / `glyph_word` / `icon_shape` / `digits`,
/// chosen uniformly at random) at independently random positions across a `width` x `height`
/// canvas — no overlap avoidance, matching the "sparse and inconsistent" scattering the style
/// calls for. Drawn on the background shade; each fragment independently picks an accent color at
/// random, which is what says the fragments are unrelated to one another.
///
/// Each arm draws one number from `*_PER_SCREEN_RANGE` to fix what its kind's own unit is worth in
/// pixels, and multiplies. Everything else about a fragment — its parts, where they sit, how far
/// it reaches — comes from `dreamcore` already in that unit, so `min_wh` is the only pixel
/// quantity here and the arms hold no geometry of their own. `icon_shape` is the exception still
/// being worked: its three variants are drawn from paths built in this file.
pub fn render_dreamcore<R: Rng>(
    palette: &Palette,
    width: u32,
    height: u32,
    fragment_count: usize,
    rng: &mut R,
) -> Pixmap {
    let mut pixmap = Pixmap::new(width, height).unwrap();

    pixmap.fill(to_color(palette.base00));

    // Where to put one fragment, given the size of the smallest rectangle that encloses everything
    // it draws. Both kinds place that rectangle, not any one mark inside it: `LeftUpper` returns
    // its top-left corner and `Center` its middle, and a caller that hands over anything but the
    // enclosing extent gets a fragment that clips at the edge instead of hanging off it by its own
    // size, which is the whole allowance this is here to grant.
    let ref_point =
        |rng: &mut R, ref_point_kind: RefPointKind, fragment_width: f32, fragment_height: f32| {
            let gen_uniform = |rng: &mut R| rng.gen_range(0.0..1.0);
            Point::from_xy(
                gen_uniform(rng) * width as f32
                    + match ref_point_kind {
                        RefPointKind::LeftUpper => -gen_uniform(rng),
                        RefPointKind::Center => gen_uniform(rng) - 0.5,
                    } * fragment_width,
                gen_uniform(rng) * height as f32
                    + match ref_point_kind {
                        RefPointKind::LeftUpper => -gen_uniform(rng),
                        RefPointKind::Center => gen_uniform(rng) - 0.5,
                    } * fragment_height,
            )
        };
    let min_wh = width.min(height) as f64;
    for _ in 0..fragment_count {
        let mut paint = Paint::default();
        paint.set_color(to_color(
            *palette.all()[8..]
                .choose(rng)
                .expect("color should be determined"),
        ));
        match rng.gen_range(0..4) {
            0 => {
                let eyes = eyes(rng);
                // Size of one mark, as how many would fit across the short side. Small enough to
                // read as a detail rather than as the subject — the style wants fragments, and a
                // pair of eyes large enough to be looked *at* resolves the scene.
                //
                // Alone among the four it is a nominal rather than an exact size: `EyeMarks` draws
                // each mark at half to twice this, where the other three kinds size their unit
                // outright. So a pair spans considerably more than one of these, and how much more
                // is the reconcile item in #2.
                const EYE_MARK_PER_SCREEN_RANGE: RangeInclusive<f64> = 10.0..=100.0;
                let base_size = min_wh / rng.gen_range(EYE_MARK_PER_SCREEN_RANGE);
                let left_upper = ref_point(
                    rng,
                    RefPointKind::LeftUpper,
                    (eyes.width() * base_size) as f32,
                    (eyes.height() * base_size) as f32,
                );
                for (x, y, size) in eyes.marks() {
                    pixmap.fill_rect(
                        Rect::from_xywh(
                            left_upper.x + (x * base_size) as f32,
                            left_upper.y + (y * base_size) as f32,
                            (size * base_size) as f32,
                            (size * base_size) as f32,
                        )
                        .unwrap(),
                        &paint,
                        Transform::identity(),
                        None,
                    );
                }
            }
            1 => {
                let glyph_word = glyph_word(rng);

                // Size of one *dot*, as how many would fit across the short side — so a cell is
                // three to five of these across and a whole word far more, which is why this range
                // sits an order of magnitude above the others. Sizing the dot rather than the cell
                // is what keeps dots the same size whether a cell is two columns or three.
                //
                // The finest of the four fragment kinds even so, because writing read at a
                // distance is texture, and a glyph large enough to be studied invites being read.
                const GLYPH_DOT_PER_SCREEN_RANGE: RangeInclusive<f64> = 100.0..=300.0;

                let base_size = min_wh / rng.gen_range(GLYPH_DOT_PER_SCREEN_RANGE);
                let left_upper = ref_point(
                    rng,
                    RefPointKind::LeftUpper,
                    (glyph_word.width() * base_size) as f32,
                    (glyph_word.height() * base_size) as f32,
                );
                for (x, y, dot_size) in glyph_word.dots() {
                    pixmap.fill_rect(
                        Rect::from_xywh(
                            left_upper.x + (x * base_size) as f32,
                            left_upper.y + (y * base_size) as f32,
                            (dot_size * base_size) as f32,
                            (dot_size * base_size) as f32,
                        )
                        .unwrap(),
                        &paint,
                        Transform::identity(),
                        None,
                    );
                }
            }
            2 => {
                let icon_shape = icon_shape(rng);

                // Nominal size of an icon, as how many would fit across the short side. The
                // coarsest of the four kinds by an order of magnitude: a pictogram is read as one
                // sign rather than as texture, so it has to be large enough to have a shape.
                const ICON_PER_SCREEN_RANGE: RangeInclusive<f64> = 3.0..=10.0;
                let base_size = (min_wh / rng.gen_range(ICON_PER_SCREEN_RANGE)) as f32;
                let radius = base_size / 2.0;
                let center = ref_point(rng, RefPointKind::Center, base_size, base_size);

                // Stroke weight, against the icon's own diameter, so an icon keeps its weight as
                // it changes size. The span is wide on purpose — a hairline outline and a heavy
                // marker-drawn one are both wanted, and which it is is the loudest thing about a
                // pictogram after its shape.
                const STROKE_WIDTH_RATIO_RANGE: RangeInclusive<f32> = 0.01..=0.1;
                let stroke = Stroke {
                    width: (base_size * rng.gen_range(STROKE_WIDTH_RATIO_RANGE)).max(1.0),
                    ..Default::default()
                };
                match icon_shape {
                    IconShape::Cross => {
                        for (start_angle, end_angle) in IconShape::cross_edges() {
                            let mut pb = PathBuilder::new();
                            pb.move_to(
                                center.x + radius * start_angle.cos() as f32,
                                center.y + radius * start_angle.sin() as f32,
                            );
                            pb.line_to(
                                center.x + radius * end_angle.cos() as f32,
                                center.y + radius * end_angle.sin() as f32,
                            );
                            pixmap.stroke_path(
                                &pb.finish().unwrap(),
                                &paint,
                                &stroke,
                                Transform::identity(),
                                None,
                            );
                        }
                    }
                    IconShape::ParallelChords {
                        angle,
                        chord_length,
                    } => {
                        for (start_angle, end_angle) in IconShape::chords(angle, chord_length) {
                            let mut pb = PathBuilder::new();
                            pb.move_to(
                                center.x + radius * start_angle.cos() as f32,
                                center.y + radius * start_angle.sin() as f32,
                            );
                            pb.line_to(
                                center.x + radius * end_angle.cos() as f32,
                                center.y + radius * end_angle.sin() as f32,
                            );
                            pixmap.stroke_path(
                                &pb.finish().unwrap(),
                                &paint,
                                &stroke,
                                Transform::identity(),
                                None,
                            );
                        }
                    }
                    IconShape::RingFragment { start_angle, sweep } => {
                        let start_angle = start_angle as f32;
                        let sweep = sweep as f32;

                        // The arc reaches tiny-skia as cubics because its path API has no arc, and
                        // turning a curve into something a rasterizer can fill is the rasterizer's
                        // own business — handing it cubics is how to say that rather than
                        // flattening the arc here and taking the job away.
                        //
                        // One cubic covers this much of the arc, its control points `4/3 *
                        // tan(span / 4)` of the radius along the tangents. At a sixth of a turn
                        // that approximation strays from the true circle by 2.4e-5 of the radius,
                        // so an icon filling a 4K short side is out by a fortieth of a pixel.
                        // Being a relative error, that bound holds at every size, which is the
                        // whole gain over subdividing against a tolerance in pixels: there is
                        // nothing left that depends on how large the icon came out.
                        const MAX_SPAN: f32 = TAU / 6.0;
                        // `.max(1.0)` for a `sweep` of exactly zero, which would otherwise leave
                        // the path a lone `move_to` and `finish` returning `None`.
                        let spans = (sweep / MAX_SPAN).ceil().max(1.0) as u32;
                        let span = sweep / spans as f32;
                        // How far along the tangent each control point sits, as a fraction of the
                        // radius.
                        let handle = 4.0 / 3.0 * (span / 4.0).tan();

                        let mut pb = PathBuilder::new();
                        pb.move_to(
                            center.x + radius * start_angle.cos(),
                            center.y + radius * start_angle.sin(),
                        );
                        for i in 0..spans {
                            let from = start_angle + span * i as f32;
                            let to = from + span;
                            pb.cubic_to(
                                center.x + radius * (from.cos() - handle * from.sin()),
                                center.y + radius * (from.sin() + handle * from.cos()),
                                center.x + radius * (to.cos() + handle * to.sin()),
                                center.y + radius * (to.sin() - handle * to.cos()),
                                center.x + radius * to.cos(),
                                center.y + radius * to.sin(),
                            );
                        }
                        pixmap.stroke_path(
                            &pb.finish().unwrap(),
                            &paint,
                            &stroke,
                            Transform::identity(),
                            None,
                        );
                    }
                }
            }
            _ => {
                let digits = digits(rng);

                // Width of one cell, as how many would fit across the short side. Between the
                // icons and the glyphs: a readout is meant to be legible as a display, which a
                // glyph is not, without being the subject, which an icon is.
                const DIGIT_PER_SCREEN_RANGE: RangeInclusive<f64> = 10.0..=30.0;

                let base_size = min_wh / rng.gen_range(DIGIT_PER_SCREEN_RANGE);
                let left_upper = ref_point(
                    rng,
                    RefPointKind::LeftUpper,
                    (digits.width() * base_size) as f32,
                    (digits.height() * base_size) as f32,
                );
                for (x, y, w, h) in digits.segments() {
                    pixmap.fill_rect(
                        Rect::from_xywh(
                            left_upper.x + (x * base_size) as f32,
                            left_upper.y + (y * base_size) as f32,
                            (w * base_size) as f32,
                            (h * base_size) as f32,
                        )
                        .unwrap(),
                        &paint,
                        Transform::identity(),
                        None,
                    );
                }
            }
        }
    }
    pixmap
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Short sides worth covering, from a small window to 8K.
    const SHORT_SIDES: [f64; 7] = [240.0, 480.0, 720.0, 1080.0, 1440.0, 2160.0, 4320.0];

    /// Distinguishable slots, so a drawn pixel can never coincide with the background.
    const TEST_PALETTE: &str = r#"
colors:
  base00: "000000"
  base01: "111111"
  base02: "222222"
  base03: "333333"
  base04: "444444"
  base05: "555555"
  base06: "666666"
  base07: "777777"
  base08: "ff0000"
  base09: "ff8800"
  base0A: "ffff00"
  base0B: "00ff00"
  base0C: "00ffff"
  base0D: "0000ff"
  base0E: "ff00ff"
  base0F: "ff88ff"
"#;

    /// Every fragment kind still puts marks on the canvas.
    ///
    /// Nothing else here would notice if one stopped. The kinds are picked internally, so a test
    /// cannot ask for one; but a render of a single fragment is one kind's work alone, and if a
    /// kind draws nothing then roughly a quarter of such renders come back as bare background.
    /// A few are bare anyway — `ref_point` may hang a fragment almost entirely off the frame, and
    /// what stays inside can land in a gap between its marks — so the bound is a rate rather than
    /// zero, set at half of what losing one kind would produce.
    ///
    /// This is what a `digits` arm that built its rectangles and dropped them on the floor got
    /// past: it compiled, it left every other test green, and the readouts were simply gone.
    #[test]
    fn no_fragment_kind_quietly_stops_drawing() {
        const RENDERS: usize = 240;
        const A_QUARTER_OF_THEM: usize = RENDERS / 4;

        let palette = Palette::parse(TEST_PALETTE).expect("test palette should parse");
        let background = palette.base00;

        let bare = (0..RENDERS)
            .filter(|seed| {
                use rand::SeedableRng;
                let mut rng = rand::rngs::StdRng::seed_from_u64(*seed as u64);
                let pixmap = render_dreamcore(&palette, 320, 180, 1, &mut rng);
                pixmap.pixels().iter().all(|pixel| {
                    (pixel.red(), pixel.green(), pixel.blue())
                        == (background.0, background.1, background.2)
                })
            })
            .count();

        assert!(
            bare * 2 < A_QUARTER_OF_THEM,
            "{bare} of {RENDERS} single-fragment renders drew nothing, which is the rate a whole \
             kind drawing nothing would produce"
        );
    }

    /// Size of one lattice cell of `octave`, in pixels — the quantity both ends of `octave_range`
    /// are stated in.
    fn cell_px(scale: f64, octave: u32) -> f64 {
        1.0 / (scale * 2f64.powi(octave as i32 - 1))
    }

    /// Every `(min_wh, scale, octaves)` a layer can actually draw, given the per-screen feature
    /// count it draws from. Sampling the count log-uniformly matches how `render_flow` draws it.
    ///
    /// `top_is_drawable` mirrors whether that draw is over a closed or a half-open range, and it
    /// has to: the top of the wash's range is the sampling limit itself, so the difference between
    /// landing on it and stopping one f64 short of it is the whole point of the range being
    /// half-open. Hence the top sample is the last representable value below the end, not a
    /// fraction of the way along — a coarser sweep would step over the boundary being tested.
    fn reachable(
        vortex_range: impl Fn(f64) -> (f64, f64),
        top_is_drawable: bool,
    ) -> Vec<(f64, f64, u32)> {
        // How finely the vortex count is sampled. Anything past a handful is enough — the
        // quantities these cases feed are monotone in it, so only the ends can bind.
        const STEPS: u32 = 16;

        let mut cases = Vec::new();
        for min_wh in SHORT_SIDES {
            let (lo, hi) = vortex_range(min_wh);
            let (lo, hi) = (lo.ln(), hi.ln());
            let top = if top_is_drawable {
                hi
            } else {
                f64::from_bits(hi.to_bits() - 1)
            };

            for i in 0..STEPS {
                let vortex_ln = lo + (hi - lo) * i as f64 / STEPS as f64;
                for scale in [vortex_ln.exp() / min_wh, top.exp() / min_wh] {
                    for octaves in octave_range(scale, min_wh) {
                        cases.push((min_wh, scale, octaves));
                    }
                }
            }
        }
        assert!(!cases.is_empty());
        cases
    }

    fn streamline_cases() -> Vec<(f64, f64, u32)> {
        reachable(
            |_| {
                (
                    *VORTEX_PER_SCREEN_RANGE.start(),
                    *VORTEX_PER_SCREEN_RANGE.end(),
                )
            },
            true,
        )
    }

    fn wash_cases() -> Vec<(f64, f64, u32)> {
        reachable(|min_wh| (1.0, min_wh), false)
    }

    /// The floor. Octave 1 is always the coarsest one in the sum and its cell can be several
    /// frames wide, so what the floor can promise is not about that octave — it is that the
    /// *smallest* count on offer already reaches an octave the frame can hold. Below that count
    /// every octave summed is a ramp across the whole image rather than texture, and it is the
    /// coarsest of them that carries the largest amplitude share.
    #[test]
    fn every_reachable_octave_count_starts_inside_the_frame() {
        for (min_wh, scale, _) in streamline_cases().into_iter().chain(wash_cases()) {
            let coarsest = *octave_range(scale, min_wh).start();
            let cell = cell_px(scale, coarsest);
            assert!(
                cell <= min_wh,
                "coarsest octave {coarsest} has a {cell}px cell, wider than the {min_wh}px frame \
                 (scale={scale})"
            );
        }
    }

    /// The ceiling, and the reason it is not `finest.max(coarsest)` doing the work: a cell of
    /// under a pixel puts the octave's period under two and the sum stops representing the field.
    /// This is what fails if `VORTEX_PER_SCREEN_RANGE` or the wash's range is opened too far.
    #[test]
    fn every_reachable_octave_count_stops_at_the_pixel_grid() {
        for (min_wh, scale, _) in streamline_cases().into_iter().chain(wash_cases()) {
            let finest = *octave_range(scale, min_wh).end();
            let cell = cell_px(scale, finest);
            assert!(
                cell >= 1.0,
                "finest octave {finest} has a {cell}px cell, under one pixel (scale={scale}, \
                 min_wh={min_wh})"
            );
        }
    }

    /// Modeled error of a central difference of step `h` lattice units against the finest octave
    /// in use: truncation from the difference's sinc response, plus round-off from cancellation.
    fn central_difference_error(h: f64, octaves: u32) -> f64 {
        let k_max = PI * 2f64.powi(octaves as i32 - 1);
        (k_max * h).powi(2) / 6.0 + f64::EPSILON / h
    }

    /// `curl_eps` claims to sit at the minimum of that sum, which is the whole reason it is
    /// derived rather than chosen. Moving either way has to cost.
    #[test]
    fn curl_eps_sits_at_the_minimum_of_the_two_error_sources() {
        for (_, scale, octaves) in streamline_cases() {
            let h = curl_eps(scale, octaves) * scale;
            let here = central_difference_error(h, octaves);
            for factor in [0.1, 0.5, 2.0, 10.0] {
                let there = central_difference_error(h * factor, octaves);
                assert!(
                    here < there,
                    "eps is not at the minimum: {factor}x gives {there} against {here} \
                     (octaves={octaves}, scale={scale})"
                );
            }
        }
    }

    /// Both failure modes the derivation names, as hard limits rather than as a margin: past
    /// `k * h = PI` the sinc goes negative and the finer octaves steer the flow backwards, and
    /// under about 1e-16 lattice units the two samples round to the same f64 and the gradient
    /// comes back exactly zero.
    #[test]
    fn curl_eps_clears_both_the_sign_flip_and_the_cancellation_floor() {
        for (_, scale, octaves) in streamline_cases() {
            let h = curl_eps(scale, octaves) * scale;
            let k_max = PI * 2f64.powi(octaves as i32 - 1);
            assert!(
                k_max * h < PI,
                "k*h = {} is past the sinc sign flip (octaves={octaves})",
                k_max * h
            );
            assert!(
                h > 1e-16,
                "h = {h} lattice units is into cancellation (octaves={octaves})"
            );
        }
    }

    /// Perpendicular distance from `p` to the segment `a`-`b`.
    fn distance_to_chord(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
        let (dx, dy) = (b.0 - a.0, b.1 - a.1);
        let length_squared = dx * dx + dy * dy;
        if length_squared == 0.0 {
            return (p.0 - a.0).hypot(p.1 - a.1);
        }
        let t = (((p.0 - a.0) * dx + (p.1 - a.1) * dy) / length_squared).clamp(0.0, 1.0);
        (p.0 - (a.0 + t * dx)).hypot(p.1 - (a.1 + t * dy))
    }

    /// How far one step of `step` departs from the arc it stands in for, in pixels, on the real
    /// field rather than on a model of it: walk the same arc in `SUBSTEPS` smaller steps and take
    /// the farthest of those points from the single chord.
    ///
    /// The median over several starts, because the maximum is not a property of the step length.
    /// A unit direction field rotates arbitrarily fast where the gradient vanishes, so a start
    /// that happens to sit near a saddle produces a large departure at any step size — for a fixed
    /// 5px step the median over this sweep tops out under 2px while single samples reach 5.
    fn median_chord_error_px(scale: f64, octaves: u32, step: f64, seed: u64) -> f64 {
        const SUBSTEPS: u32 = 16;
        const SAMPLES: u64 = 31;
        const UNBOUNDED: (f64, f64) = (f64::MIN, f64::MAX);

        let eps = curl_eps(scale, octaves);

        let mut errors: Vec<f64> = (0..SAMPLES)
            .map(|i| {
                // a fresh field per sample, not just a fresh start on one field: the quantity
                // wanted is typical of the fbm, and one field's rough patch is not that
                let noise = GradientNoise::new(seed * SAMPLES + i);
                let potential = |x: f64, y: f64| noise.fbm(x * scale, y * scale, octaves);
                let start = (1000.0 + i as f64 * 777.0, 500.0 + i as f64 * 333.0);
                let chord = advect_rk2(
                    start,
                    |x, y| curl_velocity(potential, x, y, eps),
                    step,
                    1,
                    (UNBOUNDED.0, UNBOUNDED.0),
                    (UNBOUNDED.1, UNBOUNDED.1),
                );
                let arc = advect_rk2(
                    start,
                    |x, y| curl_velocity(potential, x, y, eps),
                    step / SUBSTEPS as f64,
                    SUBSTEPS,
                    (UNBOUNDED.0, UNBOUNDED.0),
                    (UNBOUNDED.1, UNBOUNDED.1),
                );
                arc.iter()
                    .map(|&p| distance_to_chord(p, chord[0], chord[1]))
                    .fold(0.0f64, f64::max)
            })
            .collect();

        errors.sort_by(|a, b| a.partial_cmp(b).unwrap());
        errors[errors.len() / 2]
    }

    /// What `streamline_step_length` buys, measured rather than modelled. Asserting that the
    /// *modelled* error is constant would prove nothing — the model is what the formula inverts,
    /// so the product is `SCALE_PX` by algebra whatever the field does. So both quantities here
    /// come from integrating the actual curl field.
    ///
    /// Two claims. The step keeps a streamline within a pixel of the arc it approximates, which is
    /// where a polyline stops being distinguishable from the curve. And it holds that error roughly
    /// level as the field's detail and the resolution move, which is the whole point of deriving it
    /// — a step fixed in pixels swings across orders of magnitude over the same sweep, resolving
    /// far past what the output can show at one end and visibly cutting corners at the other.
    ///
    /// The error does not come out exactly level, and is not expected to: the model's curvature
    /// `2^octaves / L` is proportional to the truth, not equal to it, and the ratio drifts with
    /// octave count. What the assertion below pins is that the drift is small against what it
    /// replaces, not that it is absent.
    #[test]
    fn the_step_keeps_the_measured_chord_error_under_a_pixel_and_roughly_level() {
        // The step this replaced, kept as the thing to beat.
        const FIXED_STEP_PX: f64 = 5.0;
        // Take every Nth case rather than all of them: integrating the field for each one costs
        // far more than reading a formula, and the claim is about the sweep's shape rather than
        // about any case in it. Coprime with the number of vortex samples per short side, so the
        // subset does not land on the same point of every one.
        const SAMPLE_EVERY: usize = 23;

        let mut derived = (f64::MAX, f64::MIN);
        let mut fixed = (f64::MAX, f64::MIN);
        for (seed, (_, scale, octaves)) in streamline_cases()
            .into_iter()
            .step_by(SAMPLE_EVERY)
            .enumerate()
        {
            let step = streamline_step_length(scale, octaves);
            let e = median_chord_error_px(scale, octaves, step, seed as u64);
            derived = (derived.0.min(e), derived.1.max(e));

            let e = median_chord_error_px(scale, octaves, FIXED_STEP_PX, seed as u64);
            fixed = (fixed.0.min(e), fixed.1.max(e));
        }

        assert!(
            derived.1 < 1.0,
            "worst chord error {:.3}px is visible; the step is not fine enough",
            derived.1
        );

        let spread = |(lo, hi): (f64, f64)| hi / lo;
        assert!(
            spread(derived) * 10.0 < spread(fixed),
            "deriving the step bought only {:.0}x against a fixed one's {:.0}x, over {:?} and {:?}",
            spread(derived),
            spread(fixed),
            derived,
            fixed
        );
    }

    /// A field with no gradient cannot be warped at any strength, and its upper end would be
    /// infinite — so the window has to collapse to zero rather than to `1 / 0`.
    #[test]
    fn a_flat_field_gets_no_warp_rather_than_an_infinite_one() {
        for (dq_max, dq_typ) in [(0.0, 0.0), (0.0, 0.5), (0.5, 0.0)] {
            let window = warp_strength_window(0.01, dq_max, dq_typ);
            assert_eq!(window, (0.0, 0.0), "dq_max={dq_max}, dq_typ={dq_typ}");
        }
    }

    /// Nyquist is on the *warped* grid, which is what the picture is sampled on: two adjacent
    /// pixels start `wash_scale` apart and the warp adds up to `strength * dq_max` on top, in the
    /// worst case along the same direction. Their sum is what has to stay inside one lattice unit,
    /// and it is asserted here rather than the formula that achieves it — dividing the whole unit
    /// instead of what is left of it passes no assertion in this test.
    #[test]
    fn the_warped_grid_never_steps_more_than_one_lattice_unit_per_pixel() {
        let (dq_max, dq_typ) = (0.4, 0.12);
        for wash_scale in [1e-4, 1e-3, 1e-2, 0.1, 0.3, 0.5, 0.9, 0.999] {
            let (lo, hi) = warp_strength_window(wash_scale, dq_max, dq_typ);
            let worst = wash_scale + hi * dq_max;
            assert!(
                worst <= 1.0 + 1e-12,
                "worst warped step {worst} passes Nyquist (wash_scale={wash_scale})"
            );
            assert!(lo <= hi, "wash_scale={wash_scale}");
        }
    }

    /// A wash at its own Nyquist — one lattice unit per pixel — has already spent the whole budget,
    /// so there is nothing left to warp with and the window is zero rather than negative.
    #[test]
    fn a_wash_at_its_own_nyquist_leaves_the_warp_no_room() {
        for wash_scale in [1.0, 1.5, 4.0] {
            assert_eq!(
                warp_strength_window(wash_scale, 0.4, 0.12),
                (0.0, 0.0),
                "wash_scale={wash_scale}"
            );
        }
    }

    /// The two ends meet once the wash is finer than the warp can bend. Setting `wash_scale/dq_typ`
    /// equal to `(1 - wash_scale)/dq_max` puts that at `dq_typ / (dq_max + dq_typ)`, and past it
    /// the window collapses to the upper end alone rather than inverting.
    #[test]
    fn the_window_collapses_upward_once_the_wash_outruns_the_warp() {
        let (dq_max, dq_typ) = (0.4, 0.12);
        let meeting = dq_typ / (dq_max + dq_typ);

        let open = warp_strength_window(meeting * 0.5, dq_max, dq_typ);
        assert!(open.0 < open.1, "expected room, got {open:?}");

        for wash_scale in [meeting, meeting * 1.5, meeting * 3.0] {
            let (lo, hi) = warp_strength_window(wash_scale, dq_max, dq_typ);
            assert_eq!(lo, hi, "wash_scale={wash_scale}");
            assert_eq!(hi, (1.0 - wash_scale) / dq_max, "wash_scale={wash_scale}");
        }
    }
}
