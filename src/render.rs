//! Rendering to actual pixels — not test-driven; correctness here means "does it look right",
//! checked by eye against the baked PNG, not by unit test (see `lib.rs`).

use std::f32::consts::TAU;

use rand::Rng;
use tiny_skia::{BlendMode, Color, Paint, PathBuilder, Pixmap, Point, Rect, Stroke, Transform};

use crate::{
    dreamcore::{digits, eyes, glyph_words, icon, IconShape},
    flow::{advect_rk2, curl_velocity},
    noise::GradientNoise,
    palette::{Palette, Rgb},
};

/// Curl-noise streamlines advected over a domain-warped fbm potential, drawn as tapered,
/// additively-blended filaments — the opposite compositional philosophy from `render_dreamcore`,
/// unified with it only through sharing the same 16-color palette. `render_dreamcore` and
/// `render_flow` are deliberately separate, style-selectable outputs (a rotating set), not layers
/// meant to be composited into one image.
///
/// - Background wash: `warped_potential(x,y) = fbm(x + wx*warp_amt, y + wy*warp_amt)` (wx/wy are
///   separate-phase fbm), mapped into the desaturated `base00..base02` range — kept subtle so the
///   streamlines read as the foreground.
/// - Streamline curl: uses the *unwarped* fbm as its potential (this one is sampled millions of
///   times across all the RK2 steps of every streamline, so it stays cheap — the warp cost is
///   paid only once per background pixel, not per streamline step).
/// - Each of `streamline_count` streamlines is seeded at a random point, advected via
///   `advect_rk2`, and drawn as a 3-stage tapered filament (thin/low-alpha tip → thick/full-alpha
///   middle → thin/medium-alpha tail) in a random accent color (`base08`..`base0f`), using
///   `BlendMode::Plus` so overlapping filaments add rather than occlude.
pub fn render_flow(
    palette: &Palette,
    width: u32,
    height: u32,
    streamline_count: usize,
    rng: &mut impl Rng,
) -> Pixmap {
    let mut pixmap = Pixmap::new(width, height).unwrap();
    let wash_color = if rng.gen_bool(0.5) {
        palette.base01
    } else {
        palette.base02
    };
    let scale = rng.gen_range(3.0..15.0) / width.min(height) as f64;
    let warp_amt = rng.gen_range(0.5..10.0);
    let wx_shift = (rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0));
    let wy_shift = (rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0));
    let warp_octaves = rng.gen_range(2..10);
    let octaves = rng.gen_range(2..10);
    let noise = GradientNoise::new(rng.next_u64());
    for (i_pixel, pixel) in pixmap.pixels_mut().iter_mut().enumerate() {
        let x = (i_pixel % width as usize) as f64 * scale;
        let y = (i_pixel / width as usize) as f64 * scale;
        let warped_potential =
            (1.0 + noise.fbm(
                x + warp_amt
                    * noise.fbm(
                        x + warp_amt * wx_shift.0,
                        y + warp_amt * wx_shift.1,
                        warp_octaves,
                    ),
                y + warp_amt
                    * noise.fbm(
                        x + warp_amt * wy_shift.0,
                        y + warp_amt * wy_shift.1,
                        warp_octaves,
                    ),
                octaves,
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

    let potential_before = |x: f64, y: f64| noise.fbm(x * scale, y * scale, octaves);
    let noise_after = GradientNoise::new(rng.next_u64());
    let potential_after = |x: f64, y: f64| noise_after.fbm(x * scale, y * scale, octaves);
    let eps = rng.gen_range(0.01..0.1) / scale;
    let amount = 5.0;
    let stream_colors = &palette.all()[8..];
    let mut stream_color = to_color(stream_colors[rng.gen_range(0..stream_colors.len())]);
    for _ in 0..streamline_count {
        let start = (
            rng.gen_range(0.0..width as f64),
            rng.gen_range(0.0..height as f64),
        );
        let steps = (width.max(height) as f64 / amount) as u32 * rng.gen_range(1..=3);
        let mut rk2_step_count = 0;
        let vel = |x, y| {
            let vel_before = curl_velocity(potential_before, x, y, eps, amount);
            let vel_after = curl_velocity(potential_after, x, y, eps, amount);

            rk2_step_count += 1;
            let t = (rk2_step_count as f64 / (2.0 * steps as f64)).min(1.0);
            (
                (1.0 - t) * vel_before.0 + t * vel_after.0,
                (1.0 - t) * vel_before.1 + t * vel_after.1,
            )
        };
        let stream_points =
            advect_rk2(start, vel, steps, (0.0, 0.0), (width as f64, height as f64));
        let mut paint = Paint {
            blend_mode: BlendMode::Screen,
            anti_alias: true,
            ..Default::default()
        };
        let alpha0 = rng.gen_range(0.0..0.25);
        let alpha1 = 1.0;
        let width0 = rng.gen_range(0.0..1.0);
        let width1 = rng.gen_range(width0..5.0);
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
            stream_color.set_alpha(
                // t = 0, a = alpha0, t = 0.5, a = peak_alpha, t = 1.0, a = 0.5
                (1.0 + 2.0 * alpha0 - 4.0 * alpha1) * t.powi(2)
                    + (4.0 * alpha1 - 3.0 * alpha0 - 0.5) * t
                    + alpha0,
            );
            paint.set_color(stream_color);
            pixmap.stroke_path(
                &pb.finish().unwrap(),
                &paint,
                &Stroke {
                    // t = 0, w = w0, t = 0.5, w = peak_width, t = 1.0, w = w0
                    width: 4.0 * (width0 - width1) * (t - 0.5).powi(2) + width1,
                    ..Default::default()
                },
                Transform::identity(),
                None,
            );
        }
    }

    pixmap
}

fn to_color(rgb: Rgb) -> Color {
    Color::from_rgba8(rgb.0, rgb.1, rgb.2, 255)
}

/// Scatter `fragment_count` Dreamcore fragments (`eyes` / `glyph_cell` / `icon` / `digits`, chosen
/// uniformly at random) at independently random positions across a `width` x `height` canvas —
/// no overlap avoidance, matching the "sparse and inconsistent" scattering the style calls for.
/// Background is `palette.base00`; each fragment independently picks one of the 8 accent colors
/// (`base08`..`base0f`) at random.
pub fn render_dreamcore(
    palette: &Palette,
    width: u32,
    height: u32,
    fragment_count: usize,
    rng: &mut impl Rng,
) -> Pixmap {
    let mut pixmap = Pixmap::new(width, height).unwrap();

    pixmap.fill(to_color(palette.base00));

    for _ in 0..fragment_count {
        let palette_all = palette.all();
        let mut paint = Paint::default();
        paint.set_color(to_color(palette_all[rng.gen_range(8..palette_all.len())]));
        let min_wh = width.min(height) as f64;
        match rng.gen_range(0..4) {
            0 => {
                let eyes = eyes(rng.gen_range(0.5..1.0) * min_wh * 0.01, rng);
                let left_upper = Point::from_xy(
                    rng.gen_range(0.0..width as f32)
                        - rng.gen_range(0.0..1.0) * (eyes.size_a + eyes.gap + eyes.size_b) as f32,
                    rng.gen_range(0.0..height as f32)
                        - rng.gen_range(0.0..1.0) * eyes.size_a.max(eyes.size_b) as f32,
                );
                pixmap.fill_rect(
                    Rect::from_xywh(
                        left_upper.x,
                        left_upper.y,
                        eyes.size_a as f32,
                        eyes.size_a as f32,
                    )
                    .unwrap(),
                    &paint,
                    Transform::identity(),
                    None,
                );
                pixmap.fill_rect(
                    Rect::from_xywh(
                        left_upper.x + eyes.size_a as f32 + eyes.gap as f32,
                        left_upper.y + eyes.dy as f32,
                        eyes.size_b as f32,
                        eyes.size_b as f32,
                    )
                    .unwrap(),
                    &paint,
                    Transform::identity(),
                    None,
                );
            }
            1 => {
                let base_size = rng.gen_range(0.3..1.0) * min_wh * 0.01;
                let glyph_cell = glyph_words(rng.gen_range(1..10), rng);
                let cell_size = base_size as f32;
                let cell_gap = base_size as f32;
                let grid_gap = cell_gap * rng.gen_range(1.2..2.0) as f32;
                let left_upper = Point::from_xy(
                    rng.gen_range(0.0..width as f32)
                        - rng.gen_range(0.0..1.0)
                            * (glyph_cell.filled.len() as f32
                                * (glyph_cell.cols as f32 * (cell_size + cell_gap) - cell_gap
                                    + grid_gap)
                                - grid_gap),
                    rng.gen_range(0.0..height as f32)
                        - rng.gen_range(0.0..1.0)
                            * (glyph_cell.rows as f32 * (cell_size + cell_gap) - cell_gap),
                );
                for (i_grid, grid) in glyph_cell.filled.iter().enumerate() {
                    for (j_cell, cell_filled) in grid.iter().enumerate() {
                        if !*cell_filled {
                            continue;
                        };
                        pixmap.fill_rect(
                            Rect::from_xywh(
                                left_upper.x
                                    + (glyph_cell.cols as f32 * (cell_size + cell_gap) - cell_gap
                                        + grid_gap)
                                        * i_grid as f32
                                    + (j_cell % glyph_cell.cols) as f32 * (cell_size + cell_gap),
                                left_upper.y
                                    + (j_cell / glyph_cell.cols) as f32 * (cell_size + cell_gap),
                                cell_size,
                                cell_size,
                            )
                            .unwrap(),
                            &paint,
                            Transform::identity(),
                            None,
                        );
                    }
                }
            }
            2 => {
                let base_size = rng.gen_range(0.3..1.0) * min_wh * 0.1;
                let center = Point::from_xy(
                    rng.gen_range(0.0..1.0) * width as f32
                        + rng.gen_range(-0.5..0.5) * base_size as f32,
                    rng.gen_range(0.0..1.0) * height as f32
                        + rng.gen_range(-0.5..0.5) * base_size as f32,
                );
                let stroke = Stroke {
                    width: base_size as f32 * rng.gen_range(0.01..0.1),
                    ..Default::default()
                };
                let icon = icon(base_size, rng);
                match icon.shape {
                    IconShape::Cross => {
                        let mut pb1 = PathBuilder::new();
                        pb1.move_to(center.x + base_size as f32 / 2.0, center.y);
                        pb1.line_to(center.x - base_size as f32 / 2.0, center.y);
                        pixmap.stroke_path(
                            &pb1.finish().unwrap(),
                            &paint,
                            &stroke,
                            Transform::identity(),
                            None,
                        );
                        let mut pb2 = PathBuilder::new();
                        pb2.move_to(center.x, center.y + base_size as f32 / 2.0);
                        pb2.line_to(center.x, center.y - base_size as f32 / 2.0);
                        pixmap.stroke_path(
                            &pb2.finish().unwrap(),
                            &paint,
                            &stroke,
                            Transform::identity(),
                            None,
                        );
                    }
                    IconShape::DiagonalPair { angle } => {
                        let open_angle = rng.gen_range(0.0625 * TAU..0.125 * TAU);
                        let mut pb1 = PathBuilder::new();
                        pb1.move_to(
                            center.x + base_size as f32 / 2.0 * (angle as f32 + open_angle).cos(),
                            center.y + base_size as f32 / 2.0 * (angle as f32 + open_angle).sin(),
                        );
                        pb1.line_to(
                            center.x
                                + base_size as f32 / 2.0
                                    * (angle as f32 + TAU / 2.0 - open_angle).cos(),
                            center.y
                                + base_size as f32 / 2.0
                                    * (angle as f32 + TAU / 2.0 - open_angle).sin(),
                        );
                        pixmap.stroke_path(
                            &pb1.finish().unwrap(),
                            &paint,
                            &stroke,
                            Transform::identity(),
                            None,
                        );
                        let mut pb2 = PathBuilder::new();
                        pb2.move_to(
                            center.x + base_size as f32 / 2.0 * (angle as f32 - open_angle).cos(),
                            center.y + base_size as f32 / 2.0 * (angle as f32 - open_angle).sin(),
                        );
                        pb2.line_to(
                            center.x
                                + base_size as f32 / 2.0
                                    * (angle as f32 - TAU / 2.0 + open_angle).cos(),
                            center.y
                                + base_size as f32 / 2.0
                                    * (angle as f32 - TAU / 2.0 + open_angle).sin(),
                        );
                        pixmap.stroke_path(
                            &pb2.finish().unwrap(),
                            &paint,
                            &stroke,
                            Transform::identity(),
                            None,
                        );
                    }
                    IconShape::RingFragment { start_angle, sweep } => {
                        let mut pb = PathBuilder::new();
                        pb.move_to(
                            center.x + base_size as f32 / 2.0 * start_angle.cos() as f32,
                            center.y + base_size as f32 / 2.0 * start_angle.sin() as f32,
                        );
                        for i in 1..=36 {
                            pb.line_to(
                                center.x
                                    + base_size as f32 / 2.0
                                        * (start_angle + sweep * i as f64 / 36.0).cos() as f32,
                                center.y
                                    + base_size as f32 / 2.0
                                        * (start_angle + sweep * i as f64 / 36.0).sin() as f32,
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
                let digits = digits(rng.gen_range(1..10), rng);
                let base_size = rng.gen_range(0.3..1.0) * min_wh * 0.1;
                let digit_width = base_size as f32 * rng.gen_range(0.3..1.0);
                let digit_height = digit_width * 2.0;
                let thickness = digit_width * 0.28;
                let digit_gap = digit_width * rng.gen_range(0.1..1.0);
                let left_upper = Point::from_xy(
                    rng.gen_range(0.0..width as f32)
                        - rng.gen_range(0.0..1.0)
                            * ((digit_width + digit_gap) * digits.cells.len() as f32 - digit_gap),
                    rng.gen_range(0.0..height as f32) - rng.gen_range(0.0..1.0) * digit_height,
                );
                for (i_cell, cell) in digits.cells.iter().enumerate() {
                    for (j_segment, segment) in cell.segments.iter().enumerate() {
                        if !segment {
                            continue;
                        }
                        let left_upper = Point::from_xy(
                            left_upper.x + i_cell as f32 * (digit_width + digit_gap),
                            left_upper.y,
                        );
                        pixmap.fill_rect(
                            match j_segment {
                                0 => {
                                    // a
                                    Rect::from_xywh(
                                        left_upper.x,
                                        left_upper.y,
                                        digit_width,
                                        thickness,
                                    )
                                }
                                1 => {
                                    // b
                                    Rect::from_xywh(
                                        left_upper.x + digit_width - thickness,
                                        left_upper.y,
                                        thickness,
                                        digit_height / 2.0,
                                    )
                                }
                                2 => {
                                    // c
                                    Rect::from_xywh(
                                        left_upper.x + digit_width - thickness,
                                        left_upper.y + digit_height / 2.0,
                                        thickness,
                                        digit_height / 2.0,
                                    )
                                }
                                3 => {
                                    // d
                                    Rect::from_xywh(
                                        left_upper.x,
                                        left_upper.y + digit_height - thickness,
                                        digit_width,
                                        thickness,
                                    )
                                }
                                4 => {
                                    // e
                                    Rect::from_xywh(
                                        left_upper.x,
                                        left_upper.y + digit_height / 2.0,
                                        thickness,
                                        digit_height / 2.0,
                                    )
                                }
                                5 => {
                                    // f
                                    Rect::from_xywh(
                                        left_upper.x,
                                        left_upper.y,
                                        thickness,
                                        digit_height / 2.0,
                                    )
                                }
                                _ => {
                                    // g
                                    Rect::from_xywh(
                                        left_upper.x,
                                        left_upper.y + digit_height / 2.0 - thickness / 2.0,
                                        digit_width,
                                        thickness,
                                    )
                                }
                            }
                            .unwrap(),
                            &paint,
                            Transform::identity(),
                            None,
                        );
                    }
                }
            }
        }
    }
    pixmap
}
