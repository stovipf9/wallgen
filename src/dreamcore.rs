//! Pure shape/logic for Dreamcore fragments — no drawing. Kept separate from rendering so the
//! design invariants we already fought for, through iteration, are checkable without eyeballing
//! a PNG:
//!   - `eyes`: must never present as a level, equal-size pair — that reads as a face, which is
//!     the exact "single element resolves the whole scene" failure this project already hit once.
//!   - `glyph_words`: must never render as a fully empty box (that reads as "nothing", not
//!     "unreadable writing").
//!   - `icon`: a `RingFragment` must never sweep a full circle — that completes into a real
//!     ring, the same "resolves into one whole shape" failure as a level pair of eyes.
//!   - `digits`: each cell's segment pattern must never match a real digit 0-9 — it should
//!     read as a broken display, not an actual number.

use std::f64::consts::TAU;

use rand::Rng;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EyeMarks {
    pub size_a: f64,
    pub size_b: f64,
    pub gap: f64,
    pub dy: f64, // vertical offset between the two marks — nonzero means "not level"
}

/// Two small marks suggesting a gaze, sized relative to `base_size`. Deliberately asymmetric:
/// unequal size and never level, so they never complete into a face.
pub fn eyes(base_size: f64, rng: &mut impl Rng) -> EyeMarks {
    let size_a = base_size * rng.gen_range(0.5..2.0);
    let size_b = base_size * rng.gen_range(0.5..2.0);

    EyeMarks {
        size_a,
        size_b,
        gap: size_a.max(size_b) * rng.gen_range(1.0..5.0),
        dy: base_size
            * rng.gen_range(f64::MIN_POSITIVE..1.5)
            * if rng.gen_bool(0.5) { -1.0 } else { 1.0 },
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GlyphWords {
    pub cols: usize,
    pub rows: usize,
    pub filled: Vec<Vec<bool>>, // len == cols * rows, row-major
}

/// One asemic "character": a small grid of filled/empty cells that reads as written but
/// resolves to nothing. Never fully empty (falls back to filling at least one cell).
pub fn glyph_words(count: usize, rng: &mut impl Rng) -> GlyphWords {
    let cols = rng.gen_range(2..=3) as usize;
    let rows = rng.gen_range(3..=4) as usize;
    let mut filled = vec![];
    for _ in 0..count {
        let mut grid = vec![false; cols * rows];
        rng.fill(&mut grid[..]);
        if grid.iter().all(|f| !*f) {
            let slot_to_fill = rng.gen_range(0..grid.len());
            grid[slot_to_fill] = true;
        }

        filled.push(grid);
    }

    GlyphWords { cols, rows, filled }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IconShape {
    RingFragment { start_angle: f64, sweep: f64 },
    Cross,
    DiagonalPair { angle: f64 },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Icon {
    pub shape: IconShape,
    pub size: f64,
}

/// One pseudo-pictogram fragment — ring-fragment, cross, or diagonal-pair — that never resolves
/// into an actual sign. (`Arrow` was dropped: unlike the others, a directional arrow reads as a
/// real, functional sign — the same "single element resolves the whole scene" failure as the
/// literal door and the leveled eyes.) In particular a `RingFragment`'s `sweep` must never reach
/// a full circle (that would complete into a real ring, a "resolved" whole shape).
pub fn icon(base_size: f64, rng: &mut impl Rng) -> Icon {
    Icon {
        shape: match rng.gen_range(0..3) {
            0 => IconShape::RingFragment {
                start_angle: rng.gen_range(0.0..TAU),
                sweep: rng.gen_range(0.0..TAU * 0.95),
            },
            1 => IconShape::Cross,
            _ => IconShape::DiagonalPair {
                angle: rng.gen_range(0.0..TAU),
            },
        },
        size: base_size * rng.gen_range(f64::MIN_POSITIVE..2.0),
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SevenSegment {
    pub segments: [bool; 7], // a, b, c, d, e, f, g (standard 7-segment layout)
}

// standard 7-segment encodings for 0-9, the hex letters A-F (lowercase b/d, as on real
// calculator displays, since uppercase B/D are indistinguishable from 8/0), a couple of other
// common Latin letters (H, P), and a few katakana that happen to be reproducible on 7 segments
// (ク, ラ, リ — confirmed by eye against an actual render, not a documented standard), all in
// (a,b,c,d,e,f,g) order. This list grows opportunistically as new coincidental matches are
// spotted; it is not, and cannot be, exhaustive (see the discussion this came out of).
const REAL_CHARACTER_SEGMENTS: [[bool; 7]; 21] = [
    [true, true, true, true, true, true, false],     // 0
    [false, true, true, false, false, false, false], // 1
    [true, true, false, true, true, false, true],    // 2
    [true, true, true, true, false, false, true],    // 3
    [false, true, true, false, false, true, true],   // 4
    [true, false, true, true, false, true, true],    // 5
    [true, false, true, true, true, true, true],     // 6
    [true, true, true, false, false, false, false],  // 7
    [true, true, true, true, true, true, true],      // 8
    [true, true, true, true, false, true, true],     // 9
    [true, true, true, false, true, true, true],     // A
    [false, false, true, true, true, true, true],    // b
    [true, false, false, true, true, true, false],   // C
    [false, true, true, true, true, false, true],    // d
    [true, false, false, true, true, true, true],    // E
    [true, false, false, false, true, true, true],   // F
    [false, true, true, false, true, true, true],    // H
    [true, true, false, false, true, true, true],    // P
    [true, true, true, false, false, true, false],   // ク
    [true, false, true, false, false, false, true],  // ラ
    [false, true, true, true, false, true, false],   // リ
];

#[derive(Debug, Clone, PartialEq)]
pub struct Digits {
    pub cells: Vec<SevenSegment>,
}

/// A row of `count` broken 7-segment-style cells. Each cell's pattern is guaranteed to NOT
/// match any real digit, hex letter, or other recognizable character we've caught so far (see
/// `REAL_CHARACTER_SEGMENTS`) — it should read as a broken meter display, never an actual
/// character. This list is best-effort, not exhaustive.
pub fn digits(count: usize, rng: &mut impl Rng) -> Digits {
    let mut cells: Vec<SevenSegment> = vec![
        SevenSegment {
            segments: [false; 7]
        };
        count
    ];
    for cell in cells.iter_mut() {
        while cell.segments.iter().all(|f| !*f) || REAL_CHARACTER_SEGMENTS.contains(&cell.segments)
        {
            rng.fill(&mut cell.segments);
        }
    }
    Digits { cells }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn eyes_are_never_level() {
        let mut rng = StdRng::seed_from_u64(1);
        for _ in 0..500 {
            let e = eyes(4.0, &mut rng);
            assert_ne!(e.dy, 0.0, "eyes must never be level (that reads as a face)");
        }
    }

    #[test]
    fn eyes_are_never_equal_sized() {
        let mut rng = StdRng::seed_from_u64(2);
        for _ in 0..500 {
            let e = eyes(4.0, &mut rng);
            assert_ne!(e.size_a, e.size_b, "eyes must be unequal in size");
        }
    }

    #[test]
    fn eyes_sizes_stay_within_a_reasonable_range_of_the_base_size() {
        let mut rng = StdRng::seed_from_u64(3);
        for _ in 0..500 {
            let e = eyes(4.0, &mut rng);
            for s in [e.size_a, e.size_b] {
                assert!(s > 0.0 && s < 4.0 * 2.0, "size {s} out of expected range");
            }
        }
    }

    #[test]
    fn glyph_cell_is_never_fully_empty() {
        let mut rng = StdRng::seed_from_u64(4);
        for _ in 0..500 {
            let g = glyph_words(1, &mut rng);
            assert_eq!(g.filled[0].len(), g.cols * g.rows);
            assert!(
                g.filled[0].iter().any(|&f| f),
                "glyph must have at least one filled cell"
            );
        }
    }

    #[test]
    fn glyph_cell_dimensions_stay_within_the_designed_range() {
        // matches the original demo's asemic-character shape: 2-3 cols, 3-4 rows
        let mut rng = StdRng::seed_from_u64(5);
        for _ in 0..500 {
            let g = glyph_words(1, &mut rng);
            assert!((2..=3).contains(&g.cols), "cols {} out of range", g.cols);
            assert!((3..=4).contains(&g.rows), "rows {} out of range", g.rows);
        }
    }

    #[test]
    fn glyph_words_produces_exactly_count_characters_all_non_empty() {
        let mut rng = StdRng::seed_from_u64(9);
        for _ in 0..500 {
            let count = rng.gen_range(1..=8);
            let g = glyph_words(count, &mut rng);
            assert_eq!(
                g.filled.len(),
                count,
                "expected {count} characters, got {}",
                g.filled.len()
            );
            for (i, grid) in g.filled.iter().enumerate() {
                assert_eq!(grid.len(), g.cols * g.rows);
                assert!(
                    grid.iter().any(|&f| f),
                    "character {i} of {count} must not be fully empty"
                );
            }
        }
    }

    #[test]
    fn ring_fragment_never_sweeps_a_full_circle() {
        let mut rng = StdRng::seed_from_u64(6);
        let mut saw_ring_fragment = false;
        for _ in 0..500 {
            if let Icon {
                shape: IconShape::RingFragment { sweep, .. },
                ..
            } = icon(4.0, &mut rng)
            {
                saw_ring_fragment = true;
                assert!(
                    sweep < std::f64::consts::TAU * 0.95,
                    "ring fragment sweep {sweep} too close to a full circle"
                );
            }
        }
        assert!(saw_ring_fragment, "500 draws never produced a RingFragment");
    }

    #[test]
    fn icon_size_stays_within_a_reasonable_range_of_the_base_size() {
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..500 {
            let i = icon(4.0, &mut rng);
            assert!(
                i.size > 0.0 && i.size < 4.0 * 2.0,
                "size {} out of expected range",
                i.size
            );
        }
    }

    #[test]
    fn digits_never_match_a_real_digit() {
        let mut rng = StdRng::seed_from_u64(8);
        for _ in 0..500 {
            let d = digits(5, &mut rng);
            assert_eq!(d.cells.len(), 5);
            for cell in &d.cells {
                assert!(
                    !REAL_CHARACTER_SEGMENTS.contains(&cell.segments),
                    "cell {:?} matches a real digit or hex letter",
                    cell.segments
                );
            }
        }
    }
}
