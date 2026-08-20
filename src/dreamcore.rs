//! Pure shape/logic for Dreamcore fragments — no drawing. Kept separate from rendering so the
//! design invariants we already fought for, through iteration, are checkable without eyeballing
//! a PNG:
//!   - `eyes`: must never present as a level, equal-size pair — that reads as a face, which is
//!     the exact "single element resolves the whole scene" failure this project already hit once.
//!     Neither half needs a check any more: `DY_RATIO_RANGE` and `SMALLER_SIZE_RANGE` between them
//!     make such a pair unrepresentable.
//!   - `glyph_word`: no cell may come out with every dot unset (an empty box reads as "nothing",
//!     not as "unreadable writing").
//!   - `icon_shape`: a `RingFragment` must never sweep a full circle — that completes into a real
//!     ring, the same "resolves into one whole shape" failure as a level pair of eyes. This one has
//!     since stopped needing a check at all: `Sweep` makes it unrepresentable.
//!   - `digits`: each cell's segment pattern must never match a real digit 0-9 — it should
//!     read as a broken display, not an actual number.

use std::{
    f64::consts::TAU,
    ops::{Range, RangeInclusive},
};

use rand::{
    distributions::{Distribution, Standard},
    Rng,
};

/// Two small marks suggesting a gaze. Deliberately asymmetric: unequal size and never level, so
/// they never complete into a face. Both halves hold by construction — "never level" because
/// `DY_RATIO_RANGE` excludes zero, "unequal size" because `SMALLER_SIZE_RANGE` is half-open below
/// one — so a pair of equal or level marks is not a thing this type can hold, rather than one a
/// test happens not to have seen.
///
/// Every length here is in widths of the bigger mark, that mark being one by definition, so nothing
/// in this type is in pixels. `render` multiplies by whatever the bigger mark is worth on its
/// canvas. Only the smaller mark's size is stored, since the bigger one is the unit.
///
/// The pair is described by which mark is on top rather than by which is on the left, because that
/// is the axis the constraint lives on: how far the lower mark may drop is bounded by the upper
/// one's height. Left and right are then a free choice, made by `is_upper_left` and used only for
/// layout.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EyeMarks {
    is_upper_smaller: bool,
    is_upper_left: bool,
    smaller_size: f64,
    gap: f64,
    /// How far the lower mark sits below the upper one, as a fraction of the upper one — a ratio
    /// rather than a length, which is why it is the one field not in bigger-mark widths. `dy`
    /// converts.
    dy_ratio: f64,
}

impl EyeMarks {
    /// The unit everything else is measured in, so one by definition.
    const BIGGER_EYE_SIZE: f64 = 1.0;
    /// The other mark, as a fraction of the bigger one. Being a `Range` rather than a
    /// `RangeInclusive` is what carries "unequal size": the excluded upper end is the whole
    /// enforcement, so widening this to `..=1.0` would quietly give the pair back its face. How far
    /// below one the range starts is the separate, aesthetic half — the point at which the
    /// difference stops being visible — and no assertion reading this constant can check that.
    const SMALLER_SIZE_RANGE: Range<f64> = 0.5..1.0;
    /// Space between the marks, in bigger-mark widths — the pair reads as a gaze rather than as two
    /// unrelated dots only while the gap stays in scale with what it separates.
    const GAP_RANGE: RangeInclusive<f64> = 1.0..=5.0;
    /// How far out of level the pair sits, as a fraction of the upper mark. Both ends carry
    /// something, and the type of the range is half of it — as with `SMALLER_SIZE_RANGE`.
    ///
    /// The excluded upper end is what keeps the two marks overlapping vertically: the lower mark
    /// drops by less than the upper one's height, so the pair always shares a band and reads as one
    /// thing. At a ratio of one they would merely touch, and past it they would be two stacked dots.
    /// The lower end keeps "never level" true rather than merely representable: a smaller one would
    /// be a level pair drawn with a nonzero number in it.
    const DY_RATIO_RANGE: Range<f64> = 0.25..1.0;

    /// Every field is an independent draw, so nothing here has to happen in a particular order.
    /// Taking one field's bound from another's value is the shape to avoid: it gives the bound
    /// somewhere to be computed, and so somewhere to be computed wrongly.
    pub fn sample(rng: &mut impl Rng) -> Self {
        Self {
            is_upper_smaller: rng.gen(),
            is_upper_left: rng.gen(),
            smaller_size: Self::BIGGER_EYE_SIZE * rng.gen_range(Self::SMALLER_SIZE_RANGE),
            gap: Self::BIGGER_EYE_SIZE * rng.gen_range(Self::GAP_RANGE),
            dy_ratio: rng.gen_range(Self::DY_RATIO_RANGE),
        }
    }

    /// The drop from the upper mark to the lower one, in bigger-mark widths — the one place the
    /// stored ratio is turned into a length in this type's own unit.
    fn dy(&self) -> f64 {
        self.dy_ratio * self.upper_size()
    }

    fn lower_size(&self) -> f64 {
        if self.is_upper_smaller {
            Self::BIGGER_EYE_SIZE
        } else {
            self.smaller_size
        }
    }

    fn upper_size(&self) -> f64 {
        if self.is_upper_smaller {
            self.smaller_size
        } else {
            Self::BIGGER_EYE_SIZE
        }
    }

    /// Total span across, in bigger-mark widths
    pub fn width(&self) -> f64 {
        self.smaller_size + self.gap + Self::BIGGER_EYE_SIZE
    }

    /// Total span down, in bigger-mark widths
    pub fn height(&self) -> f64 {
        self.upper_size().max(self.dy() + self.lower_size())
    }

    /// Each mark as (left, top, size), in bigger-mark widths, from the pair's own top-left corner.
    pub fn marks(&self) -> [(f64, f64, f64); 2] {
        if self.is_upper_left {
            [
                (0.0, 0.0, self.upper_size()),
                (self.upper_size() + self.gap, self.dy(), self.lower_size()),
            ]
        } else {
            [
                (0.0, self.dy(), self.lower_size()),
                (self.lower_size() + self.gap, 0.0, self.upper_size()),
            ]
        }
    }
}

/// One asemic word: a row of cells, each a small grid of set and unset dots that reads as written
/// but resolves to nothing. Every cell in a word shares one `cols` x `rows` shape, the way a
/// typeface holds its characters to one body.
///
/// No cell is ever left with every dot unset — an empty box among filled ones reads as a space, and
/// a word of nothing but spaces reads as nothing at all. The fields are private and `sample` is the
/// only way in, so that holds of every `GlyphWord` there is.
///
/// Three sizes are in play and each has one name: a **dot** is the smallest square, a **cell** is
/// one character's grid of them, and a **word** is the row of cells. That is braille's vocabulary,
/// where a cell is likewise the character rather than the mark, and `Digits` uses `cell` the same
/// way.
///
/// Every length here is in dot widths, the dot being one by definition, so nothing in this type is
/// in pixels. `render` multiplies by whatever a dot is worth on its canvas.
#[derive(Debug, Clone, PartialEq)]
pub struct GlyphWord {
    /// Space between one cell and the next, in dot widths.
    cell_gap: f64,
    /// Dots across one cell, and down it. Every cell in the word shares them.
    cols: usize,
    rows: usize,
    /// One entry per cell, each `cols * rows` dots in row-major order.
    filled: Vec<Vec<bool>>,
}

impl GlyphWord {
    /// The unit everything else is measured in, so one by definition.
    const DOT_SIZE: f64 = 1.0;
    /// Space beside a dot, and the same again below it. Equal to the dot, so a cell is an even
    /// checker: no stroke weight is implied, and it reads as a matrix display rather than as a
    /// letterform with a thickness. Equal pitch on both axes is also what makes `ROW_RANGE` above
    /// `COL_RANGE` produce a cell taller than it is wide.
    const DOT_GAP: f64 = 1.0;
    /// Space between cells, in dot widths. Above `DOT_GAP` is what separates the characters — at
    /// exactly `DOT_GAP` the whole word would be one even grid and no cell boundary would be
    /// readable.
    const CELL_GAP_RATIO_RANGE: RangeInclusive<f64> = 1.5..=2.0;
    /// Dots across one cell. Two or three, because one column reads as a stroke rather than as a
    /// character, and four starts to resolve into a shape the eye can name.
    const COL_RANGE: RangeInclusive<usize> = 2..=3;
    /// Dots down one cell, kept above `COL_RANGE` so a cell stands taller than it is wide, the
    /// proportion writing tends to have. This holds only while the dot pitch is the same on both
    /// axes, which `render` arranges by spacing dots one dot-width apart in each direction.
    const ROW_RANGE: RangeInclusive<usize> = 3..=4;
    /// Cells in one word. From one, which reads as a mark rather than as writing, to enough to
    /// read as a word without becoming a line of prose.
    const GLYPH_WORD_LENGTH_RANGE: RangeInclusive<usize> = 1..=10;

    /// A cell that comes up empty has one dot set at random rather than being redrawn, so the
    /// number of draws does not depend on how the dice fall.
    pub fn sample(rng: &mut impl Rng) -> Self {
        let cols = rng.gen_range(Self::COL_RANGE);
        let rows = rng.gen_range(Self::ROW_RANGE);
        let cell_count = rng.gen_range(Self::GLYPH_WORD_LENGTH_RANGE);

        let mut filled = vec![];
        for _ in 0..cell_count {
            let mut cell = vec![false; cols * rows];
            rng.fill(&mut cell[..]);
            if cell.iter().all(|f| !*f) {
                let dot_to_fill = rng.gen_range(0..cell.len());
                cell[dot_to_fill] = true;
            }

            filled.push(cell);
        }

        Self {
            cell_gap: Self::DOT_SIZE * rng.gen_range(Self::CELL_GAP_RATIO_RANGE),
            cols,
            rows,
            filled,
        }
    }

    /// Across one cell: `cols` dots with a gap between each neighboring pair, so one fewer gap
    /// than dots.
    fn cell_width(&self) -> f64 {
        self.cols as f64 * (Self::DOT_SIZE + Self::DOT_GAP) - Self::DOT_GAP
    }

    /// Across the whole word, and down it, in dot widths. `render` asks `ref_point` for a
    /// rectangle this size and then draws `dots` inside it, so the two have to describe one
    /// layout — which is what `every_dot_lands_inside_the_word_it_is_reported_as_filling` holds
    /// them to. Height ignores the cell count because every cell is one row tall.
    pub fn width(&self) -> f64 {
        self.filled.len() as f64 * (self.cell_width() + self.cell_gap) - self.cell_gap
    }

    pub fn height(&self) -> f64 {
        self.rows as f64 * (Self::DOT_SIZE + Self::DOT_GAP) - Self::DOT_GAP
    }

    /// Every set dot as (left, top, size), in dot widths from the word's own top-left corner.
    ///
    /// Unset dots are absent rather than reported, so a word whose first cell has an empty left
    /// column starts short of zero — `width` and `height` describe the space the word occupies,
    /// this describes only the part of it that is drawn.
    pub fn dots(&self) -> Vec<(f64, f64, f64)> {
        let mut dots: Vec<(f64, f64, f64)> = vec![];
        let cell_width = self.cell_width();
        for (i_cell, cell) in self.filled.iter().enumerate() {
            for (j_dot, dot_filled) in cell.iter().enumerate() {
                if !*dot_filled {
                    continue;
                }
                dots.push((
                    (cell_width + self.cell_gap) * i_cell as f64
                        + (j_dot % self.cols) as f64 * (Self::DOT_SIZE + Self::DOT_GAP),
                    (j_dot / self.cols) as f64 * (Self::DOT_SIZE + Self::DOT_GAP),
                    Self::DOT_SIZE,
                ));
            }
        }
        dots
    }
}

/// An angle in radians. The type carries no range: angles are periodic, so a value outside
/// `0.0..TAU` is as valid as one inside. `Standard` drawing from a full turn is that
/// distribution's support, not a constraint on the type.
///
/// A newtype rather than a bare `f64` so an angle cannot be swapped for a length stored beside it,
/// and so the full-turn draw has one home.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Angle(f64);

impl Angle {
    pub fn radians(self) -> f64 {
        self.0
    }
}
impl Distribution<Angle> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Angle {
        Angle(rng.gen_range(0.0..TAU))
    }
}

/// An angular interval in radians: an amount turned rather than a direction.
///
/// Unlike `Angle`, the range here is an invariant and not merely a distribution's support. The
/// field is private and `Standard` is the only constructor, so every `Sweep` that exists is under a
/// full turn — an arc closing into a whole ring is unrepresentable rather than merely untested.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sweep(f64);

impl Sweep {
    /// The largest fraction of a full turn a sweep may cover. Under 1 is what keeps the arc from
    /// closing into a real ring; how far under is a judgment about when the remaining gap stops
    /// reading as a gap, and no assertion against this constant can check that half.
    const MAX_RATIO: f64 = 0.95;

    pub fn radians(self) -> f64 {
        self.0
    }
}
impl Distribution<Sweep> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Sweep {
        Sweep(rng.gen_range(0.0..Sweep::MAX_RATIO * TAU))
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// One pseudo-pictogram fragment — ring fragment, cross, or parallel chords — that never resolves
/// into an actual sign. (`Arrow` was dropped: unlike the others, a directional arrow reads as a
/// real, functional sign — the same "single element resolves the whole scene" failure as the
/// literal door and the leveled eyes.) The one constraint that used to be stated here, that a
/// `RingFragment` never reaches a full circle, is now carried by `Sweep`.
pub enum IconShape {
    RingFragment { start_angle: Angle, sweep: Sweep },
    Cross,
    ParallelChords { angle: Angle, chord_length: f64 },
}

impl IconShape {
    pub fn sample(rng: &mut impl Rng) -> Self {
        match rng.gen_range(0..3) {
            0 => Self::RingFragment {
                start_angle: rng.gen(),
                sweep: rng.gen(),
            },
            1 => Self::Cross,
            _ => Self::ParallelChords {
                angle: rng.gen(),
                chord_length: rng.gen_range(Self::length_range()),
            },
        }
    }

    fn length_range() -> RangeInclusive<f64> {
        2.0 * (1.0 / 8.0 * TAU).cos()..=2.0 * (1.0 / 16.0 * TAU).cos()
    }

    pub fn cross_edges() -> [(Angle, Angle); 2] {
        [
            (Angle(0.0), Angle(TAU / 2.0)),
            (Angle(TAU / 4.0), Angle(3.0 * TAU / 4.0)),
        ]
    }

    pub fn chords(angle: Angle, chord_length: f64) -> [(Angle, Angle); 2] {
        let angle = angle.radians();
        let theta = (chord_length / 2.0).acos();
        [
            (Angle(angle + theta), Angle(angle + TAU / 2.0 - theta)),
            (Angle(angle - theta), Angle(angle - TAU / 2.0 + theta)),
        ]
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SevenSegment {
    segments: [bool; 7], // a, b, c, d, e, f, g (standard 7-segment layout)
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

/// A row of broken seven-segment cells, to read as a failing meter display rather than as an
/// actual number.
///
/// No cell's pattern matches any real digit, hex letter, or other recognizable character caught so
/// far. The fields are private and `sample` is the only way in, so that holds of every `Digits`
/// there is — but only as far as `REAL_CHARACTER_SEGMENTS` reaches, and that table is best-effort
/// rather than exhaustive. The enforcement is airtight; the predicate it enforces is not.
///
/// Every length here is in cell widths, a cell being one by definition, so nothing in this type is
/// in pixels. `render` multiplies by whatever a cell is worth on its canvas. `cell` names the
/// character, as it does in `GlyphWord`.
#[derive(Debug, Clone, PartialEq)]
pub struct Digits {
    /// How wide a lit segment is drawn, in cell widths.
    thickness: f64,
    cells: Vec<SevenSegment>,
    /// Space between one cell and the next, in cell widths.
    digit_gap: f64,
}

impl Digits {
    /// The unit everything else is measured in, so one by definition.
    const DIGIT_WIDTH: f64 = 1.0;
    /// Twice as tall as wide, fixed rather than drawn: it is what makes a seven-segment cell read
    /// as one, the two stacked squares the layout is built from.
    const DIGIT_HEIGHT: f64 = 2.0;
    /// Segment thickness, against the cell's *width* rather than its height, so the three
    /// horizontal bars keep their weight while `DIGIT_HEIGHT` stretches the vertical ones. The
    /// span covers a thin LCD through a chunky LED.
    ///
    /// Not a stroke weight, despite reading like one: the segments are filled rectangles and this
    /// is one of their two dimensions, so changing it changes the letterform rather than how
    /// heavily it is drawn. That is why it lives here and `render`'s `STROKE_WIDTH_RATIO_RANGE`,
    /// which really is a stroke, does not.
    const THICKNESS_RATIO_RANGE: RangeInclusive<f64> = 0.14..=0.28;
    /// Cells in one readout. Long enough to look like a reading rather than a mark, short enough
    /// that a viewer does not start looking for a pattern in it.
    const DIGITS_LENGTH_RANGE: RangeInclusive<usize> = 1..=10;
    /// Space between cells, in cell widths. Reaching zero would run the readout into one block; a
    /// full cell width is where it stops being one readout.
    const DIGIT_GAP_RATIO_RANGE: RangeInclusive<f64> = 0.1..=1.0;

    pub fn sample(rng: &mut impl Rng) -> Self {
        let mut cells: Vec<SevenSegment> = vec![
            SevenSegment {
                segments: [false; 7]
            };
            rng.gen_range(Self::DIGITS_LENGTH_RANGE)
        ];
        for cell in cells.iter_mut() {
            while cell.segments.iter().all(|f| !*f)
                || REAL_CHARACTER_SEGMENTS.contains(&cell.segments)
            {
                rng.fill(&mut cell.segments);
            }
        }
        Self {
            thickness: Self::DIGIT_WIDTH * rng.gen_range(Self::THICKNESS_RATIO_RANGE),
            cells,
            digit_gap: Self::DIGIT_WIDTH * rng.gen_range(Self::DIGIT_GAP_RATIO_RANGE),
        }
    }

    /// Across the whole readout, and down it, in cell widths. `render` asks `ref_point` for a
    /// rectangle this size and then draws `segments` inside it, so the two have to describe one
    /// layout. Height is the cell's own, since a readout is one row.
    pub fn width(&self) -> f64 {
        (Self::DIGIT_WIDTH + self.digit_gap) * self.cells.len() as f64 - self.digit_gap
    }

    pub fn height(&self) -> f64 {
        Self::DIGIT_HEIGHT
    }

    /// Every lit segment as (left, top, width, height), in cell widths from the readout's own
    /// top-left corner.
    ///
    /// The seven positions are the standard layout `SevenSegment` names in its declaration —
    /// `a` across the top, `b` and `c` down the right, `d` across the bottom, `e` and `f` down the
    /// left, `g` across the middle. That layout was `render`'s to know until now, which left the
    /// type claiming a shape it did not describe.
    pub fn segments(&self) -> Vec<(f64, f64, f64, f64)> {
        let mut segments: Vec<(f64, f64, f64, f64)> = vec![];
        for (i_cell, cell) in self.cells.iter().enumerate() {
            for (j_segment, segment) in cell.segments.iter().enumerate() {
                if !segment {
                    continue;
                }
                let cell_x = i_cell as f64 * (Self::DIGIT_WIDTH + self.digit_gap);
                segments.push(match j_segment {
                    0 => {
                        // a
                        (cell_x, 0.0, Self::DIGIT_WIDTH, self.thickness)
                    }
                    1 => {
                        // b
                        (
                            cell_x + Self::DIGIT_WIDTH - self.thickness,
                            0.0,
                            self.thickness,
                            Self::DIGIT_HEIGHT / 2.0,
                        )
                    }
                    2 => {
                        // c
                        (
                            cell_x + Self::DIGIT_WIDTH - self.thickness,
                            Self::DIGIT_HEIGHT / 2.0,
                            self.thickness,
                            Self::DIGIT_HEIGHT / 2.0,
                        )
                    }
                    3 => {
                        // d
                        (
                            cell_x,
                            Self::DIGIT_HEIGHT - self.thickness,
                            Self::DIGIT_WIDTH,
                            self.thickness,
                        )
                    }
                    4 => {
                        // e
                        (
                            cell_x,
                            Self::DIGIT_HEIGHT / 2.0,
                            self.thickness,
                            Self::DIGIT_HEIGHT / 2.0,
                        )
                    }
                    5 => {
                        // f
                        (cell_x, 0.0, self.thickness, Self::DIGIT_HEIGHT / 2.0)
                    }
                    _ => {
                        // g
                        (
                            cell_x,
                            Self::DIGIT_HEIGHT / 2.0 - self.thickness / 2.0,
                            Self::DIGIT_WIDTH,
                            self.thickness,
                        )
                    }
                });
            }
        }
        segments
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn eyes_are_never_level() {
        let mut rng = StdRng::seed_from_u64(1);
        for _ in 0..500 {
            let e = EyeMarks::sample(&mut rng);
            assert_ne!(
                e.dy(),
                0.0,
                "eyes must never be level (that reads as a face)"
            );
        }
    }

    /// `marks` and `width`/`height` are two readings of one layout, and `render` trusts them to
    /// agree: it asks `ref_point` for a rectangle of that size and then draws the marks inside it.
    /// They are computed separately, so nothing but this makes them agree — and when the enclosing
    /// rectangle was `render`'s to work out, it got `dy` wrong and put a mark outside the frame's
    /// placement budget.
    ///
    /// Both marks together must therefore start exactly at the corner and reach exactly as far as
    /// the pair says it does.
    #[test]
    fn the_marks_fill_the_pair_they_are_reported_as_filling() {
        let mut rng = StdRng::seed_from_u64(11);
        for _ in 0..500 {
            let e = EyeMarks::sample(&mut rng);
            let marks = e.marks();

            let left = marks.iter().map(|&(x, ..)| x).fold(f64::MAX, f64::min);
            let top = marks.iter().map(|&(_, y, _)| y).fold(f64::MAX, f64::min);
            let right = marks
                .iter()
                .map(|&(x, _, size)| x + size)
                .fold(f64::MIN, f64::max);
            let bottom = marks
                .iter()
                .map(|&(_, y, size)| y + size)
                .fold(f64::MIN, f64::max);

            for (name, got, want) in [
                ("left", left, 0.0),
                ("top", top, 0.0),
                ("width", right - left, e.width()),
                ("height", bottom - top, e.height()),
            ] {
                assert!(
                    (got - want).abs() < 1e-9,
                    "{name} is {got} against the reported {want}, from marks {marks:?}"
                );
            }
        }
    }

    /// `dots` and `width`/`height` are two readings of one layout, and `render` trusts them to
    /// agree: it asks `ref_point` for a rectangle of that size and then draws the dots inside it.
    /// The same split cost a mark its place once already, on the pair of eyes.
    ///
    /// Unlike the pair, though, these two do not have to coincide. `width` and `height` describe
    /// the space the word occupies, dots only the part of it that is set, and a cell whose left
    /// column or top row happens to be all unset leaves a real margin. So the assertion is
    /// containment rather than equality — every dot inside the rectangle, none of it hanging out.
    #[test]
    fn every_dot_lands_inside_the_word_it_is_reported_as_filling() {
        let mut rng = StdRng::seed_from_u64(12);
        for _ in 0..500 {
            let word = GlyphWord::sample(&mut rng);
            let dots = word.dots();

            // A dot's far edge and the word's own extent are the same sum of the same terms in a
            // different order, so on `f32` they part company in the last few bits. Anything beyond
            // that is a layout disagreement rather than arithmetic.
            let reaches = |edge: f64, extent: f64| edge <= extent * (1.0 + 8.0 * f64::EPSILON);

            assert!(
                !dots.is_empty(),
                "a word of {} cells drew nothing",
                word.filled.len()
            );
            for (x, y, size) in dots {
                assert!(
                    x >= 0.0 && reaches(x + size, word.width()),
                    "a dot spans {x}..{} across a word {} wide",
                    x + size,
                    word.width()
                );
                assert!(
                    y >= 0.0 && reaches(y + size, word.height()),
                    "a dot spans {y}..{} down a word {} tall",
                    y + size,
                    word.height()
                );
            }
        }
    }

    #[test]
    fn glyph_cell_is_never_fully_empty() {
        let mut rng = StdRng::seed_from_u64(4);
        for _ in 0..500 {
            let g = GlyphWord::sample(&mut rng);
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
            let g = GlyphWord::sample(&mut rng);
            assert!(
                GlyphWord::COL_RANGE.contains(&g.cols),
                "cols {} out of range",
                g.cols
            );
            assert!(
                GlyphWord::ROW_RANGE.contains(&g.rows),
                "rows {} out of range",
                g.rows
            );
        }
    }

    #[test]
    fn glyph_words_produces_exactly_count_characters_all_non_empty() {
        let mut rng = StdRng::seed_from_u64(9);
        for _ in 0..500 {
            let g = GlyphWord::sample(&mut rng);
            for (i, grid) in g.filled.iter().enumerate() {
                assert_eq!(grid.len(), g.cols * g.rows);
                assert!(
                    grid.iter().any(|&f| f),
                    "character {i} must not be fully empty"
                );
            }
        }
    }

    #[test]
    fn ring_fragment_never_sweeps_a_full_circle() {
        let mut rng = StdRng::seed_from_u64(6);
        let mut saw_ring_fragment = false;
        for _ in 0..500 {
            if let IconShape::RingFragment { sweep, .. } = IconShape::sample(&mut rng) {
                let sweep = sweep.radians();
                saw_ring_fragment = true;
                assert!(
                    sweep < Sweep::MAX_RATIO * TAU,
                    "ring fragment sweep {sweep} too close to a full circle"
                );
            }
        }
        assert!(saw_ring_fragment, "500 draws never produced a RingFragment");
    }

    /// `chords` hands back angles and `render` turns them into points on a circle of some radius,
    /// so the length actually drawn is never stated anywhere — it falls out of the trigonometry.
    /// Which is how the two halves of this once disagreed by a factor of two: `chord_length` was a
    /// multiple of the radius on one side of the file and of the diameter on the other, and since
    /// the standard chord relation carries a two either way, both spellings looked right.
    ///
    /// Drawn on the unit circle so a radius of one makes the reported length directly comparable.
    #[test]
    fn a_chord_comes_out_the_length_it_was_asked_for() {
        let mut rng = StdRng::seed_from_u64(13);
        let mut saw_chords = false;
        for _ in 0..500 {
            let IconShape::ParallelChords {
                angle,
                chord_length,
            } = IconShape::sample(&mut rng)
            else {
                continue;
            };
            saw_chords = true;

            for (start, end) in IconShape::chords(angle, chord_length) {
                let start = start.radians();
                let end = end.radians();
                let drawn = (end.cos() - start.cos()).hypot(end.sin() - start.sin());
                assert!(
                    (drawn - chord_length).abs() < 1e-9,
                    "asked for {chord_length} and drew {drawn}"
                );
            }
        }
        assert!(saw_chords, "500 draws never produced a ParallelChords");
    }

    /// The pair is two parallel chords either side of the centre, which is what the variant is
    /// named for. Nothing in the construction says so — it comes out of the two chords being
    /// mirror images about the axis through `angle` — and an earlier comment claimed instead that
    /// a wide enough opening would square them into a cross, which cannot happen at any value.
    #[test]
    fn the_two_chords_are_parallel_and_straddle_the_centre() {
        let mut rng = StdRng::seed_from_u64(14);
        for _ in 0..500 {
            let IconShape::ParallelChords {
                angle,
                chord_length,
            } = IconShape::sample(&mut rng)
            else {
                continue;
            };

            let [first, second] = IconShape::chords(angle, chord_length).map(|(start, end)| {
                let start = start.radians();
                let end = end.radians();
                let (from, to) = ((start.cos(), start.sin()), (end.cos(), end.sin()));
                let heading = (to.1 - from.1).atan2(to.0 - from.0).rem_euclid(TAU / 2.0);
                // Signed distance from the centre, positive on one side of the chord's line and
                // negative on the other, so a pair that straddles the centre sums to zero.
                let offset = ((to.0 - from.0) * from.1 - from.0 * (to.1 - from.1)) / chord_length;
                (heading, offset)
            });

            assert!(
                (first.0 - second.0).abs() < 1e-9,
                "chords head {} and {} degrees apart",
                first.0.to_degrees(),
                second.0.to_degrees()
            );
            assert!(
                (first.1 + second.1).abs() < 1e-9,
                "chords sit {} and {} from the centre rather than either side of it",
                first.1,
                second.1
            );
            assert!(
                first.1.abs() > 1e-9,
                "the chords have collapsed onto one line through the centre"
            );
        }
    }

    #[test]
    fn digits_never_match_a_real_digit() {
        let mut rng = StdRng::seed_from_u64(8);
        for _ in 0..500 {
            let d = Digits::sample(&mut rng);
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
