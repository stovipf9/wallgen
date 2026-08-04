//! wallgen — palette-driven generative wallpaper baker.
//!
//! TDD order (easiest to hardest, each round's tests are already written and failing):
//!   1. palette   — parse a base16 color YAML into 16 RGB slots
//!   2. noise     — gradient (Perlin) noise + fbm, the base of Flow
//!   3. flow      — curl-from-potential and RK2 streamline advection (pure, no drawing)
//!   4. dreamcore — small pure-logic invariants (asymmetric eyes, non-empty glyph cells)
//!
//! Rendering to actual pixels (tiny-skia) and the CLI are a later round, once these are green —
//! deliberately not test-first here since "does this look right" isn't a unit-testable property;
//! that part gets checked by eye against the baked PNG, the same way the canvas demo was.

pub mod dreamcore;
pub mod flow;
pub mod noise;
pub mod palette;
pub mod render;
