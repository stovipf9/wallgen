//! Parses a base16 color palette from YAML (16 `baseNN` slots, bare hex values, no `#`).

use std::collections::HashMap;
use wallgen_derive::PaletteFields;
use yaml_rust::YamlLoader;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rgb(pub u8, pub u8, pub u8);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PaletteFields)]
pub struct Palette {
    base00: Rgb,
    base01: Rgb,
    base02: Rgb,
    base03: Rgb,
    base04: Rgb,
    base05: Rgb,
    base06: Rgb,
    base07: Rgb,
    base08: Rgb,
    base09: Rgb,
    base0a: Rgb,
    base0b: Rgb,
    base0c: Rgb,
    base0d: Rgb,
    base0e: Rgb,
    base0f: Rgb,
}

impl Palette {
    /// Parse the `colors:` block. Expected line shape: `  baseNN: "rrggbb"  # comment`.
    /// Keys are case-insensitive on the hex-digit suffix (`base0A` and `base0a` both map to `base0a`).
    /// Returns `Err` if any of the 16 required slots is missing or a hex value doesn't parse.
    pub fn parse(yaml: &str) -> Result<Palette, String> {
        let mut colors: HashMap<String, Rgb> = HashMap::new();
        for (key, value) in YamlLoader::load_from_str(yaml).map_err(|err| err.to_string())?[0]
            ["colors"]
            .as_hash()
            .ok_or("yaml could not be interpreted as hash")?
        {
            let hex_rgb = u32::from_str_radix(
                value.as_str().ok_or(format!(
                    "key={:?}, value={:?}: value could not be interpreted as string",
                    key, value
                ))?,
                16,
            )
            .map_err(|err| err.to_string())?;
            colors.insert(
                key.as_str()
                    .ok_or(format!(
                        "key={:?}, value={:?}: key could not be interpreted as string",
                        key, value
                    ))?
                    .to_ascii_lowercase(),
                Rgb((hex_rgb >> 16) as u8, (hex_rgb >> 8) as u8, hex_rgb as u8),
            );
        }
        Palette::from_colors(colors)
    }

    /// `base00` — the shade of a place nothing has been drawn.
    pub fn background(&self) -> Rgb {
        self.base00
    }

    /// `base01`–`base02` — a shade off the background, near enough to read as ground rather than as
    /// a mark. What a wash or a fill is allowed to be.
    pub fn surfaces(&self) -> [Rgb; 2] {
        [self.base01, self.base02]
    }

    /// `base08`–`base0f` — the saturated hues, the ones that read as a mark against `background`.
    /// The eight below them are a greyscale ramp and do not, which is the whole content of the
    /// boundary: these are the colors a renderer may draw *with*.
    ///
    /// Order is declaration order and means nothing. There is no first accent and no last one —
    /// pick with `choose`, never by index.
    pub fn accents(&self) -> [Rgb; 8] {
        [
            self.base08,
            self.base09,
            self.base0a,
            self.base0b,
            self.base0c,
            self.base0d,
            self.base0e,
            self.base0f,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // real base16 OneDark values (not synthetic test data)
    const REAL_COLORS_YAML: &str = r#"
colors:
  base00: "282c34"  # default background
  base01: "353b45"  # lighter bg (status line, line numbers)
  base02: "3e4451"  # selection bg
  base03: "545862"  # comments / invisibles
  base04: "565c64"  # dark fg (status line)
  base05: "abb2bf"  # default fg
  base06: "b6bdca"  # light fg
  base07: "c8ccd4"  # light bg
  base08: "e06c75"  # red
  base09: "d19a66"  # orange
  base0A: "e5c07b"  # yellow
  base0B: "98c379"  # green
  base0C: "56b6c2"  # cyan
  base0D: "61afef"  # blue
  base0E: "c678dd"  # magenta/purple
  base0F: "be5046"  # brown
"#;

    #[test]
    fn parses_all_16_slots_from_the_real_colors_yaml() {
        let p = Palette::parse(REAL_COLORS_YAML).expect("should parse");
        assert_eq!(p.base00, Rgb(0x28, 0x2c, 0x34));
        assert_eq!(p.base0d, Rgb(0x61, 0xaf, 0xef)); // blue — used as --accent in the demo
        assert_eq!(p.base0f, Rgb(0xbe, 0x50, 0x46));
    }

    #[test]
    fn is_case_insensitive_on_the_hex_digit_suffix() {
        let lower = REAL_COLORS_YAML
            .replace("base0A", "base0a")
            .replace("base0F", "base0f");
        let p = Palette::parse(&lower).expect("should parse lowercase keys too");
        assert_eq!(p.base0a, Rgb(0xe5, 0xc0, 0x7b));
    }

    /// Two mechanisms have to agree about where base16 puts things, and neither can check itself.
    /// `all` is generated from field declaration order; the roles name their slots by hand. A role
    /// that reaches for the wrong slot, or a field order that stops matching the scheme, shows up
    /// only as the two disagreeing — the endpoints of `all` survive any reordering of the middle,
    /// and the roles look right in isolation whatever they name.
    ///
    /// This leans on the sixteen colors being distinct, which they are in a real base16 scheme:
    /// equal ones would let a wrong slot pass.
    #[test]
    fn the_roles_pick_out_the_base16_slots_they_claim() {
        let p = Palette::parse(REAL_COLORS_YAML).unwrap();
        let all = p.all();

        assert_eq!(all.len(), 16);
        assert_eq!(all[0], p.background());
        assert_eq!(all[1..3], p.surfaces());
        assert_eq!(all[8..], p.accents());
    }

    #[test]
    fn rejects_yaml_missing_a_required_slot() {
        let broken = "colors:\n  base00: \"282c34\"\n";
        assert!(Palette::parse(broken).is_err());
    }

    #[test]
    fn rejects_a_malformed_hex_value() {
        let broken = REAL_COLORS_YAML.replace("282c34", "not-a-color");
        assert!(Palette::parse(&broken).is_err());
    }
}
