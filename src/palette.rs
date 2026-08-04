//! Parses a base16 color palette from YAML (16 `baseNN` slots, bare hex values, no `#`).

use std::collections::HashMap;
use wallgen_derive::PaletteFields;
use yaml_rust::YamlLoader;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rgb(pub u8, pub u8, pub u8);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PaletteFields)]
pub struct Palette {
    pub base00: Rgb,
    pub base01: Rgb,
    pub base02: Rgb,
    pub base03: Rgb,
    pub base04: Rgb,
    pub base05: Rgb,
    pub base06: Rgb,
    pub base07: Rgb,
    pub base08: Rgb,
    pub base09: Rgb,
    pub base0a: Rgb,
    pub base0b: Rgb,
    pub base0c: Rgb,
    pub base0d: Rgb,
    pub base0e: Rgb,
    pub base0f: Rgb,
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

    #[test]
    fn all_returns_exactly_16_colors_in_base00_to_base0f_order() {
        let p = Palette::parse(REAL_COLORS_YAML).unwrap();
        let all = p.all();
        assert_eq!(all.len(), 16);
        assert_eq!(all[0], p.base00);
        assert_eq!(all[15], p.base0f);
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
