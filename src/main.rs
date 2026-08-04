use std::fs::File;
use std::io::Read;

use clap::{Parser, ValueEnum};
use rand::{rngs::StdRng, SeedableRng};
use rand::{thread_rng, RngCore};
use wallgen::palette::Palette;
use wallgen::render::{render_dreamcore, render_flow};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Style {
    Dreamcore,
    Flow,
}

#[derive(Debug, Parser)]
#[command(author, version, about)]
struct Args {
    /// seed
    #[arg(long, value_name = "SEED")]
    seed: Option<u64>,

    /// width
    #[arg(long, default_value_t = 1920, value_name = "WIDTH")]
    width: u32,

    /// height
    #[arg(long, default_value_t = 1080, value_name = "HEIGHT")]
    height: u32,

    /// style
    #[arg(long, value_enum, default_value_t = Style::Dreamcore, value_name = "STYLE")]
    style: Style,

    /// count: dreamcore fragments or flow streamlines, depending on `style`.
    /// Defaults to a style-appropriate value if omitted.
    #[arg(long, value_name = "COUNT")]
    count: Option<usize>,

    /// palette
    #[arg(long, value_name = "PALETTE")]
    palette: Option<String>,

    /// output
    #[arg(long, default_value = "wallpaper.png", value_name = "OUTPUT")]
    output: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let yaml_str = match args.palette {
        Some(palette) => {
            let mut yaml_str = String::new();
            File::open(palette)?.read_to_string(&mut yaml_str)?;
            yaml_str
        }
        None => r#"
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
"#
        .to_string(),
    };

    let seed = args.seed.unwrap_or(thread_rng().next_u64());
    let palette = Palette::parse(yaml_str.as_str())?;
    let mut rng = StdRng::seed_from_u64(seed);
    let pixmap = match args.style {
        Style::Dreamcore => render_dreamcore(
            &palette,
            args.width,
            args.height,
            args.count.unwrap_or(40),
            &mut rng,
        ),
        Style::Flow => render_flow(
            &palette,
            args.width,
            args.height,
            args.count.unwrap_or(100),
            &mut rng,
        ),
    };
    pixmap.save_png(args.output.as_str())?;
    println!("wrote {} (seed={})", args.output, seed);
    Ok(())
}
