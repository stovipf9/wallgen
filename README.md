# wallgen

A palette-driven generative wallpaper baker. Give it a [base16](https://github.com/chriskempson/base16) color palette and a seed, and it bakes a static PNG wallpaper — no live rendering, no GPU cost at runtime, just a one-shot image you can hand to any wallpaper daemon.

Two styles are available:

- **`flow`** — curl-noise dye advection: streamlines traced through a divergence-free vector field derived from domain-warped fractal (fbm) noise, drawn as tapered, additively-blended filaments.
- **`dreamcore`** — small, semantically-loaded but non-resolving fragments (broken glyphs, mismatched eyes, pseudo-icons, broken seven-segment digits) scattered sparsely across a flat, hard-edged background.

Both styles read the same 16-color palette, so a whole rotating set stays visually coherent regardless of style.

## Usage

```sh
wallgen --style dreamcore --palette colors.yaml --seed 42 --output wallpaper.png
```

```
Options:
      --seed <SEED>        seed
      --width <WIDTH>      width [default: 1920]
      --height <HEIGHT>    height [default: 1080]
      --style <STYLE>      style [default: dreamcore] [possible values: dreamcore, flow]
      --count <COUNT>      count: dreamcore fragments or flow streamlines, depending on
                            `style`. Defaults to a style-appropriate value if omitted
      --palette <PALETTE>  palette
      --output <OUTPUT>    output [default: wallpaper.png]
```

If `--seed` is omitted, one is chosen at random and printed on completion, so a specific result can always be reproduced later. If `--palette` is omitted, a built-in [base16 OneDark](https://github.com/chriskempson/base16-onedark-scheme) palette is used.

## Palette format

A YAML file with a `colors:` map of the 16 standard base16 slots, values as bare hex (no `#`):

```yaml
colors:
  base00: "282c34"  # default background
  base01: "353b45"
  base02: "3e4451"
  base03: "545862"
  base04: "565c64"
  base05: "abb2bf"  # default fg
  base06: "b6bdca"
  base07: "c8ccd4"
  base08: "e06c75"  # accent colors from here on (base08-base0f)
  base09: "d19a66"
  base0A: "e5c07b"
  base0B: "98c379"
  base0C: "56b6c2"
  base0D: "61afef"
  base0E: "c678dd"
  base0F: "be5046"
```

Keys are case-insensitive on the hex digit (`base0A` and `base0a` both work). Any base16-compatible palette works out of the box — pywal, base16-shell themes, etc. all produce this shape.

## Building

```sh
cargo build --release
```

The binary is at `target/release/wallgen`. No runtime dependencies beyond what Cargo pulls in.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option.
