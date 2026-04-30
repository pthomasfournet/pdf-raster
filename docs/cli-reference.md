# CLI Reference

```
pdf-raster [OPTIONS] <INPUT> <OUTPUT_PREFIX>
```

Renders PDF pages to image files. Drop-in replacement for `pdftoppm` in scripts.

**Arguments:**

- `<INPUT>` — path to the PDF file. Pass `-` to read from stdin.
- `<OUTPUT_PREFIX>` — output filename prefix. The page number and extension are appended automatically (e.g. prefix `out` → `out-1.ppm`, `out-2.ppm`, …).

---

## Page range

| Flag | Short | Default | Description |
|---|---|---|---|
| `--first-page N` | `-f N` | `1` | First page to render (1-based). |
| `--last-page N` | `-l N` | last page | Last page to render (1-based). |
| `--odd` | `-o` | off | Render only odd-numbered pages. |
| `--even` | `-e` | off | Render only even-numbered pages. |
| `--singlefile` | | off | Stop after the first matching page. |

`--odd` and `--even` are mutually exclusive.

If `--first-page` < 1, it is clamped to 1 with a warning. If `--last-page` exceeds the document length, it is clamped to the last page with a warning.

---

## Resolution and scaling

| Flag | Default | Description |
|---|---|---|
| `-r DPI` / `--resolution DPI` | `150` | Render resolution for both axes. |
| `--rx DPI` | same as `-r` | Horizontal resolution only (overrides `-r` for X). |
| `--ry DPI` | same as `-r` | Vertical resolution only (overrides `-r` for Y). |
| `--scale-to N` | — | Scale the longest edge to N pixels (preserves aspect ratio). |
| `--scale-to-x N` | — | Scale width to N pixels. |
| `--scale-to-y N` | — | Scale height to N pixels. |

DPI values must be positive finite numbers; non-positive values are rejected at startup.

---

## Crop

| Flag | Short | Description |
|---|---|---|
| `--crop-x N` | `-x N` | Crop X offset in pixels. |
| `--crop-y N` | `-y N` | Crop Y offset in pixels. |
| `--crop-width N` | `-W N` | Crop width in pixels. |
| `--crop-height N` | `-H N` | Crop height in pixels. |
| `--cropbox` | | Use the PDF CropBox instead of MediaBox as the page boundary. |

---

## Output format

| Flag | Extension | Description |
|---|---|---|
| _(default)_ | `.ppm` | RGB PPM (Netpbm P6). |
| `--gray` | `.pgm` | Grayscale PGM (Netpbm P5, BT.709). Can combine with `--png`. |
| `--mono` | `.pbm` | 1-bit mono PBM (Netpbm P4, 50% threshold). Can combine with `--png`. |
| `--png` | `.png` | PNG. Combine with `--gray` / `--mono` for grayscale/mono PNG. |
| `--jpeg` | `.jpg` | JPEG. |
| `--jpegcmyk` | `.jpg` | JPEG in CMYK colour space. |
| `--tiff` | `.tif` | TIFF. |
| `--jpegopt N` | — | JPEG quality 0–100 (default 75). |

**Extension matrix:**

| Format flag | `--mono` | `--gray` | Neither |
|---|---|---|---|
| _(default / PPM)_ | `.pbm` | `.pgm` | `.ppm` |
| `--png` | `.png` | `.png` | `.png` |
| `--jpeg` | `.jpg` | `.jpg` | `.jpg` |
| `--tiff` | `.tif` | `.tif` | `.tif` |

---

## Rendering

| Flag | Default | Description |
|---|---|---|
| `--aa yes\|no` | `yes` | Enable raster anti-aliasing. |
| `--aaVector yes\|no` | `yes` | Enable vector anti-aliasing. |
| `--hide-annotations` | off | Skip rendering PDF annotations. |
| `--overprint` | off | Enable overprint preview. |
| `--thinlinemode MODE` | `default` | Thin-line rendering: `default`, `solid`, or `shape`. |

---

## Passwords

| Flag | Description |
|---|---|
| `--opw PASSWORD` | Owner password for encrypted PDFs. |
| `--upw PASSWORD` | User password for encrypted PDFs. |

---

## Parallelism and output

| Flag | Default | Description |
|---|---|---|
| `--threads N` | `0` (auto) | Number of render threads. `0` = one thread per logical CPU. |
| `--sep CHAR` | `-` | Separator between prefix and page number (`out-1.ppm`, `out_1.ppm`, etc.). |
| `--forcenum N` | — | Zero-pad page numbers to at least N digits (`001`, `002`, …). |
| `-P` / `--progress` | off | Print per-page progress (page done, elapsed, ETA) to stderr. |

---

## Exit codes

| Code | Meaning |
|---|---|
| `0` | All pages rendered successfully. |
| `1` | One or more pages failed (errors printed to stderr), or invalid arguments. |

---

## Examples

```bash
# Render all pages at 150 DPI (default), output out-1.ppm, out-2.ppm, …
pdf-raster document.pdf out

# Render pages 3–7 at 300 DPI, grayscale PNG
pdf-raster -f 3 -l 7 -r 300 --gray --png document.pdf out

# Render all pages at 300 DPI, 4 threads, with progress
pdf-raster -r 300 --threads 4 -P document.pdf out

# Render only odd pages
pdf-raster --odd document.pdf out

# Render first page only (--singlefile stops after first match)
pdf-raster --singlefile document.pdf cover

# Render at 150 DPI, zero-pad page numbers to 3 digits: out-001.ppm
pdf-raster -r 150 --forcenum 3 document.pdf out

# Render with underscore separator: out_1.ppm
pdf-raster --sep _ document.pdf out

# Compare output against pdftoppm (pixel-diff test)
tests/compare/compare.sh -r 150 -f 1 -l 5 document.pdf
```

---

## Pixel-diff comparison

`tests/compare/compare.sh` compares pdf-raster output against pdftoppm page-by-page using ImageMagick RMSE.

**Requirements:** `pdftoppm`, `pdf-raster` (release build in `$PATH` or `target/release/`), ImageMagick (`compare`), `bc`.

```bash
tests/compare/compare.sh [OPTIONS] <PDF>

  -r DPI          Render resolution (default: 150)
  -f PAGE         First page (default: 1)
  -l PAGE         Last page (default: all)
  -t THRESHOLD    Max RMSE per page, 0–255 scale (default: 2.0)
  -v              Verbose: print all pages, not just failures
  -o DIR          Write diff images to DIR
  -n              Dry run: print commands without executing
```

Exit code 0 if all pages are within threshold; 1 if any fail.

```bash
# Example output
── Summary ──────────────────────────────────────────────────────────────
  PDF:       document.pdf
  DPI:       150
  Pages:     5 total, 5 passed, 0 failed
  Avg RMSE:  0.8432 / 255
  Max RMSE:  1.2100 / 255
  Threshold: 2.0 / 255
─────────────────────────────────────────────────────────────────────────
PASSED
```
