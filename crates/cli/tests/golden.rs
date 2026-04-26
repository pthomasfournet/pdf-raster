//! Golden-image regression tests.
//!
//! For each entry in [`CASES`] the test:
//!   1. Runs `pdf-raster` on the fixture PDF.
//!   2. Parses every rendered PPM as a raw RGB byte slice.
//!   3. Loads the matching reference PPM from `tests/golden/ref/`.
//!   4. Asserts that the mean absolute error (MAE) per channel is ≤ [`MAE_LIMIT`].
//!
//! # Regenerating references
//!
//! Run `tests/golden/generate.sh` from the workspace root.  That script
//! renders the same fixtures with `pdftoppm` and writes the results into
//! `tests/golden/ref/`.  Commit the updated PPMs.
//!
//! # Environment
//!
//! The binary path is resolved at compile time via `CARGO_BIN_EXE_pdf-raster`.
//! The workspace root is derived from `CARGO_MANIFEST_DIR` (this crate's
//! manifest lives at `crates/cli/`).

use std::path::{Path, PathBuf};
use std::process::Command;

// ── constants ─────────────────────────────────────────────────────────────────

/// Maximum allowed mean absolute error per channel (0-255 scale).
///
/// Rendering differences arise from sub-pixel rounding and minor divergences
/// in FreeType / anti-aliasing behaviour vs. the C++ reference.  A value of
/// 4.0 is tight enough to catch real regressions while tolerating unavoidable
/// implementation deltas.
const MAE_LIMIT: f64 = 4.0;

/// DPI used for all golden renders — low enough for fast tests, high enough
/// to exercise real rendering paths.
const DPI: u32 = 72;

// ── test case table ───────────────────────────────────────────────────────────

/// A single golden-test case.
struct Case {
    /// Basename of the PDF in `tests/fixtures/`.
    pdf: &'static str,
    /// Prefix used when naming reference files in `tests/golden/ref/`.
    ///
    /// Reference files are named `<prefix>-<page>.ppm` where `<page>` is
    /// the 1-based page number, zero-padded to match pdftoppm's convention
    /// for the document's total page count.
    ref_prefix: &'static str,
    /// First page to test, 1-based.
    first: u32,
    /// Last page to test, 1-based.
    last: u32,
    /// Total page count of the PDF — determines pdftoppm's zero-pad width.
    total_pages: u32,
}

/// All registered golden test cases.
///
/// Keep these small — a few pages each — so `cargo test` completes quickly.
/// Large-page-count fixtures belong in the benchmark suite, not here.
const CASES: &[Case] = &[
    Case {
        pdf: "cryptic-rite.pdf",
        ref_prefix: "cryptic-rite-72",
        first: 1,
        last: 3,
        total_pages: 7,   // 7 pages → pdftoppm uses 1-digit padding
    },
    Case {
        pdf: "ritual-14th.pdf",
        ref_prefix: "ritual-14th-72",
        first: 1,
        last: 3,
        total_pages: 41,  // 41 pages → pdftoppm uses 2-digit padding
    },
];

// ── helpers ───────────────────────────────────────────────────────────────────

fn workspace_root() -> PathBuf {
    // CARGO_MANIFEST_DIR is `<workspace>/crates/cli`; walk up two levels.
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .expect("crates/ dir")
        .parent()
        .expect("workspace root")
        .to_owned()
}

fn fixtures_dir() -> PathBuf {
    workspace_root().join("tests/fixtures")
}

fn ref_dir() -> PathBuf {
    workspace_root().join("tests/golden/ref")
}

/// Parse a binary P6 PPM file.  Returns `(width, height, pixels_rgb)`.
///
/// Panics with a clear message on any parse failure so test output is readable.
fn parse_ppm(path: &Path) -> (u32, u32, Vec<u8>) {
    let data = std::fs::read(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {}", path.display(), e));

    // PPM header: "P6\n<w> <h>\n255\n" then raw RGB bytes.
    // Skip comment lines starting with '#'.
    let mut pos = 0usize;

    let consume_whitespace_and_comments = |p: &mut usize, buf: &[u8]| {
        loop {
            while *p < buf.len() && (buf[*p] == b' ' || buf[*p] == b'\t' || buf[*p] == b'\n' || buf[*p] == b'\r') {
                *p += 1;
            }
            if *p < buf.len() && buf[*p] == b'#' {
                while *p < buf.len() && buf[*p] != b'\n' {
                    *p += 1;
                }
            } else {
                break;
            }
        }
    };

    // Magic
    assert!(
        data.get(pos..pos + 2) == Some(b"P6"),
        "not a P6 PPM file: {}",
        path.display()
    );
    pos += 2;

    let read_token = |p: &mut usize, buf: &[u8]| -> String {
        consume_whitespace_and_comments(p, buf);
        let start = *p;
        while *p < buf.len() && !matches!(buf[*p], b' ' | b'\t' | b'\n' | b'\r') {
            *p += 1;
        }
        String::from_utf8_lossy(&buf[start..*p]).into_owned()
    };

    let w: u32 = read_token(&mut pos, &data)
        .parse()
        .unwrap_or_else(|_| panic!("bad width in {}", path.display()));
    let h: u32 = read_token(&mut pos, &data)
        .parse()
        .unwrap_or_else(|_| panic!("bad height in {}", path.display()));
    let maxval: u32 = read_token(&mut pos, &data)
        .parse()
        .unwrap_or_else(|_| panic!("bad maxval in {}", path.display()));

    assert!(
        maxval == 255,
        "only 8-bit PPMs supported (got maxval={maxval}) in {}",
        path.display()
    );

    // One whitespace byte separates header from pixel data.
    pos += 1;

    let expected = (w * h * 3) as usize;
    let pixels = data[pos..].to_vec();
    assert_eq!(
        pixels.len(),
        expected,
        "pixel data length mismatch in {}: expected {expected}, got {}",
        path.display(),
        pixels.len()
    );

    (w, h, pixels)
}

/// Zero-pad a number to `width` decimal digits.
fn zero_pad(n: u32, width: usize) -> String {
    format!("{n:0>width$}")
}

// ── core test logic ───────────────────────────────────────────────────────────

fn run_case(case: &Case) {
    let binary = env!("CARGO_BIN_EXE_pdf-raster");
    let pdf_path = fixtures_dir().join(case.pdf);
    let ref_dir = ref_dir();

    // pdftoppm zero-pads page numbers to the width needed for total_pages.
    // e.g. 7-page doc → 1 digit; 41-page doc → 2 digits; 100-page doc → 3 digits.
    let pad_width = case.total_pages.to_string().len();

    // Render into a temp directory.
    let tmp = tempfile::tempdir().expect("tempdir");
    let out_prefix = tmp.path().join("page");

    let status = Command::new(binary)
        .args([
            "-r",
            &DPI.to_string(),
            "-f",
            &case.first.to_string(),
            "-l",
            &case.last.to_string(),
            pdf_path.to_str().expect("pdf path is valid UTF-8"),
            out_prefix.to_str().expect("tmp path is valid UTF-8"),
        ])
        .status()
        .unwrap_or_else(|e| panic!("failed to spawn pdf-raster: {e}"));

    assert!(
        status.success(),
        "pdf-raster exited with {} for {}",
        status,
        case.pdf
    );

    let mut page_failures: Vec<String> = Vec::new();

    for page in case.first..=case.last {
        let page_str = zero_pad(page, pad_width);

        let out_file = tmp.path().join(format!("page-{page_str}.ppm"));
        let ref_file = ref_dir.join(format!("{}-{}.ppm", case.ref_prefix, page_str));

        assert!(
            ref_file.exists(),
            "reference file missing: {}\n  Run tests/golden/generate.sh to create it.",
            ref_file.display()
        );
        assert!(
            out_file.exists(),
            "pdf-raster did not produce page {page} for {}: expected {}",
            case.pdf,
            out_file.display()
        );

        let (ref_w, ref_h, ref_px) = parse_ppm(&ref_file);
        let (out_w, out_h, out_px) = parse_ppm(&out_file);

        // Allow ±1px dimension difference from independent rounding.
        let w_ok = ref_w.abs_diff(out_w) <= 1;
        let h_ok = ref_h.abs_diff(out_h) <= 1;
        if !w_ok || !h_ok {
            page_failures.push(format!(
                "  page {page}: dimension mismatch — ref {ref_w}×{ref_h}, got {out_w}×{out_h}"
            ));
            continue;
        }

        // Compare over the overlapping region to handle the ±1px cases.
        let cmp_w = ref_w.min(out_w) as usize;
        let cmp_h = ref_h.min(out_h) as usize;
        let ref_stride = ref_w as usize * 3;
        let out_stride = out_w as usize * 3;

        let mut sum_diff: u64 = 0;
        let mut n_samples: u64 = 0;
        for row in 0..cmp_h {
            let r = &ref_px[row * ref_stride..row * ref_stride + cmp_w * 3];
            let o = &out_px[row * out_stride..row * out_stride + cmp_w * 3];
            for (&a, &b) in r.iter().zip(o.iter()) {
                sum_diff += a.abs_diff(b) as u64;
                n_samples += 1;
            }
        }
        let page_mae = sum_diff as f64 / n_samples as f64;

        if page_mae > MAE_LIMIT {
            page_failures.push(format!(
                "  page {page}: MAE {page_mae:.4} > limit {MAE_LIMIT:.1}  \
                 (ref {ref_w}×{ref_h}, got {out_w}×{out_h})"
            ));
        }
    }

    if !page_failures.is_empty() {
        panic!(
            "golden test FAILED for {} @{DPI}dpi:\n{}\n\n\
             To inspect diffs: bash tests/compare/compare.sh -r {DPI} \
             -f {} -l {} tests/fixtures/{}",
            case.pdf,
            page_failures.join("\n"),
            case.first,
            case.last,
            case.pdf,
        );
    }
}

// ── test entry points ─────────────────────────────────────────────────────────

#[test]
fn golden_cryptic_rite() {
    run_case(&CASES[0]);
}

#[test]
fn golden_ritual_14th() {
    run_case(&CASES[1]);
}
