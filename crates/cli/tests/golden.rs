//! Golden-image regression tests.
// Infrastructure is complete; CASES is empty until fixture PDFs are added.
// Suppress dead_code warnings on all helpers until cases are registered.
#![allow(dead_code)]
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
//! renders the same fixtures with the release `pdf-raster` binary and writes
//! the results into `tests/golden/ref/`.  Commit the updated PPMs.
//!
//! # Environment
//!
//! The binary path is resolved at compile time via `CARGO_BIN_EXE_pdf-raster`.
//! During `cargo test` this is the *debug* build.  References are generated
//! with the release build; minor rendering differences between build profiles
//! are not expected, but regenerate references with debug if tests fail
//! consistently only in CI.
//!
//! The workspace root is derived from `CARGO_MANIFEST_DIR` (this crate's
//! manifest lives at `crates/cli/`).

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

// ── constants ─────────────────────────────────────────────────────────────────

/// Maximum allowed mean absolute error per channel (0-255 scale).
///
/// Rendering differences arise from sub-pixel rounding and minor divergences
/// in `FreeType` / anti-aliasing behaviour vs. the C++ reference.  A value of
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
    /// the 1-based page number, zero-padded to match `pdf-raster`'s convention
    /// for the document's total page count.
    ref_prefix: &'static str,
    /// First page to test, 1-based.
    first: u32,
    /// Last page to test, 1-based.
    last: u32,
    /// Total page count of the PDF — determines the zero-pad width.
    ///
    /// Must match the actual page count of the fixture so that the file names
    /// produced by `pdf-raster` align with the reference files.
    total_pages: u32,
}

/// All registered golden test cases.
///
/// Keep these small — a few pages each — so `cargo test` completes quickly.
/// Large-page-count fixtures belong in the benchmark suite, not here.
///
/// **Ordering matters**: the test entry points reference cases by index.
/// Add new cases at the end and add a corresponding `#[test]` fn.
///
/// # Adding fixtures
///
/// Place PDF files in `tests/fixtures/`, add a `Case` entry here and a
/// matching entry in `tests/golden/generate.sh`, then run:
///
/// ```bash
/// cargo build --release -p pdf-raster
/// bash tests/golden/generate.sh
/// git add tests/golden/ref/
/// ```
///
/// Fixture PDFs are gitignored — each contributor provides their own.
/// The reference PPMs in `tests/golden/ref/` are committed once generated.
const CASES: &[Case] = &[
    // Add cases here. Example:
    //
    // Case {
    //     pdf: "my-document.pdf",
    //     ref_prefix: "my-document-72",
    //     first: 1,
    //     last: 3,
    //     total_pages: 10,
    // },
];

// ── path helpers ──────────────────────────────────────────────────────────────

fn workspace_root() -> PathBuf {
    // CARGO_MANIFEST_DIR is `<workspace>/crates/cli`; walk up two levels.
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let crates = manifest
        .parent()
        .unwrap_or_else(|| panic!("expected crates/ parent of {}", manifest.display()));
    crates
        .parent()
        .unwrap_or_else(|| panic!("expected workspace root parent of {}", crates.display()))
        .to_owned()
}

fn fixtures_dir() -> PathBuf {
    workspace_root().join("tests/fixtures")
}

fn ref_dir() -> PathBuf {
    workspace_root().join("tests/golden/ref")
}

// ── PPM parsing ───────────────────────────────────────────────────────────────

/// Skip ASCII whitespace and `#`-comment lines, advancing `pos` in `buf`.
fn skip_whitespace_and_comments(buf: &[u8], pos: &mut usize) {
    loop {
        while *pos < buf.len() && matches!(buf[*pos], b' ' | b'\t' | b'\n' | b'\r') {
            *pos += 1;
        }
        if *pos < buf.len() && buf[*pos] == b'#' {
            while *pos < buf.len() && buf[*pos] != b'\n' {
                *pos += 1;
            }
        } else {
            break;
        }
    }
}

/// Read the next whitespace-delimited token from `buf`, advancing `pos`.
fn read_token(buf: &[u8], pos: &mut usize) -> String {
    skip_whitespace_and_comments(buf, pos);
    let start = *pos;
    while *pos < buf.len() && !matches!(buf[*pos], b' ' | b'\t' | b'\n' | b'\r') {
        *pos += 1;
    }
    String::from_utf8_lossy(&buf[start..*pos]).into_owned()
}

/// Parse a binary P6 PPM file.  Returns `(width, height, pixels_rgb)`.
///
/// Panics with a clear message on any parse failure so test output is readable.
fn parse_ppm(path: &Path) -> (u32, u32, Vec<u8>) {
    let data =
        std::fs::read(path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));

    let mut pos = 0usize;

    // Magic — must be "P6".
    assert!(
        data.get(pos..pos + 2) == Some(b"P6"),
        "not a P6 PPM file: {} (got {:?})",
        path.display(),
        data.get(..4)
            .map(|b| String::from_utf8_lossy(b).into_owned())
    );
    pos += 2;

    let w: u32 = read_token(&data, &mut pos)
        .parse()
        .unwrap_or_else(|e| panic!("bad width in {}: {e}", path.display()));
    let h: u32 = read_token(&data, &mut pos)
        .parse()
        .unwrap_or_else(|e| panic!("bad height in {}: {e}", path.display()));
    let maxval: u32 = read_token(&data, &mut pos)
        .parse()
        .unwrap_or_else(|e| panic!("bad maxval in {}: {e}", path.display()));

    assert!(
        maxval == 255,
        "only 8-bit PPMs supported (got maxval={maxval}) in {}",
        path.display()
    );

    // PPM spec: exactly one whitespace character separates the header from the
    // pixel data.  We must consume exactly one byte here regardless of whether
    // it is '\n', '\r\n', etc., so that CRLF files do not corrupt pixel reads.
    assert!(
        pos < data.len(),
        "PPM file truncated before pixel data: {}",
        path.display()
    );
    pos += 1; // consume exactly the one mandatory separator byte

    let n_pixels = w as usize * h as usize; // safe: usize arithmetic, no u32 overflow
    let expected = n_pixels * 3;
    let remaining = data.len().saturating_sub(pos);
    assert_eq!(
        remaining,
        expected,
        "pixel data length mismatch in {}: expected {expected} bytes ({w}×{h}×3), got {remaining}",
        path.display()
    );

    assert!(
        n_pixels > 0,
        "zero-size image in {}: {w}×{h}",
        path.display()
    );

    (w, h, data[pos..].to_vec())
}

/// Zero-pad `n` to `width` decimal digits.
fn zero_pad(n: u32, width: usize) -> String {
    format!("{n:0>width$}")
}

// ── core test logic ───────────────────────────────────────────────────────────

/// Compare one rendered page against its reference; return `Some(msg)` on
/// failure or `None` on pass.  Tolerates ±1 px dimension drift from
/// independent rounding paths.
fn compare_page(page: u32, case: &Case, ref_file: &Path, out_file: &Path) -> Option<String> {
    assert!(
        ref_file.exists(),
        "reference file missing: {}\n  Run: bash tests/golden/generate.sh",
        ref_file.display()
    );
    assert!(
        out_file.exists(),
        "pdf-raster did not produce page {page} for {} (expected {})",
        case.pdf,
        out_file.display()
    );

    let (ref_w, ref_h, ref_px) = parse_ppm(ref_file);
    let (out_w, out_h, out_px) = parse_ppm(out_file);

    if ref_w.abs_diff(out_w) > 1 || ref_h.abs_diff(out_h) > 1 {
        return Some(format!(
            "  page {page}: dimension mismatch — ref {ref_w}×{ref_h}, got {out_w}×{out_h}"
        ));
    }

    let cmp_w = ref_w.min(out_w) as usize;
    let cmp_h = ref_h.min(out_h) as usize;
    // parse_ppm asserts n_pixels > 0, and the dimension check above gates
    // cmp_{w,h} ≥ ref_{w,h} − 1, so the comparison region is non-empty.
    debug_assert!(cmp_w > 0 && cmp_h > 0, "comparison region is empty");

    let ref_stride = ref_w as usize * 3;
    let out_stride = out_w as usize * 3;

    let mut sum_diff: u64 = 0;
    let mut n_samples: u64 = 0;
    for row in 0..cmp_h {
        let r = &ref_px[row * ref_stride..row * ref_stride + cmp_w * 3];
        let o = &out_px[row * out_stride..row * out_stride + cmp_w * 3];
        for (&a, &b) in r.iter().zip(o.iter()) {
            sum_diff += u64::from(a.abs_diff(b));
            n_samples += 1;
        }
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "sum_diff ≤ 255 × pixel-count and n_samples = pixel-count; \
                  both are bounded by image area ≪ 2^52, so the f64 cast \
                  is lossless for any plausible test fixture"
    )]
    let page_mae = sum_diff as f64 / n_samples as f64;

    (page_mae > MAE_LIMIT).then(|| {
        format!(
            "  page {page}: MAE {page_mae:.4} > limit {MAE_LIMIT:.1}  \
             (ref {ref_w}×{ref_h}, got {out_w}×{out_h})"
        )
    })
}

fn run_case(case: &Case) {
    let binary = env!("CARGO_BIN_EXE_pdf-raster");
    let pdf_path = fixtures_dir().join(case.pdf);
    let ref_dir = ref_dir();

    assert!(
        pdf_path.exists(),
        "fixture PDF missing: {}\n  Add it to tests/fixtures/ and re-run generate.sh.",
        pdf_path.display()
    );

    // Pad width matches pdf-raster's digit_width(total_pages) in naming.rs.
    let pad_width = case.total_pages.to_string().len();

    // Render into a temp directory co-located with the fixtures so that both
    // live on the same filesystem (avoids cross-device rename issues and
    // ensures the temp dir is not on a size-limited tmpfs mount).
    let tmp = tempfile::tempdir_in(fixtures_dir())
        .expect("failed to create tempdir alongside test fixtures");
    let out_prefix = tmp.path().join("page");

    let dpi_str = DPI.to_string();
    let first_str = case.first.to_string();
    let last_str = case.last.to_string();

    let output = Command::new(binary)
        .args([
            "-r",
            &dpi_str,
            "-f",
            &first_str,
            "-l",
            &last_str,
            pdf_path.to_str().expect("pdf path must be valid UTF-8"),
            out_prefix
                .to_str()
                .expect("tempdir path must be valid UTF-8"),
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output()
        .unwrap_or_else(|e| panic!("failed to spawn pdf-raster ({binary}): {e}"));

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!(
            "pdf-raster exited with {} for {}:\n{stderr}",
            output.status, case.pdf
        );
    }

    let mut page_failures: Vec<String> = Vec::new();

    for page in case.first..=case.last {
        let page_str = zero_pad(page, pad_width);
        let out_file = tmp.path().join(format!("page-{page_str}.ppm"));
        let ref_file = ref_dir.join(format!("{}-{page_str}.ppm", case.ref_prefix));
        if let Some(failure) = compare_page(page, case, &ref_file, &out_file) {
            page_failures.push(failure);
        }
    }

    assert!(
        page_failures.is_empty(),
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

// ── test entry points ─────────────────────────────────────────────────────────
//
// Add a #[test] fn for each Case in CASES, referencing it by index.
// Example:
//
// #[test]
// fn golden_my_document() {
//     run_case(&CASES[0]);
// }
