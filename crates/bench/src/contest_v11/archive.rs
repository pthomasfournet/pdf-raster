//! Synthetic archive builder.
//!
//! Concatenates real fixture PDFs into one document, rebuilding the page
//! tree and xref via qpdf.  We use qpdf as a one-shot external tool because
//! rebuilding xref by hand is hundreds of lines of code and qpdf has 20
//! years of corner-case handling we'd otherwise re-invent.
//!
//! Repeats fixtures cyclically until the file size lands at or above the
//! requested target.  Writes via `--pages base.pdf 1-z f1.pdf 1-z f2.pdf 1-z ...`.

use std::path::{Path, PathBuf};
use std::process::Command;

const FIXTURE_NAMES: &[&str] = &[
    "corpus-04-ebook-mixed.pdf",
    "corpus-05-academic-book.pdf",
    "corpus-08-scan-dct-1927.pdf",
    "corpus-09-scan-dct-1836.pdf",
];

/// Sanity bound on iteration count.  At ~150 MB/cycle we'd hit 10 GB in well
/// under 100 cycles; 200 is plenty of headroom while still bounding a runaway.
const MAX_CYCLES: u32 = 200;

/// Build a synthetic PDF at `out` by concatenating corpus fixtures via qpdf.
///
/// `target_bytes` is the **cumulative size of the input fixture files**
/// consumed before the loop stops, not the output file size.  qpdf
/// deduplicates shared PDF objects across pages, so the output is
/// typically 2–3× smaller than the input byte sum.  To produce a
/// roughly 10 GiB output file, pass ~25–30 GiB as `target_bytes`.
///
/// Returns an error if qpdf is missing, fixtures are missing, or qpdf
/// exits non-zero.
#[expect(
    clippy::cast_precision_loss,
    reason = "byte counts are formatted for human display only"
)]
pub fn build(out: &Path, target_bytes: u64) -> Result<(), String> {
    if Command::new("qpdf").arg("--version").output().is_err() {
        return Err("qpdf not found on PATH; install with `apt install qpdf` and retry".into());
    }

    let fixture_dir = locate_fixture_dir()?;
    if !fixture_dir.is_dir() {
        return Err(format!(
            "fixture directory does not exist: {}",
            fixture_dir.display()
        ));
    }

    let fixtures: Vec<PathBuf> = FIXTURE_NAMES
        .iter()
        .map(|name| fixture_dir.join(name))
        .collect();
    for f in &fixtures {
        if !f.is_file() {
            return Err(format!("fixture missing: {}", f.display()));
        }
    }

    // Build the per-page argument list: each fixture path followed by "1-z".
    // The first fixture is the qpdf "input"; subsequent ones land in the
    // --pages spec.
    let mut total_bytes = 0u64;
    let mut pages_spec: Vec<String> = Vec::new();
    let mut iteration = 0u32;
    'outer: while total_bytes < target_bytes {
        for f in &fixtures {
            let sz = std::fs::metadata(f)
                .map_err(|e| format!("metadata on {}: {e}", f.display()))?
                .len();
            pages_spec.push(f.to_string_lossy().into_owned());
            pages_spec.push("1-z".into());
            total_bytes = total_bytes.saturating_add(sz);
            if total_bytes >= target_bytes {
                break 'outer;
            }
        }
        iteration += 1;
        if iteration > MAX_CYCLES {
            return Err(format!(
                "archive builder exceeded {MAX_CYCLES} cycles before reaching \
                 target {target_bytes} bytes (current cumulative size: {total_bytes})",
            ));
        }
    }

    // qpdf needs the base + at least one extra (path, "1-z") pair under
    // --pages to produce a meaningful concatenation.  After the two
    // base-related removes below, `pages_spec` becomes the --pages body —
    // it must contain at least one extra pair (≥ 2 entries) for qpdf
    // not to reject the invocation.  So pre-remove length ≥ 4.
    if pages_spec.len() < 4 {
        return Err(format!(
            "target_bytes {target_bytes} too small to fill an archive: \
             only {} fixture entries enqueued, need at least 4 \
             (one base + one extra page-spec pair)",
            pages_spec.len(),
        ));
    }

    // qpdf invocation: `qpdf <base> --pages <p1> 1-z <p2> 1-z ... -- <out>`.
    // The first fixture path acts as the base; we skip its first "1-z" pair
    // in the --pages spec since qpdf takes the base from the positional arg.
    let base = pages_spec.remove(0);
    // "1-z" for the base PDF; qpdf doesn't repeat it inside --pages.
    let _first_range = pages_spec.remove(0);

    eprintln!(
        "qpdf concatenating {} fixture-instances → {} (target {:.1} GB)",
        pages_spec.len() / 2,
        out.display(),
        target_bytes as f64 / (1u64 << 30) as f64,
    );

    let mut cmd = Command::new("qpdf");
    cmd.arg(&base).arg("--pages");
    cmd.args(&pages_spec);
    cmd.arg("--").arg(out);

    let status = cmd
        .status()
        .map_err(|e| format!("qpdf invocation failed: {e}"))?;
    if !status.success() {
        return Err(format!("qpdf exited with {status}"));
    }
    let final_size = std::fs::metadata(out)
        .map_err(|e| format!("metadata on output: {e}"))?
        .len();
    eprintln!(
        "archive built: {} bytes ({:.2} GB)",
        final_size,
        final_size as f64 / (1u64 << 30) as f64
    );
    Ok(())
}

/// Find the workspace's tests/fixtures directory.
///
/// `CARGO_MANIFEST_DIR` is the bench crate's Cargo.toml directory; the
/// fixtures live two levels up (workspace root) at `tests/fixtures/`.
fn locate_fixture_dir() -> Result<PathBuf, String> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let path = PathBuf::from(manifest_dir)
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures");
    path.canonicalize()
        .map_err(|e| format!("cannot resolve fixture dir at {}: {e}", path.display()))
}
