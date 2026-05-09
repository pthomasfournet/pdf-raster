//! Synthetic archive builder.
//!
//! Concatenates real fixture PDFs into one document, rebuilding the page
//! tree and xref via qpdf.  We use qpdf as a one-shot external tool because
//! rebuilding xref by hand is hundreds of lines of code and qpdf has 20
//! years of corner-case handling we'd otherwise re-invent.
//!
//! qpdf deduplicates byte-identical objects across an `--pages` invocation,
//! so naively repeating the same fixture N times collapses to roughly the
//! size of one fixture (~76× shrink observed on a 30-cycle test).  To
//! defeat that, each cycle iteration writes the fixtures out via a
//! single-input qpdf rewrite first; the rewrite renumbers every object
//! fresh, so the final concat sees N distinct files with no shared
//! object IDs and no dedup is possible.
//!
//! `target_output_bytes` is therefore the *output* size budget, not an
//! input-byte sum — we keep adding rewritten cycles until the running
//! sum of cycle-output sizes meets the target.

use std::path::{Path, PathBuf};
use std::process::Command;

const FIXTURE_NAMES: &[&str] = &[
    "corpus-04-ebook-mixed.pdf",
    "corpus-05-academic-book.pdf",
    "corpus-08-scan-dct-1927.pdf",
    "corpus-09-scan-dct-1836.pdf",
];

/// Sanity bound on cycle count — guards against pathological tiny-target
/// cases that would loop forever.  At ~510 MB output per cycle (4 distinct
/// DCT-heavy fixtures) we'd hit a 10 GB target in ~20 cycles; 50 is plenty.
const MAX_CYCLES: u32 = 50;

/// Build a synthetic PDF archive at `out` with output size approximately
/// `target_output_bytes`.
///
/// Uses qpdf to first rewrite each fixture into a uniquely-numbered copy
/// per cycle (defeating qpdf's cross-input dedup), then concatenates all
/// the per-cycle copies into the final archive.  Output size tracks
/// input cumulative size to within a few percent.
///
/// Returns an error if qpdf is missing, fixtures are missing, or qpdf
/// exits non-zero on any sub-step.
#[expect(
    clippy::cast_precision_loss,
    reason = "byte counts are formatted for human display only"
)]
#[expect(
    clippy::too_many_lines,
    reason = "qpdf orchestration is sequential by nature; splitting just to pass the line count would add friction"
)]
pub fn build(out: &Path, target_output_bytes: u64) -> Result<(), String> {
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

    let cycle_size: u64 = fixtures
        .iter()
        .map(|f| std::fs::metadata(f).map_or(0, |m| m.len()))
        .sum();
    if cycle_size == 0 {
        return Err("fixture sizes sum to zero — cannot build archive".into());
    }
    let cycles_needed = u32::try_from(target_output_bytes.div_ceil(cycle_size).max(1))
        .map_err(|_| format!("target {target_output_bytes} bytes requires unbounded cycles"))?;
    if cycles_needed > MAX_CYCLES {
        return Err(format!(
            "target {target_output_bytes} bytes needs {cycles_needed} cycles \
             (cycle is ~{cycle_size} bytes); MAX_CYCLES={MAX_CYCLES}",
        ));
    }

    // Stage all rewritten copies in a sibling directory of `out`.  Cleaned
    // up at end; survives panics via the Drop on `_staging`.
    let staging_root = out
        .parent()
        .ok_or_else(|| format!("output path has no parent: {}", out.display()))?
        .join(".phase11_archive_staging");
    let _ = std::fs::remove_dir_all(&staging_root);
    std::fs::create_dir_all(&staging_root)
        .map_err(|e| format!("create staging dir {}: {e}", staging_root.display()))?;
    let _staging = StagingDir(staging_root.clone());

    eprintln!(
        "archive builder: target {:.2} GB output, {} cycles × {} fixtures each \
         (cycle output ~{:.2} GB)",
        target_output_bytes as f64 / (1u64 << 30) as f64,
        cycles_needed,
        fixtures.len(),
        cycle_size as f64 / (1u64 << 30) as f64,
    );

    // Step 1: per-cycle rewrites.  Each cycle produces N distinct copies
    // of the fixtures with renumbered object IDs.
    let mut concat_inputs: Vec<PathBuf> =
        Vec::with_capacity((cycles_needed as usize).saturating_mul(fixtures.len()));
    for cycle in 0..cycles_needed {
        for (i, src) in fixtures.iter().enumerate() {
            let dst = staging_root.join(format!("c{i}_cycle{cycle}.pdf"));
            let status = Command::new("qpdf")
                .arg(src)
                .arg("--pages")
                .arg(src)
                .arg("1-z")
                .arg("--")
                .arg(&dst)
                .status()
                .map_err(|e| format!("qpdf rewrite spawn: {e}"))?;
            if !status.success() {
                return Err(format!(
                    "qpdf rewrite of {} failed: {status}",
                    src.display()
                ));
            }
            concat_inputs.push(dst);
        }
        if cycle % 5 == 0 || cycle + 1 == cycles_needed {
            let staged_so_far: u64 = concat_inputs
                .iter()
                .map(|p| std::fs::metadata(p).map_or(0, |m| m.len()))
                .sum();
            eprintln!(
                "  rewrote cycle {}/{} ({} files, {:.2} GB staged)",
                cycle + 1,
                cycles_needed,
                concat_inputs.len(),
                staged_so_far as f64 / (1u64 << 30) as f64,
            );
        }
    }

    // Step 2: final concat.  qpdf args follow the shape:
    //   qpdf <base> --pages <p1> 1-z <p2> 1-z ... -- <out>
    // The base is the first staged file; the rest go into --pages with 1-z.
    let mut concat_iter = concat_inputs.iter();
    let base = concat_iter
        .next()
        .ok_or("no inputs staged for concat")?
        .clone();
    let pages_args: Vec<String> = concat_iter
        .flat_map(|p| [p.to_string_lossy().into_owned(), "1-z".to_string()])
        .collect();
    if pages_args.is_empty() {
        return Err("internal: only one cycle's worth of inputs staged".into());
    }

    eprintln!(
        "qpdf concatenating {} staged files → {}",
        concat_inputs.len(),
        out.display(),
    );

    let mut cmd = Command::new("qpdf");
    cmd.arg(&base).arg("--pages");
    cmd.args(&pages_args);
    cmd.arg("--").arg(out);

    let status = cmd
        .status()
        .map_err(|e| format!("qpdf concat spawn: {e}"))?;
    if !status.success() {
        return Err(format!("qpdf concat exited with {status}"));
    }

    let final_size = std::fs::metadata(out)
        .map_err(|e| format!("metadata on output: {e}"))?
        .len();
    eprintln!(
        "archive built: {} bytes ({:.2} GB)",
        final_size,
        final_size as f64 / (1u64 << 30) as f64,
    );
    Ok(())
}

/// RAII wrapper that wipes the staging directory on drop.  Keeps the
/// directory through a normal return; cleans on panic too.
struct StagingDir(PathBuf);

impl Drop for StagingDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
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
