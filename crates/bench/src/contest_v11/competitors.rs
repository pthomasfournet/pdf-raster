//! mutool / pdftoppm subprocess wrappers.
//!
//! Each function returns `Some(elapsed_ms)` if the competitor is installed
//! and runs successfully, or `None` if the competitor is missing or fails.
//! No subprocess panic propagates — bench harnesses always run to completion
//! and report skipped competitors at the end.

use std::path::Path;
use std::process::Command;
use std::time::Instant;

#[derive(Debug)]
pub struct CompetitorResult {
    pub name: &'static str,
    pub elapsed_ms: Option<f64>,
}

fn time_command(name: &'static str, mut cmd: Command) -> CompetitorResult {
    let t0 = Instant::now();
    match cmd.output() {
        Ok(out) if out.status.success() => CompetitorResult {
            name,
            elapsed_ms: Some(t0.elapsed().as_secs_f64() * 1e3),
        },
        Ok(_) | Err(_) => CompetitorResult {
            name,
            elapsed_ms: None,
        },
    }
}

/// Render a single page with `mutool draw -r 150 -o <out> <archive> <page>`.
/// Cleans the output file before returning so the on-disk cost is not
/// accidentally measured on subsequent runs.
pub fn mutool_render(archive: &Path, page: u32, out: &Path) -> CompetitorResult {
    if Command::new("mutool").arg("-v").output().is_err() {
        return CompetitorResult {
            name: "mutool",
            elapsed_ms: None,
        };
    }
    let mut cmd = Command::new("mutool");
    cmd.args([
        "draw",
        "-r",
        "150",
        "-o",
        out.to_str().expect("non-utf8 path"),
        archive.to_str().expect("non-utf8 archive path"),
        &page.to_string(),
    ]);
    let result = time_command("mutool", cmd);
    let _ = std::fs::remove_file(out);
    result
}

/// Render a single page with `pdftoppm -f <p> -l <p> -r 150 <archive> <prefix>`.
pub fn pdftoppm_render(archive: &Path, page: u32, out_prefix: &Path) -> CompetitorResult {
    if Command::new("pdftoppm").arg("-v").output().is_err() {
        return CompetitorResult {
            name: "pdftoppm",
            elapsed_ms: None,
        };
    }
    let p_str = page.to_string();
    let mut cmd = Command::new("pdftoppm");
    cmd.args([
        "-f",
        &p_str,
        "-l",
        &p_str,
        "-r",
        "150",
        archive.to_str().expect("non-utf8 archive path"),
        out_prefix.to_str().expect("non-utf8 prefix"),
    ]);
    let result = time_command("pdftoppm", cmd);
    // pdftoppm always emits `<prefix>-NNNNNN.ppm`; clean only files that
    // match that exact shape so we don't accidentally delete unrelated
    // `<prefix>_log.txt` etc. that a user happened to leave in /tmp.
    if let Some(parent) = out_prefix.parent()
        && let Some(stem) = out_prefix.file_name().and_then(|s| s.to_str())
        && let Ok(entries) = std::fs::read_dir(parent)
    {
        let stem_dash = format!("{stem}-");
        for ent in entries.flatten() {
            let path = ent.path();
            let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
                continue;
            };
            if name.starts_with(&stem_dash) && path.extension().is_some_and(|e| e == "ppm") {
                let _ = std::fs::remove_file(&path);
            }
        }
    }
    result
}
