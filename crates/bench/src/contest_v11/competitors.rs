//! mutool / pdftoppm subprocess wrappers.
//!
//! Each function returns `Some(elapsed_ms)` if the competitor is installed
//! and runs successfully, or `None` if the competitor is missing, fails to
//! spawn, or exits non-zero.  Stderr is forwarded to the harness's stderr
//! when a subprocess fails so the operator can see why.
//!
//! Non-UTF-8 paths are not supported — the harness uses ASCII-only `/tmp`
//! prefixes by construction.

use std::path::Path;
use std::process::Command;
use std::time::Instant;

#[derive(Debug)]
pub struct CompetitorResult {
    pub name: &'static str,
    pub elapsed_ms: Option<f64>,
}

/// Time a fully-built `Command`, returning `Some(elapsed_ms)` only on
/// successful exit.  On any failure (spawn error, non-zero exit) emit a
/// stderr breadcrumb so `NOT INSTALLED or FAILED` in the report has a
/// matching log line for diagnosis.
fn time_command(name: &'static str, mut cmd: Command) -> CompetitorResult {
    let t0 = Instant::now();
    match cmd.output() {
        Ok(out) if out.status.success() => CompetitorResult {
            name,
            elapsed_ms: Some(t0.elapsed().as_secs_f64() * 1e3),
        },
        Ok(out) => {
            eprintln!(
                "[{name}] failed (exit {}): {}",
                out.status,
                String::from_utf8_lossy(&out.stderr).trim()
            );
            CompetitorResult {
                name,
                elapsed_ms: None,
            }
        }
        Err(e) => {
            eprintln!("[{name}] not spawnable: {e}");
            CompetitorResult {
                name,
                elapsed_ms: None,
            }
        }
    }
}

/// Render a single page with `mutool draw -r 150 -o <out> <archive> <page>`.
/// Returns `None` if mutool is not on PATH or the run fails; `time_command`
/// already detects spawn-failure, so no separate install probe is needed.
/// Cleans the output file before returning so on-disk cost is not
/// accidentally measured on subsequent runs.
pub fn mutool_render(archive: &Path, page: u32, out: &Path) -> CompetitorResult {
    let (Some(out_str), Some(archive_str)) = (out.to_str(), archive.to_str()) else {
        return CompetitorResult {
            name: "mutool",
            elapsed_ms: None,
        };
    };
    let mut cmd = Command::new("mutool");
    cmd.args([
        "draw",
        "-r",
        "150",
        "-o",
        out_str,
        archive_str,
        &page.to_string(),
    ]);
    let result = time_command("mutool", cmd);
    let _ = std::fs::remove_file(out);
    result
}

/// Render a single page with `pdftoppm -f <p> -l <p> -r 150 <archive> <prefix>`.
/// Returns `None` if pdftoppm is not on PATH or the run fails.
pub fn pdftoppm_render(archive: &Path, page: u32, out_prefix: &Path) -> CompetitorResult {
    let (Some(prefix_str), Some(archive_str)) = (out_prefix.to_str(), archive.to_str()) else {
        return CompetitorResult {
            name: "pdftoppm",
            elapsed_ms: None,
        };
    };
    let p_str = page.to_string();
    let mut cmd = Command::new("pdftoppm");
    cmd.args([
        "-f",
        &p_str,
        "-l",
        &p_str,
        "-r",
        "150",
        archive_str,
        prefix_str,
    ]);
    let result = time_command("pdftoppm", cmd);
    // pdftoppm zero-pads its page index — `<prefix>-000123.ppm` for page 123.
    // Build the exact path and unlink it.  Avoids the `read_dir(/tmp)` scan
    // that the previous shape did and removes the risk of deleting an
    // unrelated neighbour file that happened to share the prefix.
    let expected_out = out_prefix.with_file_name(format!(
        "{}-{:06}.ppm",
        out_prefix
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(""),
        page,
    ));
    let _ = std::fs::remove_file(&expected_out);
    result
}
