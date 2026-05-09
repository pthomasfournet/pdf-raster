//! Four-event contest harness.
//!
//! Subcommands:
//!   - `build-archive <out> [bytes]` — concatenate fixtures into a synthetic PDF
//!   - `e1 <archive> [page]` — first-pixel: render single page
//!   - `e2 <archive> [first] [count]` — sustained: render N consecutive pages
//!   - `e3 <archives.txt>` — cross-doc: page 1 of each
//!   - `e4 <archive>` — random-access: 1000 random pages
//!   - `all <archive> <list>` — run all four
//!
//! See: docs/superpowers/specs/2026-05-09-phase-11-million-page-archive-contest.md

mod archive;
mod competitors;
mod events;
mod io_uring_open;

use std::path::{Path, PathBuf};
use std::process::ExitCode;

const DEFAULT_ARCHIVE_BYTES: u64 = 10 * 1024 * 1024 * 1024; // 10 GB

/// Pull the next positional arg or print `hint` and exit 1.  Avoids the
/// `.expect()`-with-Rust-panic-style-message UX that was here before
/// (a CLI tool's missing-arg message should not look like a crash).
fn next_or_exit(args: &mut impl Iterator<Item = String>, hint: &str) -> Result<String, ExitCode> {
    if let Some(v) = args.next() {
        Ok(v)
    } else {
        eprintln!("{hint}");
        Err(ExitCode::from(1))
    }
}

/// Parse the next optional positional arg (e.g. page index) as `u32`,
/// or fall back to `default` when absent.  Exits 1 on parse error.
fn next_u32_or(
    args: &mut impl Iterator<Item = String>,
    default: u32,
    label: &str,
) -> Result<u32, ExitCode> {
    let Some(s) = args.next() else {
        return Ok(default);
    };
    s.parse::<u32>().map_err(|_| {
        eprintln!("{label}: expected u32, got {s:?}");
        ExitCode::from(1)
    })
}

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let Some(cmd) = args.next() else {
        usage();
        return ExitCode::from(1);
    };

    let result = (|| -> Result<ExitCode, ExitCode> {
        match cmd.as_str() {
            "build-archive" => {
                let out: PathBuf =
                    next_or_exit(&mut args, "usage: contest_v11 build-archive <out> [bytes]")?
                        .into();
                let target_bytes = match args.next() {
                    None => DEFAULT_ARCHIVE_BYTES,
                    Some(s) => s.parse::<u64>().map_err(|_| {
                        eprintln!("bytes: expected u64, got {s:?}");
                        ExitCode::from(1)
                    })?,
                };
                if let Err(e) = archive::build(&out, target_bytes) {
                    eprintln!("build-archive failed: {e}");
                    return Ok(ExitCode::from(2));
                }
                Ok(ExitCode::SUCCESS)
            }
            "e1" => {
                let archive: PathBuf =
                    next_or_exit(&mut args, "usage: e1 <archive> [page]")?.into();
                let page = next_u32_or(&mut args, 50_000, "page")?;
                print_event_with_competitors_e1(&archive, page);
                Ok(ExitCode::SUCCESS)
            }
            "e2" => {
                let archive: PathBuf =
                    next_or_exit(&mut args, "usage: e2 <archive> [first] [count]")?.into();
                let first = next_u32_or(&mut args, 50_000, "first")?;
                let count = next_u32_or(&mut args, 100, "count")?;
                print_event_only(events::e2(&archive, first, count));
                Ok(ExitCode::SUCCESS)
            }
            "e3" => {
                let list: PathBuf = next_or_exit(&mut args, "usage: e3 <archives.txt>")?.into();
                print_event_only(events::e3(&list));
                Ok(ExitCode::SUCCESS)
            }
            "e4" => {
                let archive: PathBuf = next_or_exit(&mut args, "usage: e4 <archive>")?.into();
                print_event_only(events::e4(&archive));
                Ok(ExitCode::SUCCESS)
            }
            "all" => {
                let archive: PathBuf =
                    next_or_exit(&mut args, "usage: all <archive> <list>")?.into();
                let list: PathBuf = next_or_exit(&mut args, "usage: all <archive> <list>")?.into();
                run_all(&archive, &list);
                Ok(ExitCode::SUCCESS)
            }
            other => {
                eprintln!("unknown subcommand: {other}");
                usage();
                Ok(ExitCode::from(1))
            }
        }
    })();
    match result {
        Ok(code) | Err(code) => code,
    }
}

fn print_event_only(r: Result<events::EventResult, String>) {
    match r {
        Ok(ev) => println!(
            "{}: {:.1} ms ({} pages)",
            ev.name, ev.elapsed_ms, ev.pages_rendered
        ),
        Err(e) => eprintln!("error: {e}"),
    }
}

fn print_event_with_competitors_e1(archive: &Path, page: u32) {
    match events::e1(archive, page) {
        Ok(ev) => println!("ours    : {:.1} ms (E1, page {page})", ev.elapsed_ms),
        Err(e) => {
            eprintln!("ours    : error: {e}");
            return;
        }
    }
    let mu = competitors::mutool_render(archive, page, Path::new("/tmp/p11_mutool_e1.ppm"));
    let pp = competitors::pdftoppm_render(archive, page, Path::new("/tmp/p11_pdftoppm_e1"));
    print_competitor(&mu);
    print_competitor(&pp);
}

fn print_competitor(r: &competitors::CompetitorResult) {
    match r.elapsed_ms {
        Some(ms) => println!("{:8}: {ms:.1} ms", r.name),
        None => println!("{:8}: NOT INSTALLED or FAILED", r.name),
    }
}

fn run_all(archive: &Path, list: &Path) {
    println!("=== E1 ===");
    print_event_with_competitors_e1(archive, 50_000);
    println!("\n=== E2 ===");
    print_event_only(events::e2(archive, 50_000, 100));
    println!("\n=== E3 ===");
    print_event_only(events::e3(list));
    println!("\n=== E4 ===");
    print_event_only(events::e4(archive));
}

fn usage() {
    eprintln!(
        "usage: contest_v11 <subcommand>

Subcommands:
  build-archive <out> [bytes]   Build a synthetic PDF archive of approximately
                                <bytes> bytes (default: 10 GiB) at <out>.
  e1 <archive> [page]           First-pixel: render a single page.
  e2 <archive> [first] [count]  Sustained: render N consecutive pages.
  e3 <archives.txt>             Cross-doc: render page 1 of each archive.
  e4 <archive>                  Random-access: render 1000 random pages.
  all <archive> <list>          Run all four events back-to-back.
"
    );
}
