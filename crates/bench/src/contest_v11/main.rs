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

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let Some(cmd) = args.next() else {
        usage();
        return ExitCode::from(1);
    };
    match cmd.as_str() {
        "build-archive" => {
            let Some(out_arg) = args.next() else {
                eprintln!("usage: contest_v11 build-archive <out> [bytes]");
                return ExitCode::from(1);
            };
            let out: PathBuf = out_arg.into();
            let target_bytes = args.next().map_or(DEFAULT_ARCHIVE_BYTES, |s| {
                s.parse::<u64>()
                    .expect("bytes argument must be a positive integer")
            });
            if let Err(e) = archive::build(&out, target_bytes) {
                eprintln!("build-archive failed: {e}");
                return ExitCode::from(2);
            }
            ExitCode::SUCCESS
        }
        "e1" => {
            let archive: PathBuf = args.next().expect("usage: e1 <archive> [page]").into();
            let page = args
                .next()
                .map_or(50_000, |s| s.parse::<u32>().expect("page must be u32"));
            print_event_with_competitors_e1(&archive, page);
            ExitCode::SUCCESS
        }
        "e2" => {
            let archive: PathBuf = args
                .next()
                .expect("usage: e2 <archive> [first] [count]")
                .into();
            let first = args
                .next()
                .map_or(50_000, |s| s.parse::<u32>().expect("u32"));
            let count = args.next().map_or(100, |s| s.parse::<u32>().expect("u32"));
            print_event_only(events::e2(&archive, first, count));
            ExitCode::SUCCESS
        }
        "e3" => {
            let list: PathBuf = args.next().expect("usage: e3 <archives.txt>").into();
            print_event_only(events::e3(&list));
            ExitCode::SUCCESS
        }
        "e4" => {
            let archive: PathBuf = args.next().expect("usage: e4 <archive>").into();
            print_event_only(events::e4(&archive));
            ExitCode::SUCCESS
        }
        "all" => {
            let archive: PathBuf = args.next().expect("usage: all <archive> <list>").into();
            let list: PathBuf = args.next().expect("usage: all <archive> <list>").into();
            run_all(&archive, &list);
            ExitCode::SUCCESS
        }
        other => {
            eprintln!("unknown subcommand: {other}");
            usage();
            ExitCode::from(1)
        }
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
