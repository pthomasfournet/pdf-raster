//! Four-event contest harness (skeleton).
//!
//! Subcommands implemented in this commit:
//!   - `build-archive <out> [bytes]` — concatenate fixtures into a synthetic PDF
//!
//! Subcommands not yet implemented (lands in a follow-up):
//!   - `e1 <archive>` — first-pixel: render page 50000
//!   - `e2 <archive>` — sustained: render pages 50000-50099
//!   - `e3 <archives.txt>` — cross-doc: page 1 of each
//!   - `e4 <archive>` — random-access: 1000 random pages
//!   - `all <archive> <list>` — run all four
//!
//! See: docs/superpowers/specs/2026-05-09-phase-11-million-page-archive-contest.md

mod archive;

use std::path::PathBuf;
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
        "e1" | "e2" | "e3" | "e4" | "all" => {
            eprintln!(
                "{cmd}: not yet implemented in this build of contest_v11; \
                 see ROADMAP.md for the four-event runners."
            );
            ExitCode::from(3)
        }
        other => {
            eprintln!("unknown subcommand: {other}");
            usage();
            ExitCode::from(1)
        }
    }
}

fn usage() {
    eprintln!(
        "usage: contest_v11 <subcommand>

Subcommands:
  build-archive <out> [bytes]   Build a synthetic PDF archive of approximately
                                <bytes> bytes (default: 10 GiB) at <out>.
  e1 <archive>                  (not yet implemented)
  e2 <archive>                  (not yet implemented)
  e3 <archives.txt>             (not yet implemented)
  e4 <archive>                  (not yet implemented)
  all <archive> <list>          (not yet implemented)
"
    );
}
