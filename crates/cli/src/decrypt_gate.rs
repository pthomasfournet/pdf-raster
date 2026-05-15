//! Liability gate for qpdf-assisted decryption on the production CLI path.
//!
//! The QA harness auto-decrypts (private owned-texts tooling); the CLI
//! must NOT silently strip copy-protection.  An encrypted document here
//! triggers an interactive private-copy / liability confirmation
//! (default: no), unless the operator has explicitly affirmed
//! authorisation via `--decrypt-owned` or `RROCKET_DECRYPT_OWNED=1`.
//!
//! A non-interactive process with no bypass set never auto-proceeds; it
//! aborts with a clear, accurate error instead.

use std::io::{IsTerminal, Write};

/// The liability waiver shown on stderr before the interactive prompt.
pub const DISCLAIMER: &str = "\
This document is encrypted (PDF Standard Security Handler). Continuing will \
remove its copy-protection using qpdf. This capability is intended SOLELY \
for documents you own or are otherwise legally entitled to access (e.g. \
personal copies of books you have purchased). By continuing you affirm you \
have the lawful right to decrypt this document. The authors of this \
software accept NO liability for any use of this feature, and provide it \
WITHOUT WARRANTY. This is not legal advice. Proceed? [y/N]";

/// Environment variable that, when set to `1`, bypasses the interactive
/// prompt (operator has affirmed authorisation for unattended use).
pub const ENV_BYPASS: &str = "RROCKET_DECRYPT_OWNED";

/// Pure decision function for the decrypt gate — no I/O, fully unit
/// testable.
///
/// - `flag`  — `--decrypt-owned` was passed.
/// - `env`   — `RROCKET_DECRYPT_OWNED=1` is set.
/// - `interactive` — stdin is a terminal (a human can answer).
/// - `answer` — the first non-whitespace character the user typed at the
///   prompt, lowercased, or `None` if the prompt was not shown / no input.
///
/// Decision table:
/// - Explicit bypass (`flag` or `env`) → proceed, no prompt.
/// - Non-interactive with no bypass → abort (never auto-proceed silently).
/// - Interactive: proceed ONLY on an explicit `y` answer; default (and any
///   other answer, including `None`) → abort.
#[must_use]
pub const fn should_decrypt(
    interactive: bool,
    flag: bool,
    env: bool,
    answer: Option<char>,
) -> bool {
    if flag || env {
        return true;
    }
    if !interactive {
        return false;
    }
    matches!(answer, Some('y'))
}

/// Read `RROCKET_DECRYPT_OWNED` and return `true` only for the exact
/// value `1` (a defined opt-in, not merely "set to anything").
#[must_use]
pub fn env_bypass_set() -> bool {
    std::env::var(ENV_BYPASS).is_ok_and(|v| v == "1")
}

/// Run the interactive liability gate and return whether decryption is
/// authorised.
///
/// Prints the disclaimer + prompt to stderr and reads one line from
/// stdin.  Non-interactive stdin with no bypass returns `false` without
/// prompting (the caller surfaces the clear `EncryptedDocument` error).
#[must_use]
pub fn prompt_decrypt(flag: bool) -> bool {
    let env = env_bypass_set();
    let interactive = std::io::stdin().is_terminal();

    // Fast paths: explicit bypass, or non-interactive (no prompt possible).
    if flag || env {
        return should_decrypt(interactive, flag, env, None);
    }
    if !interactive {
        return false;
    }

    let mut stderr = std::io::stderr();
    let _ = writeln!(stderr, "{DISCLAIMER}");
    let _ = stderr.flush();

    let mut line = String::new();
    let answer = match std::io::stdin().read_line(&mut line) {
        Ok(0) | Err(_) => None, // EOF or read error → default (no).
        Ok(_) => line.trim().chars().next().map(|c| c.to_ascii_lowercase()),
    };

    should_decrypt(interactive, flag, env, answer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flag_bypass_proceeds_even_noninteractive() {
        assert!(should_decrypt(false, true, false, None));
        assert!(should_decrypt(true, true, false, None));
    }

    #[test]
    fn env_bypass_proceeds_even_noninteractive() {
        assert!(should_decrypt(false, false, true, None));
    }

    #[test]
    fn noninteractive_no_bypass_aborts() {
        assert!(!should_decrypt(false, false, false, None));
        assert!(!should_decrypt(false, false, false, Some('y')));
    }

    #[test]
    fn interactive_defaults_to_no() {
        // No answer (just Enter) → abort.
        assert!(!should_decrypt(true, false, false, None));
        // Explicit "n" → abort.
        assert!(!should_decrypt(true, false, false, Some('n')));
        // Anything that is not 'y' → abort.
        assert!(!should_decrypt(true, false, false, Some('x')));
    }

    #[test]
    fn interactive_explicit_yes_proceeds() {
        assert!(should_decrypt(true, false, false, Some('y')));
    }

    #[test]
    fn bypass_dominates_a_no_answer() {
        // Operator bypass set: a stray "n" at a (non-shown) prompt is
        // irrelevant — the operator already affirmed authorisation.
        assert!(should_decrypt(true, true, false, Some('n')));
        assert!(should_decrypt(true, false, true, Some('n')));
    }

    #[test]
    fn disclaimer_has_required_substance() {
        assert!(DISCLAIMER.contains("PDF Standard Security Handler"));
        assert!(DISCLAIMER.contains("qpdf"));
        assert!(DISCLAIMER.contains("NO liability"));
        assert!(DISCLAIMER.contains("WITHOUT WARRANTY"));
        assert!(DISCLAIMER.contains("not legal advice"));
        assert!(DISCLAIMER.contains("[y/N]"));
    }
}
