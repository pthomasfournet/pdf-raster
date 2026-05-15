//! qpdf-assisted decryption preprocess for encrypted PDFs.
//!
//! The renderer has no in-process crypto subsystem (by design — out of
//! scope).  Encrypted documents using the PDF Standard Security Handler
//! are instead handed off to the external `qpdf` tool, which writes an
//! unencrypted copy that the parser then opens transparently.
//!
//! The corpus's encrypted documents are password-less, permissions-only
//! PDFs (`User password` empty; restrictive permission flags only).
//! `qpdf --decrypt --password=` removes the `/Encrypt` dictionary on all
//! of them.  A PDF that requires a real password causes qpdf to exit
//! non-zero; that case is reported accurately rather than masked as
//! "document has no pages".
//!
//! The caller decides *whether* decryption is authorised (the CLI gates
//! it behind an interactive liability waiver / explicit operator bypass;
//! the private QA harness auto-authorises).  This module only performs
//! the mechanical decrypt once told it may.

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::error::PdfError;

/// Error message when `qpdf` is not on `PATH`.
const MSG_QPDF_ABSENT: &str = "document is encrypted (PDF Standard Security Handler) and qpdf was not \
     found on PATH — install qpdf to open encrypted PDFs";

/// Error message when qpdf ran but failed (typically: a real password is
/// required, which this preprocess intentionally does not handle).
fn msg_password_protected(qpdf_stderr: &str) -> String {
    let tail = qpdf_stderr.trim();
    if tail.is_empty() {
        "document is encrypted and password-protected; password-based \
         decryption is not supported"
            .to_owned()
    } else {
        format!(
            "document is encrypted and could not be decrypted by qpdf \
             (likely password-protected; password-based decryption is not \
             supported). qpdf: {tail}"
        )
    }
}

/// Error message handed back when the CLI liability gate is declined or
/// the process is non-interactive with no operator bypass set.
#[must_use]
pub fn msg_gate_declined() -> String {
    "document is encrypted (PDF Standard Security Handler); decryption was \
     not authorised. Re-run interactively and confirm the private-copy / \
     liability prompt, or set --decrypt-owned / RROCKET_DECRYPT_OWNED=1 for \
     unattended use."
        .to_owned()
}

/// RAII guard that owns a qpdf-decrypted temporary file and removes it on
/// drop.  The decrypted PDF must outlive the memory map taken over it, so
/// the [`crate::Document`] holds this guard for its whole lifetime.
///
/// Backed by `tempfile::NamedTempFile`, whose own `Drop` unlinks the file;
/// the guard wrapper exists so the type can be stored opaquely and so the
/// path is queryable without exposing the `tempfile` type in the public
/// API.
#[derive(Debug)]
pub struct DecryptGuard {
    /// `None` for documents that were not encrypted (no temp file created;
    /// the original path is used directly).  Held only so
    /// `NamedTempFile`'s `Drop` unlinks the decrypted plaintext when the
    /// owning `Document` is dropped; observable via [`Self::has_temp`].
    temp: Option<tempfile::NamedTempFile>,
}

impl DecryptGuard {
    /// A no-op guard for documents that did not need decryption.
    #[must_use]
    pub const fn none() -> Self {
        Self { temp: None }
    }

    /// True when this guard owns a decrypted temp file (i.e. the document
    /// was encrypted and qpdf produced a plaintext copy).  Exists so the
    /// RAII field has a real read path (it is otherwise only consulted by
    /// `NamedTempFile`'s own `Drop`) and so callers/tests can assert that
    /// unencrypted inputs never spawned qpdf.
    #[must_use]
    pub const fn has_temp(&self) -> bool {
        self.temp.is_some()
    }
}

/// Run `qpdf --decrypt --password= <input> <tempfile>` and return a guard
/// owning the decrypted temp file plus its path.
///
/// The empty password covers every password-less permissions-only PDF.
/// A document needing a real password makes qpdf exit non-zero — surfaced
/// as [`PdfError::EncryptedDocument`] with an accurate message, never
/// "no pages".
///
/// # Errors
/// - [`PdfError::EncryptedDocument`] if `qpdf` is absent on `PATH`.
/// - [`PdfError::EncryptedDocument`] if qpdf exits non-zero (password-
///   protected or otherwise undecryptable).
/// - [`PdfError::Io`] if the temp file could not be created.
pub fn qpdf_decrypt_to_temp(input: &Path) -> Result<(PathBuf, DecryptGuard), PdfError> {
    // NamedTempFile under the system temp dir; its Drop unlinks the file
    // even on the error paths below (the early returns drop the partially
    // written temp), so no decrypted plaintext is leaked.
    let tmp = tempfile::Builder::new()
        .prefix("rrocket-decrypt-")
        .suffix(".pdf")
        .tempfile()?;
    let tmp_path = tmp.path().to_path_buf();

    let output = Command::new("qpdf")
        .arg("--decrypt")
        .arg("--password=")
        .arg(input)
        .arg(&tmp_path)
        .output();

    let output = match output {
        Ok(o) => o,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            // `tmp` drops here → temp file unlinked.
            return Err(PdfError::EncryptedDocument(MSG_QPDF_ABSENT.to_owned()));
        }
        Err(e) => return Err(PdfError::Io(e)),
    };

    // qpdf exit codes: 0 = success, 3 = success with warnings (the
    // decrypted copy is still usable).  Anything else (notably 2 =
    // invalid password) is a hard failure.
    let code = output.status.code();
    if !matches!(code, Some(0) | Some(3)) {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(PdfError::EncryptedDocument(msg_password_protected(&stderr)));
    }

    Ok((tmp_path, DecryptGuard { temp: Some(tmp) }))
}

/// Resolve the path the parser should actually open.
///
/// - Not encrypted → `(original path, no-op guard)`; no qpdf spawn, no
///   temp file.  This is the hot path for unencrypted documents and must
///   stay a pure structural check.
/// - Encrypted + `authorized == false` → [`PdfError::EncryptedDocument`]
///   (gate declined); fail accurately rather than silently strip.
/// - Encrypted + `authorized == true` → qpdf-decrypt to a temp file and
///   return that path; the returned guard cleans the temp file on drop.
///
/// `is_encrypted` is supplied by the caller (it has already opened the
/// document to read the trailer) so this function does not re-parse.
///
/// # Errors
/// Propagates [`qpdf_decrypt_to_temp`]'s errors, plus the gate-declined
/// [`PdfError::EncryptedDocument`].
pub fn resolve_source(
    original: &Path,
    is_encrypted: bool,
    authorized: bool,
) -> Result<(PathBuf, DecryptGuard), PdfError> {
    if !is_encrypted {
        return Ok((original.to_path_buf(), DecryptGuard::none()));
    }
    if !authorized {
        return Err(PdfError::EncryptedDocument(msg_gate_declined()));
    }
    qpdf_decrypt_to_temp(original)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unencrypted_is_passthrough_no_temp() {
        let p = Path::new("/tmp/whatever.pdf");
        let (resolved, guard) = resolve_source(p, false, false).expect("passthrough");
        assert_eq!(resolved, p);
        assert!(!guard.has_temp(), "no temp file for unencrypted input");
    }

    #[test]
    fn encrypted_unauthorized_errors_clearly() {
        let p = Path::new("/tmp/enc.pdf");
        let err = resolve_source(p, true, false).expect_err("must refuse");
        match err {
            PdfError::EncryptedDocument(msg) => {
                assert!(msg.contains("not authorised"));
                assert!(!msg.contains("no pages"));
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn gate_declined_message_is_actionable() {
        let m = msg_gate_declined();
        assert!(m.contains("--decrypt-owned"));
        assert!(m.contains("RROCKET_DECRYPT_OWNED=1"));
        assert!(!m.contains("no pages"));
    }

    #[test]
    fn qpdf_absent_message_is_actionable() {
        assert!(MSG_QPDF_ABSENT.contains("install qpdf"));
        assert!(!MSG_QPDF_ABSENT.contains("no pages"));
    }

    #[test]
    fn password_protected_message_distinct_from_absent() {
        let m = msg_password_protected("qpdf: invalid password");
        assert!(m.contains("password"));
        assert!(!m.contains("install qpdf"));
        assert!(!m.contains("no pages"));
    }
}
