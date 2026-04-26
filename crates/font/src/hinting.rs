//! `FreeType` load-flag policy, matching `SplashFTFont::getFTLoadFlags`.
//!
//! The hinting flags are derived from three inputs:
//! - font type (Type 1 vs TrueType vs CFF/other)
//! - whether anti-aliasing is enabled
//! - global hinting / slight-hinting preferences
//!
//! The mapping is a direct port of the C++ static function
//! `getFTLoadFlags` in `SplashFTFont.cc`.

use freetype::face::LoadFlag;

/// Font type, used to select the appropriate `FreeType` hinting mode.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FontKind {
    /// PostScript Type 1 (or Type 1C / CFF).
    Type1,
    /// TrueType (or OpenType with TrueType outlines).
    TrueType,
    /// All other outline formats (OpenType/CFF, CID, etc.).
    Other,
}

/// Compute the `FreeType` `LOAD_*` flags for a given rendering context.
///
/// Mirrors `getFTLoadFlags` from `SplashFTFont.cc` exactly:
///
/// | condition | flag added |
/// |-----------|-----------|
/// | AA enabled | `NO_BITMAP` (skip embedded bitmaps, render outline) |
/// | hinting disabled | `NO_HINTING` |
/// | slight hinting | `TARGET_LIGHT` |
/// | TrueType + AA + normal hinting | `NO_AUTOHINT` (FT2 autohinter unreliable on subsets) |
/// | Type 1 + normal hinting | `TARGET_LIGHT` |
#[must_use]
pub fn load_flags(kind: FontKind, aa: bool, ft_hinting: bool, slight_hinting: bool) -> LoadFlag {
    let mut flags = LoadFlag::DEFAULT;

    if aa {
        flags |= LoadFlag::NO_BITMAP;
    }

    if ft_hinting {
        if slight_hinting {
            flags |= LoadFlag::TARGET_LIGHT;
        } else {
            match kind {
                FontKind::TrueType => {
                    // FT2's autohinting doesn't always work well with font subsets;
                    // disable it when anti-aliasing is on to avoid halo artefacts.
                    if aa {
                        flags |= LoadFlag::NO_AUTOHINT;
                    }
                }
                FontKind::Type1 => {
                    // Type 1 fonts render better with light hinting.
                    flags |= LoadFlag::TARGET_LIGHT;
                }
                FontKind::Other => {}
            }
        }
    } else {
        flags |= LoadFlag::NO_HINTING;
    }

    flags
}

#[cfg(test)]
mod tests {
    use super::*;
    use freetype::face::LoadFlag;

    #[test]
    fn no_hinting_overrides_all() {
        let flags = load_flags(FontKind::TrueType, true, false, false);
        assert!(
            flags.contains(LoadFlag::NO_HINTING),
            "hinting disabled must set NO_HINTING"
        );
        assert!(
            !flags.contains(LoadFlag::TARGET_LIGHT),
            "NO_HINTING must not also set TARGET_LIGHT"
        );
    }

    #[test]
    fn aa_adds_no_bitmap() {
        let flags = load_flags(FontKind::Other, true, true, false);
        assert!(
            flags.contains(LoadFlag::NO_BITMAP),
            "AA must add NO_BITMAP to render outlines"
        );
    }

    #[test]
    fn no_aa_no_bitmap_flag() {
        let flags = load_flags(FontKind::Other, false, true, false);
        assert!(
            !flags.contains(LoadFlag::NO_BITMAP),
            "without AA, embedded bitmaps should be allowed"
        );
    }

    #[test]
    fn slight_hinting_adds_target_light() {
        let flags = load_flags(FontKind::TrueType, false, true, true);
        assert!(
            flags.contains(LoadFlag::TARGET_LIGHT),
            "slight hinting must set TARGET_LIGHT"
        );
    }

    #[test]
    fn truetype_aa_normal_hinting_adds_no_autohint() {
        let flags = load_flags(FontKind::TrueType, true, true, false);
        assert!(
            flags.contains(LoadFlag::NO_AUTOHINT),
            "TrueType + AA + normal hinting must disable autohinter"
        );
    }

    #[test]
    fn truetype_mono_normal_hinting_no_autohint() {
        // Without AA, the autohinter tossup is left to `FreeType`.
        let flags = load_flags(FontKind::TrueType, false, true, false);
        assert!(
            !flags.contains(LoadFlag::NO_AUTOHINT),
            "TrueType without AA must not forcibly disable autohinter"
        );
    }

    #[test]
    fn type1_normal_hinting_adds_target_light() {
        let flags = load_flags(FontKind::Type1, false, true, false);
        assert!(
            flags.contains(LoadFlag::TARGET_LIGHT),
            "Type 1 + normal hinting must set TARGET_LIGHT"
        );
    }
}
