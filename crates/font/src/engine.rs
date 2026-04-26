//! Font engine — owns the `FreeType` library and creates [`FontFace`] instances.
//!
//! [`FontEngine`] is the Rust equivalent of `SplashFTFontEngine`.  One
//! instance is created at startup and shared (via `Arc<Mutex<FontEngine>>`)
//! across threads.  It is responsible for:
//!
//! 1. Initialising and holding the `FT_Library` handle.
//! 2. Loading font faces from bytes (embedded PDF font streams) or file paths.
//! 3. Assigning monotonically increasing [`FaceId`] values.
//! 4. Wrapping loaded faces with the rendering parameters needed by [`FontFace`].
//!
//! # Thread safety
//!
//! `FreeType` itself is not thread-safe for concurrent operations on the same
//! `FT_Library`.  `FontEngine` is therefore wrapped in a `Mutex`; callers must
//! lock it for the duration of any face-load call.  Once a [`FontFace`] is
//! constructed it is used without locking from its owning thread.

use std::sync::{Arc, Mutex};

use freetype::Library;

use crate::face::{FontFace, FontMatrix};
use crate::hinting::FontKind;
use crate::key::FaceId;

/// Shared, thread-safe font engine handle.
pub type SharedEngine = Arc<Mutex<FontEngine>>;

/// Parameters passed when loading a font face.
///
/// Grouping into a struct keeps [`FontEngine::load_memory_face`] and
/// [`FontEngine::load_file_face`] under the 7-argument limit.
pub struct FaceParams {
    /// Which kind of outline format this font uses.
    pub kind: FontKind,
    /// Glyph-index map: `code_to_gid[char_code]` → `FT_UInt`.
    /// Pass an empty `Vec` for the identity map.
    pub code_to_gid: Vec<u32>,
    /// 2×2 font/device transform matrix `[a, b, c, d]`.
    pub mat: FontMatrix,
    /// 2×2 text transform matrix `[a, b, c, d]`.
    pub text_mat: FontMatrix,
}

/// The font engine — owns `FT_Library` and assigns face IDs.
pub struct FontEngine {
    lib: Library,
    /// Monotonically increasing face-ID counter.
    next_id: u32,
    /// Whether anti-aliasing is globally enabled.
    pub aa: bool,
    /// Whether `FreeType` hinting is enabled.
    pub ft_hinting: bool,
    /// Whether slight-hinting mode is enabled.
    pub slight_hinting: bool,
}

impl FontEngine {
    /// Initialise the `FreeType` library and return a shared engine handle.
    ///
    /// # Errors
    ///
    /// Returns an error if `FreeType` initialisation fails (library not
    /// found or internal `FT_Init_FreeType` error).
    pub fn init(
        aa: bool,
        ft_hinting: bool,
        slight_hinting: bool,
    ) -> Result<SharedEngine, freetype::Error> {
        let lib = Library::init()?;
        Ok(Arc::new(Mutex::new(Self {
            lib,
            next_id: 0,
            aa,
            ft_hinting,
            slight_hinting,
        })))
    }

    /// Load a face from an in-memory font buffer.
    ///
    /// `font_data` is the raw font file bytes (Type 1, TrueType, CFF, etc.).
    /// `face_index` selects the sub-face within a font collection (0 for
    /// single-face files).
    ///
    /// # Errors
    ///
    /// Returns [`LoadError::FreeType`] if `FreeType` cannot parse the data,
    /// or [`LoadError::DegenerateSize`] if the size matrix is degenerate.
    pub fn load_memory_face(
        &mut self,
        font_data: Vec<u8>,
        face_index: isize,
        params: FaceParams,
    ) -> Result<FontFace, LoadError> {
        let ft_face = self
            .lib
            .new_memory_face(font_data, face_index)
            .map_err(LoadError::FreeType)?;

        let id = self.alloc_id();
        FontFace::new(
            id,
            ft_face,
            params,
            self.aa,
            self.ft_hinting,
            self.slight_hinting,
        )
        .ok_or(LoadError::DegenerateSize)
    }

    /// Load a face from a file path.
    ///
    /// `face_index` selects the sub-face within a font collection.
    ///
    /// # Errors
    ///
    /// Returns [`LoadError::FreeType`] if `FreeType` cannot open or parse the
    /// file, or [`LoadError::DegenerateSize`] if the size matrix is degenerate.
    pub fn load_file_face(
        &mut self,
        path: &str,
        face_index: isize,
        params: FaceParams,
    ) -> Result<FontFace, LoadError> {
        let ft_face = self
            .lib
            .new_face(path, face_index)
            .map_err(LoadError::FreeType)?;

        let id = self.alloc_id();
        FontFace::new(
            id,
            ft_face,
            params,
            self.aa,
            self.ft_hinting,
            self.slight_hinting,
        )
        .ok_or(LoadError::DegenerateSize)
    }

    /// Allocate the next `FaceId`, incrementing the internal counter.
    ///
    /// `FaceId`s wrap at `u32::MAX` — in practice a process loads at most tens of
    /// thousands of faces, so wrap-around never occurs.
    fn alloc_id(&mut self) -> FaceId {
        let id = FaceId(self.next_id);
        debug_assert!(
            self.next_id < u32::MAX,
            "FaceId counter wrapped; too many font faces loaded in this process"
        );
        self.next_id = self.next_id.wrapping_add(1);
        id
    }
}

/// Errors that can occur when loading a font face.
#[derive(Debug)]
pub enum LoadError {
    /// `FreeType` returned an error (e.g. corrupt font data, missing file).
    FreeType(freetype::Error),
    /// The font size matrix is degenerate (zero scale or zero bounding box).
    DegenerateSize,
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FreeType(e) => write!(f, "FreeType error: {e:?}"),
            Self::DegenerateSize => {
                write!(f, "font face has degenerate size matrix (zero scale)")
            }
        }
    }
}

impl std::error::Error for LoadError {}
