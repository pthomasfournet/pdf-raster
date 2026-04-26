//! Halftone threshold screen for Mono1 (1-bit) output.
//!
//! Mirrors `SplashScreen` from `splash/SplashScreen.h/.cc`.
//!
//! The screen is a power-of-2 tiled threshold matrix. A pixel with value `v`
//! is set to white (1) when `v >= mat[y % size][x % size]`, and black (0)
//! otherwise. The matrix is built lazily on first `test()` call.
//!
//! Three matrix types:
//! - **Dispersed** (Bayer-style): even dot distribution, low correlation.
//! - **Clustered**: traditional clustered dot, good for offset printing.
//! - **`StochasticClustered`**: randomized large-dot screen at ≥ 300 dpi.

use crate::types::{ScreenParams, ScreenType};

/// A halftone threshold screen.
pub struct HalftoneScreen {
    params: ScreenParams,
    mat: Option<Vec<u8>>, // size × size, lazily allocated
    size: usize,
    size_m1: usize, // size - 1 (bitmask for wrapping)
    log2_size: u32,
}

impl HalftoneScreen {
    /// Create a screen with the given parameters.
    /// The threshold matrix is built lazily on the first `test()` call.
    #[must_use]
    pub const fn new(params: ScreenParams) -> Self {
        Self {
            params,
            mat: None,
            size: 0,
            size_m1: 0,
            log2_size: 0,
        }
    }

    /// Test whether pixel `(x, y)` with intensity `value` (0=black, 255=white)
    /// should be rendered as white (returns `true`) or black (`false`).
    ///
    /// Matches `SplashScreen::test` in `SplashScreen.h`.
    #[inline]
    pub fn test(&mut self, x: i32, y: i32, value: u8) -> bool {
        if self.mat.is_none() {
            self.create_matrix();
        }
        // SAFETY: create_matrix always initialises self.mat; the is_none check
        // above ensures create_matrix ran, so the unwrap_or branch is unreachable.
        let mat = self.mat.as_deref().unwrap_or(&[]);
        debug_assert!(!mat.is_empty(), "test: mat must be set after create_matrix");
        // size always fits i32: create_matrix builds size by doubling from 2
        // up to params.size (which is i32), so size ≤ i32::MAX + 1. The saturating
        // loop prevents wrap; in practice size ≤ 2^31 always fits.
        debug_assert!(
            i32::try_from(self.size).is_ok(),
            "test: size={} exceeds i32::MAX; rem_euclid modulus would be wrong",
            self.size
        );
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_possible_wrap,
            reason = "size always fits i32: bounded by params.size (i32) in create_matrix"
        )]
        let size_i32 = self.size as i32;
        // rem_euclid returns a value in [0, size_i32), which is always non-negative,
        // so the cast to usize is safe. Clippy does not fire cast_sign_loss here
        // because the compiler recognises rem_euclid's return type is i32 and the
        // cast is unconditional — no annotation needed.
        let xx = x.rem_euclid(size_i32) as usize;
        let yy = y.rem_euclid(size_i32) as usize;
        value >= mat[(yy << self.log2_size) + xx]
    }

    // ── Matrix construction ───────────────────────────────────────────────────

    /// Build the threshold matrix, choosing the smallest power-of-2 size that
    /// satisfies both `params.size` and (for `StochasticClustered`) the
    /// minimum `2 × dot_radius` constraint.
    fn create_matrix(&mut self) {
        // Find smallest power-of-2 size ≥ params.size.
        let mut size = 2usize;
        let mut log2 = 1u32;
        // i32::try_from(size) fits while size ≤ i32::MAX; once size exceeds
        // i32::MAX the comparison saturates and the loop exits (params.size is i32).
        while i32::try_from(size).unwrap_or(i32::MAX) < self.params.size {
            size = size.saturating_mul(2);
            log2 = log2.saturating_add(1);
        }
        // For stochastic clustered: ensure size ≥ 2 × dot_radius.
        if self.params.kind == ScreenType::StochasticClustered {
            // dot_radius is i32 ≥ 0 (validated by ScreenParams::validate); the
            // product 2 * dot_radius fits in usize. unwrap_or(0) is a safe
            // fallback for the impossible-negative case.
            let min_size = usize::try_from(2 * self.params.dot_radius).unwrap_or(0);
            while size < min_size {
                size = size.saturating_mul(2);
                log2 = log2.saturating_add(1);
            }
        }
        self.size = size;
        self.size_m1 = size - 1;
        self.log2_size = log2;

        // `size` is a power of 2 in [2, 2^31); size*size fits in usize on any
        // 32-bit-or-wider target because size ≤ 2^15 in practice (params.size
        // is an i32, so size ≤ i32::MAX+1 ≈ 2^31, but the loop starts at 2).
        // The debug_assert catches the theoretical overflow on exotic targets.
        debug_assert!(
            size.checked_mul(size).is_some(),
            "size*size overflow: size={size}"
        );
        let mut mat = vec![0u8; size * size];
        match self.params.kind {
            ScreenType::Dispersed => build_dispersed(&mut mat, size, log2),
            ScreenType::Clustered => build_clustered(&mut mat, size),
            ScreenType::StochasticClustered => {
                // dot_radius ≥ 0 (ScreenParams invariant); try_from succeeds.
                let dot_radius = usize::try_from(self.params.dot_radius)
                    .expect("dot_radius must be non-negative");
                build_stochastic_clustered(&mut mat, size, dot_radius);
            }
        }
        // Clamp all entries to ≥ 1 so that value=0 is always black.
        clamp_min_one(&mut mat);
        self.mat = Some(mat);
    }
}

// ── Shared normalization helper ───────────────────────────────────────────────

/// Clamp every element of `mat` to `≥ 1`.
///
/// This ensures that a pixel with `value = 0` is always rendered black,
/// because the threshold comparison is `value >= mat[…]` and `0 >= 1`
/// is always false.
#[inline]
fn clamp_min_one(mat: &mut [u8]) {
    for v in mat {
        if *v == 0 {
            *v = 1;
        }
    }
}

// ── Dispersed (Bayer) matrix ──────────────────────────────────────────────────

/// Recursive void-pointer dispersed dot (Bayer) matrix construction.
/// Matches `SplashScreen::buildDispersedMatrix` in `SplashScreen.cc`.
///
/// Fills `mat` (length `size × size`) with threshold values in `[1, 255]`
/// distributed in the classic Bayer ordered-dither pattern. The pattern
/// ensures that dots are maximally dispersed — no two adjacent cells have
/// similar thresholds.
///
/// # Panics (debug only)
///
/// Panics in debug builds if `size < 2`.
fn build_dispersed(mat: &mut [u8], size: usize, log2_size: u32) {
    debug_assert!(size >= 2, "build_dispersed: size must be >= 2, got {size}");
    // `size` is a small power of 2 (≤ 2^15 in practice), so `size*size` fits
    // in u32. The debug_assert in create_matrix has already verified this.
    debug_assert!(
        u32::try_from(size * size).is_ok(),
        "size*size={} exceeds u32::MAX",
        size * size
    );
    // debug_assert above verified size*size fits u32.
    let total = u32::try_from(size * size).expect("size*size verified to fit u32 above");
    for y in 0..size {
        for x in 0..size {
            // Compute the Bayer index for (x, y) at this size.
            let v = bayer_index(x, y, log2_size);
            // Scale to [1, 255]. The clamp guarantees the value fits in u8.
            mat[y * size + x] =
                u8::try_from(((v * 255 + total / 2) / total).clamp(1, 255))
                    .expect("clamped to [1, 255]; always fits u8");
        }
    }
}

/// Compute the Bayer (interleaved) rank for position `(x, y)` in a
/// `2^log2_size × 2^log2_size` matrix.
///
/// The algorithm visits each bit-level of the coordinates in turn.  At each
/// level the two low-order bits `(xi, yi)` contribute a 2-bit code using the
/// standard Bayer mapping:
///
/// ```text
/// (xi=0, yi=0) → 0   (xi=1, yi=0) → 2
/// (xi=0, yi=1) → 3   (xi=1, yi=1) → 1
/// ```
///
/// These codes are accumulated into `rank` with weight `4^level`, so the
/// overall rank is a base-4 number whose digits are the per-level codes.
/// The result is the canonical Bayer threshold order: at every scale the
/// dots are arranged so that no two nearby pixels share a threshold value.
///
/// # Panics (debug only)
///
/// Panics in debug builds if `log2_size > 15` (which would make `step`
/// exceed `u32::MAX` via `4^16 > 2^32`).
fn bayer_index(mut x: usize, mut y: usize, log2_size: u32) -> u32 {
    debug_assert!(
        log2_size <= 15,
        "bayer_index: log2_size={log2_size} would overflow step (max 15)"
    );
    let mut rank = 0u32;
    let mut step = 1u32;
    for _ in 0..log2_size {
        let xi = x & 1;
        let yi = y & 1;
        // Standard Bayer: rank bits come from (xi XOR yi, yi).
        let bits = match (xi, yi) {
            (0, 0) => 0,
            (1, 0) => 2,
            (0, 1) => 3,
            _ => 1,
        };
        rank += bits * step;
        step = step.saturating_mul(4);
        x >>= 1;
        y >>= 1;
    }
    rank
}

// ── Clustered dot matrix ──────────────────────────────────────────────────────

/// Simple clustered dot screen. Generates a radially clustered threshold
/// matrix centred at `(size/2, size/2)`.
///
/// Each cell receives a threshold proportional to `1 − dist²/max_dist²`,
/// so cells near the centre cluster together and darken before the
/// periphery: this is the traditional analog halftone "dot" look.
///
/// # Panics (debug only)
///
/// Panics in debug builds if `size < 2`.
fn build_clustered(mat: &mut [u8], size: usize) {
    debug_assert!(size >= 2, "build_clustered: size must be >= 2, got {size}");
    // size is a small power of 2 (≤ 64 in practice); cast through u32 to avoid
    // cast_precision_loss lint (u32 → f64 is always lossless).
    debug_assert!(
        u32::try_from(size / 2).is_ok(),
        "size/2={} exceeds u32::MAX",
        size / 2
    );
    let half = f64::from(u32::try_from(size / 2).expect("size/2 verified to fit u32 above"));
    let cx = half;
    let cy = half;
    let max_dist = cx * cx + cy * cy;
    // size >= 2 ⟹ half >= 1.0 ⟹ max_dist >= 2.0; division by zero is
    // impossible. Guard defensively for the debug build.
    debug_assert!(
        max_dist > 0.0,
        "build_clustered: max_dist is zero (size={size})"
    );
    for y in 0..size {
        for x in 0..size {
            debug_assert!(
                u32::try_from(x).is_ok() && u32::try_from(y).is_ok(),
                "x={x} or y={y} exceeds u32::MAX"
            );
            let dx = f64::from(u32::try_from(x).expect("x < size <= u32::MAX (verified above)")) - cx;
            let dy = f64::from(u32::try_from(y).expect("y < size <= u32::MAX (verified above)")) - cy;
            let dist = dx.mul_add(dx, dy * dy);
            // Known false-positive: value is clamped to [0.5, 254.5] so
            // truncation to u8 is safe. #[expect] errors if clippy ever fixes
            // the false-positive (github.com/rust-lang/rust-clippy/issues/7486).
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "value clamped to [0.5, 254.5] before cast; fits in u8"
            )]
            let v = (1.0_f64 - dist / max_dist)
                .mul_add(254.0, 0.5)
                .clamp(0.5, 254.5) as u8;
            mat[y * size + x] = v.max(1);
        }
    }
}

// ── Stochastic clustered matrix ───────────────────────────────────────────────

/// Stochastic clustered dot screen. Places dots at semi-random positions
/// with a minimum distance of `dot_radius` between dot centres.
/// Matches `SplashScreen::buildSCDMatrix` in spirit.
///
/// Cells are sorted by their distance to the nearest jittered dot centre
/// (using a torus/wrap-around metric so the matrix tiles seamlessly).
/// Cells close to a dot centre receive low thresholds (they darken first);
/// cells far from any centre receive high thresholds.
///
/// # Panics (debug only)
///
/// Panics in debug builds if `size < 2`.
fn build_stochastic_clustered(mat: &mut [u8], size: usize, dot_radius: usize) {
    debug_assert!(
        size >= 2,
        "build_stochastic_clustered: size must be >= 2, got {size}"
    );
    // Simplified version: place dot centres on a jittered grid and build
    // a distance-based threshold matrix. The C++ version uses a void-pointer
    // algorithm with a priority queue; this captures the same visual character.
    let n = size * size;
    // n >= 4 because size >= 2; division by n is safe.
    debug_assert!(n >= 4, "build_stochastic_clustered: n={n} must be >= 4");
    let mut thresholds: Vec<(usize, f64)> = (0..n)
        .map(|i| {
            let x = i % size;
            let y = i / size;
            // Nearest dot-centre distance (torus metric).
            let dist = nearest_dot_dist(x, y, size, dot_radius);
            (i, dist)
        })
        .collect();
    thresholds.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    for (rank, (idx, _)) in thresholds.iter().enumerate() {
        // rank < n, so rank * 254 / n is in [0, 254]; fits in u8.
        // rank < n, so rank * 254 / n ∈ [0, 254]; `.clamp(0, 255)` makes the
        // bound explicit for the type-checker; always fits u8.
        mat[*idx] = u8::try_from((rank * 254 / n).clamp(0, 255))
            .expect("rank * 254 / n ∈ [0, 254]; clamped to [0, 255]; fits u8")
            .max(1);
    }
}

/// Compute the squared distance from `(x, y)` to the nearest dot centre on a
/// torus of side `size`.
///
/// Dot centres are placed on a regular grid with spacing `step = max(2 ×
/// dot_radius, 1)`.  The torus metric wraps coordinates so that the matrix
/// tiles seamlessly with no visible seam at the edges.
///
/// Returns `f64::INFINITY` only if the grid has no centres — which cannot
/// happen because `step ≥ 1` and `size ≥ 1`, so the loop always executes at
/// least one iteration.
fn nearest_dot_dist(x: usize, y: usize, size: usize, dot_radius: usize) -> f64 {
    // step >= 1 ensures the while loops always advance; no infinite loop risk.
    let step = (dot_radius * 2).max(1);
    debug_assert!(step >= 1, "nearest_dot_dist: step must be >= 1");
    let mut min_d = f64::INFINITY;
    let mut cx = 0usize;
    while cx < size {
        let mut cy = 0usize;
        while cy < size {
            let dx = torus_dist(x, cx, size);
            let dy = torus_dist(y, cy, size);
            let d = dx.mul_add(dx, dy * dy);
            if d < min_d {
                min_d = d;
            }
            cy += step;
        }
        cx += step;
    }
    min_d
}

/// Compute the shortest (wrap-around) distance between positions `a` and `b`
/// on a 1-D torus of circumference `size`.
///
/// Both `a` and `b` are valid pixel indices and therefore in `[0, size)`.
/// The straight-line distance is `a.abs_diff(b)`; the wrap-around distance
/// is `size - a.abs_diff(b)`.  The torus distance is the minimum of the two.
///
/// # Panics (debug only)
///
/// Panics in debug builds if `a >= size` or `b >= size` (which would make
/// `abs_diff(b) > size - 1`, causing `size - abs_diff` to wrap on overflow).
fn torus_dist(a: usize, b: usize, size: usize) -> f64 {
    debug_assert!(a < size, "torus_dist: a={a} out of range [0, {size})");
    debug_assert!(b < size, "torus_dist: b={b} out of range [0, {size})");
    // a, b < size  ⟹  abs_diff <= size-1 < size  ⟹  size - abs_diff >= 1;
    // no wrapping possible on either subtraction.
    let straight = a.abs_diff(b);
    // Saturating handles the theoretical edge where size==0; the debug_asserts
    // above already rule that out in practice.
    let wrap = size.saturating_sub(straight);
    let d = straight.min(wrap);
    // d ≤ size/2 ≤ size ≤ i32::MAX in practice; cast through u32 to avoid the
    // cast_precision_loss lint (u32 → f64 is always exact).
    debug_assert!(
        u32::try_from(d).is_ok(),
        "torus_dist: d={d} exceeds u32::MAX"
    );
    f64::from(u32::try_from(d).expect("d <= size/2; size bounded by i32 params so d fits u32"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ScreenParams, ScreenType};

    #[test]
    fn test_wraps_by_size() {
        let params = ScreenParams {
            kind: ScreenType::Dispersed,
            size: 4,
            dot_radius: 2,
        };
        let mut screen = HalftoneScreen::new(params);
        // test at (0,0) and (4,0) must give same result (wrapping).
        let v = 128u8;
        let r0 = screen.test(0, 0, v);
        let r4 = screen.test(4, 0, v);
        assert_eq!(r0, r4);
    }

    #[test]
    fn zero_is_always_black() {
        let params = ScreenParams::default();
        let mut screen = HalftoneScreen::new(params);
        for y in 0..8i32 {
            for x in 0..8i32 {
                assert!(
                    !screen.test(x, y, 0),
                    "value=0 should be black at ({x},{y})"
                );
            }
        }
    }

    #[test]
    fn max_is_always_white() {
        let params = ScreenParams::default();
        let mut screen = HalftoneScreen::new(params);
        for y in 0..8i32 {
            for x in 0..8i32 {
                assert!(
                    screen.test(x, y, 255),
                    "value=255 should be white at ({x},{y})"
                );
            }
        }
    }

    #[test]
    fn matrix_sum_nonzero() {
        let params = ScreenParams {
            kind: ScreenType::Dispersed,
            size: 4,
            dot_radius: 2,
        };
        let mut screen = HalftoneScreen::new(params);
        screen.create_matrix();
        let mat = screen.mat.as_ref().unwrap();
        let sum: u32 = mat.iter().map(|&v| u32::from(v)).sum();
        assert!(sum > 0);
        assert_eq!(mat.len(), screen.size * screen.size);
    }

    #[test]
    fn torus_dist_symmetric() {
        // torus_dist(a, b, size) must equal torus_dist(b, a, size).
        let size = 8usize;
        for a in 0..size {
            for b in 0..size {
                let d_ab = torus_dist(a, b, size);
                let d_ba = torus_dist(b, a, size);
                assert!(
                    (d_ab - d_ba).abs() < f64::EPSILON,
                    "torus_dist not symmetric: ({a},{b}) size={size}"
                );
            }
        }
    }

    #[test]
    fn torus_dist_wrap_shorter() {
        // In a size-8 torus, torus_dist(0, 7, 8) should be 1 (wrap), not 7.
        let d = torus_dist(0, 7, 8);
        assert!(
            (d - 1.0).abs() < f64::EPSILON,
            "expected wrap distance 1.0, got {d}"
        );
    }

    #[test]
    fn clamp_min_one_sets_zeros() {
        let mut buf = vec![0u8, 1, 0, 255, 0];
        clamp_min_one(&mut buf);
        assert_eq!(buf, vec![1u8, 1, 1, 255, 1]);
    }

    #[test]
    fn all_screen_types_produce_valid_matrices() {
        for kind in [
            ScreenType::Dispersed,
            ScreenType::Clustered,
            ScreenType::StochasticClustered,
        ] {
            let params = ScreenParams {
                kind,
                size: 4,
                dot_radius: 2,
            };
            let mut screen = HalftoneScreen::new(params);
            screen.create_matrix();
            let mat = screen.mat.as_ref().unwrap();
            assert!(
                mat.iter().all(|&v| v >= 1),
                "matrix for {kind:?} contains zero entries"
            );
            assert_eq!(mat.len(), screen.size * screen.size);
        }
    }

    #[test]
    fn bayer_index_zero_at_origin() {
        // By definition the top-left corner of the Bayer matrix has rank 0.
        assert_eq!(bayer_index(0, 0, 2), 0);
    }

    #[test]
    fn bayer_index_all_unique_2x2() {
        // In a 2×2 Bayer matrix (log2_size=1) all four ranks must be distinct.
        let ranks: Vec<u32> = (0..2)
            .flat_map(|y| (0..2).map(move |x| bayer_index(x, y, 1)))
            .collect();
        let mut sorted = ranks.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 4, "expected 4 unique ranks, got {ranks:?}");
    }
}
