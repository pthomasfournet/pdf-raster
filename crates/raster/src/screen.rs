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
    ///
    /// # Panics
    ///
    /// This function builds the threshold matrix on first call via an internal
    /// `unwrap()` that is safe because `create_matrix` always sets `self.mat`.
    #[inline]
    pub fn test(&mut self, x: i32, y: i32, value: u8) -> bool {
        if self.mat.is_none() {
            self.create_matrix();
        }
        let mat = self.mat.as_ref().unwrap();
        // rem_euclid handles negative coordinates correctly: result is always
        // in [0, size), and size is a power of 2 so this is equivalent to
        // (x & size_m1) for non-negative x.
        let size_i32 = i32::try_from(self.size).unwrap_or(i32::MAX);
        let xx = usize::try_from(x.rem_euclid(size_i32)).unwrap_or(0);
        let yy = usize::try_from(y.rem_euclid(size_i32)).unwrap_or(0);
        value >= mat[(yy << self.log2_size) + xx]
    }

    // ── Matrix construction ───────────────────────────────────────────────────

    fn create_matrix(&mut self) {
        // Find smallest power-of-2 size ≥ params.size.
        let mut size = 2usize;
        let mut log2 = 1u32;
        while i32::try_from(size).unwrap_or(i32::MAX) < self.params.size {
            size *= 2;
            log2 += 1;
        }
        // For stochastic clustered: ensure size ≥ 2 × dot_radius.
        if self.params.kind == ScreenType::StochasticClustered {
            let min_size = usize::try_from(2 * self.params.dot_radius).unwrap_or(0);
            while size < min_size {
                size *= 2;
                log2 += 1;
            }
        }
        self.size = size;
        self.size_m1 = size - 1;
        self.log2_size = log2;

        let mut mat = vec![0u8; size * size];
        match self.params.kind {
            ScreenType::Dispersed => build_dispersed(&mut mat, size, log2),
            ScreenType::Clustered => build_clustered(&mut mat, size),
            ScreenType::StochasticClustered => {
                let dot_radius = usize::try_from(self.params.dot_radius).unwrap_or(0);
                build_stochastic_clustered(&mut mat, size, dot_radius);
            }
        }
        // Clamp all entries to ≥ 1 so that value=0 is always black.
        for v in &mut mat {
            if *v == 0 {
                *v = 1;
            }
        }
        self.mat = Some(mat);
    }
}

// ── Dispersed (Bayer) matrix ──────────────────────────────────────────────────

/// Recursive void-pointer dispersed dot (Bayer) matrix construction.
/// Matches `SplashScreen::buildDispersedMatrix` in `SplashScreen.cc`.
fn build_dispersed(mat: &mut [u8], size: usize, log2_size: u32) {
    // Fill using the recursive Bayer pattern.
    let total = u32::try_from(size * size).unwrap_or(u32::MAX);
    for y in 0..size {
        for x in 0..size {
            // Compute the Bayer index for (x, y) at this size.
            let v = bayer_index(x, y, log2_size);
            // Scale to [1, 255].
            mat[y * size + x] =
                u8::try_from(((v * 255 + total / 2) / total).clamp(1, 255)).unwrap_or(255);
        }
    }
}

/// Compute the Bayer (interleaved) rank for position (x, y) in a 2^n × 2^n matrix.
fn bayer_index(mut x: usize, mut y: usize, log2_size: u32) -> u32 {
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
        step *= 4;
        x >>= 1;
        y >>= 1;
    }
    rank
}

// ── Clustered dot matrix ──────────────────────────────────────────────────────

/// Simple clustered dot screen. Generates a radially clustered threshold matrix.
fn build_clustered(mat: &mut [u8], size: usize) {
    // size is a small power of 2 (≤ 64 in practice); cast through u32 to avoid
    // cast_precision_loss lint (u32 → f64 is always lossless).
    let half = f64::from(u32::try_from(size / 2).unwrap_or(u32::MAX));
    let cx = half;
    let cy = half;
    let max_dist = cx * cx + cy * cy;
    for y in 0..size {
        for x in 0..size {
            let dx = f64::from(u32::try_from(x).unwrap_or(u32::MAX)) - cx;
            let dy = f64::from(u32::try_from(y).unwrap_or(u32::MAX)) - cy;
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
fn build_stochastic_clustered(mat: &mut [u8], size: usize, dot_radius: usize) {
    // Simplified version: place dot centres on a jittered grid and build
    // a distance-based threshold matrix. The C++ version uses a void-pointer
    // algorithm with a priority queue; this captures the same visual character.
    let n = size * size;
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
        mat[*idx] = u8::try_from((rank * 254 / n).clamp(0, 255))
            .unwrap_or(255)
            .max(1);
    }
}

fn nearest_dot_dist(x: usize, y: usize, size: usize, dot_radius: usize) -> f64 {
    let step = (dot_radius * 2).max(1);
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

fn torus_dist(a: usize, b: usize, size: usize) -> f64 {
    // a and b are < size (both are valid pixel indices), so the difference
    // fits in u32 (size ≤ 64 in practice, always < 2^31).
    let d = a.abs_diff(b).min(size - a.abs_diff(b));
    // d ≤ size ≤ 64; cast through u32 to avoid cast_precision_loss lint.
    f64::from(u32::try_from(d).unwrap_or(u32::MAX))
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
}
