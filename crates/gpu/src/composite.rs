//! CPU Porter-Duff compositing and soft-mask application.

/// CPU fallback for `composite_rgba8`.
pub fn composite_rgba8_cpu(src: &[u8], dst: &mut [u8]) {
    for (s, d) in src.chunks_exact(4).zip(dst.chunks_exact_mut(4)) {
        let a_src = u32::from(s[3]);
        if a_src == 0 {
            continue;
        }
        if a_src == 255 {
            d.copy_from_slice(s);
            continue;
        }
        let a_dst = u32::from(d[3]);
        let inv = 255 - a_src;
        let a_out = a_src + (a_dst * inv + 127) / 255;
        if a_out == 0 {
            continue;
        }
        for c in 0..3 {
            let blended =
                (u32::from(s[c]) * a_src + u32::from(d[c]) * a_dst * inv / 255 + a_out / 2) / a_out;
            d[c] = blended.min(255) as u8;
        }
        // a_out = a_src + (a_dst * inv + 127) / 255 ≤ 255 + 255 = 510, so min(255) is needed.
        d[3] = a_out.min(255) as u8;
    }
}

/// CPU fallback for `apply_soft_mask`.
pub fn apply_soft_mask_cpu(pixels: &mut [u8], mask: &[u8]) {
    for (p, &m) in pixels.chunks_exact_mut(4).zip(mask) {
        let a = u32::from(p[3]);
        let m = u32::from(m);
        // a*m is at most 255*255 = 65025; +127 = 65152 < u32::MAX; /255 ≤ 255: safe cast.
        #[expect(
            clippy::cast_possible_truncation,
            reason = "result ≤ 255, always fits u8"
        )]
        let scaled = ((a * m + 127) / 255) as u8;
        p[3] = scaled;
    }
}

#[cfg(test)]
mod tests {
    use super::{apply_soft_mask_cpu, composite_rgba8_cpu};

    #[test]
    fn composite_cpu_opaque_src() {
        let src = [200u8, 100, 50, 255];
        let mut dst = [10u8, 20, 30, 128];
        composite_rgba8_cpu(&src, &mut dst);
        assert_eq!(dst, [200, 100, 50, 255]);
    }

    #[test]
    fn composite_cpu_transparent_src() {
        let src = [200u8, 100, 50, 0];
        let mut dst = [10u8, 20, 30, 128];
        let expected = dst;
        composite_rgba8_cpu(&src, &mut dst);
        assert_eq!(dst, expected);
    }

    #[test]
    fn composite_cpu_half_alpha() {
        let src = [255u8, 255, 255, 128];
        let mut dst = [0u8, 0, 0, 255];
        composite_rgba8_cpu(&src, &mut dst);
        assert!(dst[0] >= 126 && dst[0] <= 130, "r={}", dst[0]);
        assert!(dst[1] >= 126 && dst[1] <= 130, "g={}", dst[1]);
        assert!(dst[2] >= 126 && dst[2] <= 130, "b={}", dst[2]);
        assert_eq!(dst[3], 255);
    }

    #[test]
    fn soft_mask_cpu_full() {
        let mut pixels = [100u8, 150, 200, 240];
        let mask = [255u8];
        apply_soft_mask_cpu(&mut pixels, &mask);
        assert_eq!(pixels[3], 240);
    }

    #[test]
    fn soft_mask_cpu_half() {
        let mut pixels = [100u8, 150, 200, 200];
        let mask = [128u8];
        apply_soft_mask_cpu(&mut pixels, &mask);
        assert_eq!(pixels[3], 100);
    }

    #[test]
    fn soft_mask_cpu_zero() {
        let mut pixels = [100u8, 150, 200, 255, 10, 20, 30, 128];
        let mask = [0u8, 0];
        apply_soft_mask_cpu(&mut pixels, &mask);
        assert_eq!(pixels[3], 0);
        assert_eq!(pixels[7], 0);
    }
}
