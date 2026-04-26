//! Output filename generation, matching pdftoppm naming conventions.

use crate::args::{Args, OutputFormat};

/// Compute the output filename for a given page.
///
/// Examples (prefix = "out", sep = '-', format = Ppm):
/// - page 1, total 10 → `"out-1.ppm"`
/// - page 1, total 100, `force_num_digits` = Some(3) → `"out-001.ppm"`
pub fn output_path(args: &Args, page_num: i32, total_pages: i32, format: OutputFormat) -> String {
    let digits = args
        .force_num_digits
        .unwrap_or_else(|| digit_width(total_pages));

    let sep = args.separator;
    let ext = format.extension();
    format!("{}{sep}{page_num:0digits$}.{ext}", args.output_prefix)
}

/// Minimum number of digits needed to represent `n`.
const fn digit_width(n: i32) -> usize {
    if n <= 0 {
        return 1;
    }
    n.ilog10() as usize + 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::args::{AaFlag, Args, ThinLineMode};

    fn make_args(prefix: &str, sep: char, force: Option<usize>) -> Args {
        Args {
            input: "in.pdf".into(),
            output_prefix: prefix.into(),
            first_page: 1,
            last_page: None,
            odd_only: false,
            even_only: false,
            single_file: false,
            resolution: None,
            resolution_x: None,
            resolution_y: None,
            scale_to: None,
            scale_to_x: None,
            scale_to_y: None,
            crop_x: None,
            crop_y: None,
            crop_w: None,
            crop_h: None,
            use_cropbox: false,
            png: false,
            jpeg: false,
            jpegcmyk: false,
            tiff: false,
            gray: false,
            mono: false,
            jpeg_quality: 75,
            antialias: AaFlag::Yes,
            vector_antialias: AaFlag::Yes,
            hide_annotations: false,
            overprint: false,
            thin_line_mode: ThinLineMode::Default,
            owner_password: None,
            user_password: None,
            num_threads: 0,
            separator: sep,
            force_num_digits: force,
            progress: false,
        }
    }

    #[test]
    fn single_digit_total() {
        let a = make_args("out", '-', None);
        assert_eq!(output_path(&a, 1, 9, OutputFormat::Ppm), "out-1.ppm");
    }

    #[test]
    fn two_digit_total_pads() {
        let a = make_args("out", '-', None);
        assert_eq!(output_path(&a, 1, 10, OutputFormat::Ppm), "out-01.ppm");
        assert_eq!(output_path(&a, 10, 10, OutputFormat::Ppm), "out-10.ppm");
    }

    #[test]
    fn force_num_overrides_auto() {
        let a = make_args("doc", '-', Some(3));
        assert_eq!(output_path(&a, 1, 5, OutputFormat::Png), "doc-001.png");
        assert_eq!(output_path(&a, 42, 5, OutputFormat::Png), "doc-042.png");
    }

    #[test]
    fn custom_separator() {
        let a = make_args("file", '_', None);
        assert_eq!(output_path(&a, 2, 9, OutputFormat::Jpeg), "file_2.jpg");
    }

    #[test]
    fn three_digit_total() {
        let a = make_args("p", '-', None);
        assert_eq!(output_path(&a, 99, 100, OutputFormat::Tiff), "p-099.tif");
    }
}
