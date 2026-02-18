use std::path::Path;

use anyhow::Context;
use image::{ImageBuffer, Rgb, RgbImage};

pub fn save_preview_montage(
    save_path: &Path,
    input_chw: &[f32],
    target_map: &[f32],
    pred_map: &[f32],
    h: usize,
    w: usize,
) -> anyhow::Result<()> {
    let pixels = h * w;
    if input_chw.len() != 6 * pixels {
        anyhow::bail!("Expected input tensor with 6 channels for preview");
    }
    if target_map.len() != pixels || pred_map.len() != pixels {
        anyhow::bail!("Target/prediction map has invalid shape for preview");
    }

    let left = chw_to_rgb(&input_chw[0..(3 * pixels)], h, w);
    let right = chw_to_rgb(&input_chw[(3 * pixels)..(6 * pixels)], h, w);
    let target = normalize_map_to_rgb(target_map, h, w);
    let pred = normalize_map_to_rgb(pred_map, h, w);

    let mut montage: RgbImage = ImageBuffer::new((w * 4) as u32, h as u32);
    blit(&mut montage, &left, 0, 0);
    blit(&mut montage, &right, w, 0);
    blit(&mut montage, &target, w * 2, 0);
    blit(&mut montage, &pred, w * 3, 0);

    if let Some(parent) = save_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create preview dir: {}", parent.display()))?;
    }
    montage
        .save(save_path)
        .with_context(|| format!("Failed to write preview image: {}", save_path.display()))?;
    Ok(())
}

fn chw_to_rgb(chw: &[f32], h: usize, w: usize) -> RgbImage {
    let pixels = h * w;
    let mut image: RgbImage = ImageBuffer::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let r = (chw[idx] * 255.0).clamp(0.0, 255.0) as u8;
            let g = (chw[pixels + idx] * 255.0).clamp(0.0, 255.0) as u8;
            let b = (chw[2 * pixels + idx] * 255.0).clamp(0.0, 255.0) as u8;
            image.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
        }
    }
    image
}

fn normalize_map_to_rgb(map: &[f32], h: usize, w: usize) -> RgbImage {
    let values: Vec<f32> = map
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    let (vmin, vmax) = if values.is_empty() {
        (0.0, 1.0)
    } else {
        let mut sorted = values;
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let low_idx = ((sorted.len() as f32) * 0.05).floor() as usize;
        let high_idx = ((sorted.len() as f32) * 0.95).floor() as usize;
        (
            sorted[low_idx.min(sorted.len() - 1)],
            sorted[high_idx.min(sorted.len() - 1)],
        )
    };

    let scale = (vmax - vmin).max(1e-6);
    let mut image: RgbImage = ImageBuffer::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let value = map[idx];
            let norm = if value.is_finite() {
                ((value - vmin) / scale).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let gray = (norm * 255.0).round() as u8;
            image.put_pixel(x as u32, y as u32, Rgb([gray, gray, gray]));
        }
    }
    image
}

fn blit(dst: &mut RgbImage, src: &RgbImage, x_offset: usize, y_offset: usize) {
    for (x, y, pixel) in src.enumerate_pixels() {
        dst.put_pixel(x + x_offset as u32, y + y_offset as u32, *pixel);
    }
}
