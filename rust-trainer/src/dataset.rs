use std::fs;
use std::fs::File;
use std::path::{Component, Path, PathBuf};

use anyhow::Context;
use blake2::{Blake2s256, Digest};
use ndarray::{Array2, Array3};
use ndarray_npy::{NpzReader, NpzWriter};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

#[derive(Clone, Debug)]
pub struct StereoSample {
    pub left_rgb_path: PathBuf,
    pub right_rgb_path: PathBuf,
    pub disparity_path: PathBuf,
}

#[derive(Clone, Debug)]
pub struct AugmentConfig {
    pub brightness_jitter: f32,
    pub contrast_jitter: f32,
    pub saturation_jitter: f32,
    pub hue_jitter: f32,
    pub gamma_jitter: f32,
    pub noise_std_max: f32,
    pub blur_prob: f32,
    pub blur_sigma_max: f32,
    pub blur_kernel_size: usize,
}

#[derive(Clone, Debug)]
pub struct DatasetConfig {
    pub image_height: usize,
    pub image_width: usize,
    pub augment: bool,
    pub augment_cfg: AugmentConfig,
    pub cache_root: Option<PathBuf>,
    pub require_cache: bool,
}

#[derive(Debug)]
pub struct SampleTensors {
    pub input: Vec<f32>,
    pub target: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct FoundationStereoDataset {
    pub samples: Vec<StereoSample>,
    pub cfg: DatasetConfig,
}

impl FoundationStereoDataset {
    pub fn new(samples: Vec<StereoSample>, cfg: DatasetConfig) -> anyhow::Result<Self> {
        if samples.is_empty() {
            anyhow::bail!("No samples were provided");
        }
        Ok(Self { samples, cfg })
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn load_item(&self, index: usize, rng: &mut StdRng) -> anyhow::Result<SampleTensors> {
        let sample = self
            .samples
            .get(index)
            .with_context(|| format!("Sample index out of bounds: {index}"))?;

        let mut left = None;
        let mut right = None;
        let mut target = None;
        let mut loaded_from_cache = false;
        let mut cache_file = None;

        if let Some(cache_root) = &self.cfg.cache_root {
            let path = cache_root.join(sample_cache_relpath(sample));
            if path.exists() {
                match load_cached_sample(&path, self.cfg.image_height, self.cfg.image_width)? {
                    Some((left_loaded, right_loaded, target_loaded)) => {
                        left = Some(left_loaded);
                        right = Some(right_loaded);
                        target = Some(target_loaded);
                        loaded_from_cache = true;
                    }
                    None if self.cfg.require_cache => {
                        anyhow::bail!(
                            "Cache entry is invalid or shape-mismatched for sample: {}",
                            path.display()
                        );
                    }
                    None => {}
                }
            } else if self.cfg.require_cache {
                anyhow::bail!("Required cache entry not found: {}", path.display());
            }
            cache_file = Some(path);
        }

        if left.is_none() || right.is_none() || target.is_none() {
            let left_loaded = load_rgb_as_chw(
                &sample.left_rgb_path,
                self.cfg.image_height,
                self.cfg.image_width,
            )?;
            let right_loaded = load_rgb_as_chw(
                &sample.right_rgb_path,
                self.cfg.image_height,
                self.cfg.image_width,
            )?;
            let target_loaded = load_disparity_as_chw(
                &sample.disparity_path,
                self.cfg.image_height,
                self.cfg.image_width,
            )?;
            left = Some(left_loaded);
            right = Some(right_loaded);
            target = Some(target_loaded);
        }

        let mut left = left.expect("left checked");
        let mut right = right.expect("right checked");
        let target = target.expect("target checked");

        if let Some(cache_file) = &cache_file {
            if !self.cfg.require_cache && !loaded_from_cache {
                save_cached_sample(
                    cache_file,
                    &left,
                    &right,
                    &target,
                    self.cfg.image_height,
                    self.cfg.image_width,
                )?;
            }
        }

        if self.cfg.augment {
            augment_rgb(
                &mut left,
                self.cfg.image_height,
                self.cfg.image_width,
                &self.cfg.augment_cfg,
                rng,
            )?;
            augment_rgb(
                &mut right,
                self.cfg.image_height,
                self.cfg.image_width,
                &self.cfg.augment_cfg,
                rng,
            )?;
        }

        let mut input = Vec::with_capacity(6 * self.cfg.image_height * self.cfg.image_width);
        input.extend_from_slice(&left);
        input.extend_from_slice(&right);

        Ok(SampleTensors { input, target })
    }

    pub fn build_cache(&self, overwrite: bool) -> anyhow::Result<(usize, usize)> {
        let cache_root = self
            .cfg
            .cache_root
            .as_ref()
            .context("Cache root must be set to build cache")?;

        fs::create_dir_all(cache_root)
            .with_context(|| format!("Failed to create cache root: {}", cache_root.display()))?;

        let mut written = 0usize;
        let mut skipped = 0usize;
        for index in 0..self.samples.len() {
            let sample = &self.samples[index];
            let cache_file = cache_root.join(sample_cache_relpath(sample));
            if cache_file.exists() && !overwrite {
                skipped += 1;
                continue;
            }
            let left = load_rgb_as_chw(
                &sample.left_rgb_path,
                self.cfg.image_height,
                self.cfg.image_width,
            )?;
            let right = load_rgb_as_chw(
                &sample.right_rgb_path,
                self.cfg.image_height,
                self.cfg.image_width,
            )?;
            let target = load_disparity_as_chw(
                &sample.disparity_path,
                self.cfg.image_height,
                self.cfg.image_width,
            )?;
            save_cached_sample(
                &cache_file,
                &left,
                &right,
                &target,
                self.cfg.image_height,
                self.cfg.image_width,
            )?;
            written += 1;
        }

        Ok((written, skipped))
    }
}

pub fn discover_samples(dataset_root: &Path) -> anyhow::Result<Vec<StereoSample>> {
    if !dataset_root.exists() {
        anyhow::bail!("Dataset root does not exist: {}", dataset_root.display());
    }

    let mut scene_dirs: Vec<PathBuf> = fs::read_dir(dataset_root)
        .with_context(|| format!("Failed to read dataset root: {}", dataset_root.display()))?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .collect();
    scene_dirs.sort();

    let mut samples = Vec::new();
    for scene_dir in scene_dirs {
        let left_rgb_dir = scene_dir.join("dataset/data/left/rgb");
        let right_rgb_dir = scene_dir.join("dataset/data/right/rgb");
        let disparity_dir = scene_dir.join("dataset/data/left/disparity");

        if !(left_rgb_dir.exists() && right_rgb_dir.exists() && disparity_dir.exists()) {
            continue;
        }

        let mut disparity_paths: Vec<PathBuf> = fs::read_dir(&disparity_dir)
            .with_context(|| format!("Failed to read disparity dir: {}", disparity_dir.display()))?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| {
                path.extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| ext.eq_ignore_ascii_case("png"))
                    .unwrap_or(false)
            })
            .collect();
        disparity_paths.sort();

        for disparity_path in disparity_paths {
            let Some(stem) = disparity_path.file_stem().and_then(|stem| stem.to_str()) else {
                continue;
            };

            let left_path = resolve_frame_path(&left_rgb_dir, stem);
            let right_path = resolve_frame_path(&right_rgb_dir, stem);
            if let (Some(left_rgb_path), Some(right_rgb_path)) = (left_path, right_path) {
                samples.push(StereoSample {
                    left_rgb_path,
                    right_rgb_path,
                    disparity_path,
                });
            }
        }
    }

    Ok(samples)
}

pub fn split_samples(
    mut samples: Vec<StereoSample>,
    val_fraction: f32,
    seed: u64,
) -> anyhow::Result<(Vec<StereoSample>, Vec<StereoSample>)> {
    if !(0.0..1.0).contains(&val_fraction) {
        anyhow::bail!("--val-fraction must be in [0, 1), got {val_fraction}");
    }

    let mut rng = StdRng::seed_from_u64(seed);
    samples.shuffle(&mut rng);

    if val_fraction == 0.0 {
        return Ok((samples, Vec::new()));
    }

    let len = samples.len();
    let mut val_count = ((len as f32) * val_fraction).floor() as usize;
    val_count = val_count.max(1).min(len);
    if val_count >= len {
        anyhow::bail!(
            "Validation set consumes all data. Reduce --val-fraction or provide more samples"
        );
    }

    let train_count = len - val_count;
    let val_samples = samples.split_off(train_count);
    Ok((samples, val_samples))
}

pub fn sample_cache_relpath(sample: &StereoSample) -> PathBuf {
    let left_parts: Vec<String> = sample
        .left_rgb_path
        .components()
        .filter_map(component_to_string)
        .collect();

    if let Some(dataset_idx) = left_parts.iter().position(|part| part == "dataset") {
        if dataset_idx > 0 {
            let scene_name = &left_parts[dataset_idx - 1];
            let stem = sample
                .disparity_path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .unwrap_or("sample");
            return PathBuf::from(scene_name).join(format!("{stem}.npz"));
        }
    }

    let source_key = format!(
        "{}|{}|{}",
        sample.left_rgb_path.to_string_lossy(),
        sample.right_rgb_path.to_string_lossy(),
        sample.disparity_path.to_string_lossy()
    );
    let mut hasher = Blake2s256::new();
    hasher.update(source_key.as_bytes());
    let digest = hasher.finalize();
    let hash = hex::encode(&digest[..8]);
    let stem = sample
        .disparity_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("sample");

    PathBuf::from("misc").join(format!("{stem}_{hash}.npz"))
}

fn resolve_frame_path(frame_dir: &Path, stem: &str) -> Option<PathBuf> {
    ["jpg", "jpeg", "png"]
        .iter()
        .map(|ext| frame_dir.join(format!("{stem}.{ext}")))
        .find(|candidate| candidate.exists())
}

fn component_to_string(component: Component<'_>) -> Option<String> {
    match component {
        Component::Normal(v) => Some(v.to_string_lossy().to_string()),
        _ => None,
    }
}

fn load_rgb_as_chw(path: &Path, target_h: usize, target_w: usize) -> anyhow::Result<Vec<f32>> {
    let rgb = image::open(path)
        .with_context(|| format!("Failed to open RGB image: {}", path.display()))?
        .to_rgb8();
    let (src_w_u32, src_h_u32) = rgb.dimensions();
    let src_w = src_w_u32 as usize;
    let src_h = src_h_u32 as usize;

    let mut channels = [
        vec![0f32; src_h * src_w],
        vec![0f32; src_h * src_w],
        vec![0f32; src_h * src_w],
    ];

    for (idx, pixel) in rgb.pixels().enumerate() {
        channels[0][idx] = f32::from(pixel[0]) / 255.0;
        channels[1][idx] = f32::from(pixel[1]) / 255.0;
        channels[2][idx] = f32::from(pixel[2]) / 255.0;
    }

    let r = resize_bilinear_channel(&channels[0], src_h, src_w, target_h, target_w);
    let g = resize_bilinear_channel(&channels[1], src_h, src_w, target_h, target_w);
    let b = resize_bilinear_channel(&channels[2], src_h, src_w, target_h, target_w);

    let pixels = target_h * target_w;
    let mut out = vec![0f32; 3 * pixels];
    out[0..pixels].copy_from_slice(&r);
    out[pixels..(2 * pixels)].copy_from_slice(&g);
    out[(2 * pixels)..(3 * pixels)].copy_from_slice(&b);
    Ok(out)
}

fn load_disparity_as_chw(
    path: &Path,
    target_h: usize,
    target_w: usize,
) -> anyhow::Result<Vec<f32>> {
    let rgb = image::open(path)
        .with_context(|| format!("Failed to open disparity image: {}", path.display()))?
        .to_rgb8();
    let (src_w_u32, src_h_u32) = rgb.dimensions();
    let src_w = src_w_u32 as usize;
    let src_h = src_h_u32 as usize;

    let mut src_disp = vec![0f32; src_h * src_w];
    for (idx, pixel) in rgb.pixels().enumerate() {
        let r = f32::from(pixel[0]);
        let g = f32::from(pixel[1]);
        let b = f32::from(pixel[2]);
        src_disp[idx] = (r * 255.0 * 255.0 + g * 255.0 + b) / 1000.0;
    }

    let mut resized = resize_bilinear_channel(&src_disp, src_h, src_w, target_h, target_w);
    let width_scale = target_w as f32 / src_w as f32;
    resized.iter_mut().for_each(|value| *value *= width_scale);
    Ok(resized)
}

fn resize_bilinear_channel(
    src: &[f32],
    src_h: usize,
    src_w: usize,
    dst_h: usize,
    dst_w: usize,
) -> Vec<f32> {
    if src_h == dst_h && src_w == dst_w {
        return src.to_vec();
    }

    let mut out = vec![0f32; dst_h * dst_w];
    let scale_y = src_h as f32 / dst_h as f32;
    let scale_x = src_w as f32 / dst_w as f32;

    for y in 0..dst_h {
        let in_y = ((y as f32 + 0.5) * scale_y - 0.5)
            .max(0.0)
            .min((src_h - 1) as f32);
        let y0 = in_y.floor() as usize;
        let y1 = (y0 + 1).min(src_h - 1);
        let wy = in_y - y0 as f32;

        for x in 0..dst_w {
            let in_x = ((x as f32 + 0.5) * scale_x - 0.5)
                .max(0.0)
                .min((src_w - 1) as f32);
            let x0 = in_x.floor() as usize;
            let x1 = (x0 + 1).min(src_w - 1);
            let wx = in_x - x0 as f32;

            let top = src[y0 * src_w + x0] * (1.0 - wx) + src[y0 * src_w + x1] * wx;
            let bottom = src[y1 * src_w + x0] * (1.0 - wx) + src[y1 * src_w + x1] * wx;
            out[y * dst_w + x] = top * (1.0 - wy) + bottom * wy;
        }
    }

    out
}

fn sample_factor(rng: &mut StdRng, jitter: f32) -> f32 {
    if jitter <= 0.0 {
        return 1.0;
    }
    let low = (1.0 - jitter).max(0.0);
    let high = 1.0 + jitter;
    rng.random_range(low..=high)
}

fn sample_hue_shift(rng: &mut StdRng, jitter: f32) -> f32 {
    if jitter <= 0.0 {
        return 0.0;
    }
    rng.random_range(-jitter..=jitter)
}

fn sample_gamma(rng: &mut StdRng, jitter: f32) -> f32 {
    if jitter <= 0.0 {
        return 1.0;
    }
    let low = (1.0 - jitter).max(0.1);
    let high = (1.0 + jitter).max(low);
    rng.random_range(low..=high)
}

fn sample_noise_std(rng: &mut StdRng, max_std: f32) -> f32 {
    if max_std <= 0.0 {
        return 0.0;
    }
    rng.random_range(0.0..=max_std)
}

fn should_apply_blur(rng: &mut StdRng, blur_prob: f32, blur_sigma_max: f32) -> bool {
    blur_prob > 0.0 && blur_sigma_max > 0.0 && rng.random::<f32>() < blur_prob
}

fn sample_blur_sigma(rng: &mut StdRng, blur_sigma_max: f32) -> f32 {
    let sigma_min = 0.1;
    let sigma_max = blur_sigma_max.max(sigma_min);
    rng.random_range(sigma_min..=sigma_max)
}

fn augment_rgb(
    image_chw: &mut [f32],
    h: usize,
    w: usize,
    cfg: &AugmentConfig,
    rng: &mut StdRng,
) -> anyhow::Result<()> {
    adjust_brightness(image_chw, sample_factor(rng, cfg.brightness_jitter));
    adjust_contrast(image_chw, sample_factor(rng, cfg.contrast_jitter));
    adjust_saturation(image_chw, h, w, sample_factor(rng, cfg.saturation_jitter));
    adjust_hue(image_chw, h, w, sample_hue_shift(rng, cfg.hue_jitter));
    adjust_gamma(image_chw, sample_gamma(rng, cfg.gamma_jitter));

    if should_apply_blur(rng, cfg.blur_prob, cfg.blur_sigma_max) {
        let sigma = sample_blur_sigma(rng, cfg.blur_sigma_max);
        gaussian_blur(image_chw, h, w, cfg.blur_kernel_size, sigma);
    }

    let noise_std = sample_noise_std(rng, cfg.noise_std_max);
    if noise_std > 0.0 {
        let normal = Normal::new(0.0, noise_std as f64)
            .context("Failed to create Gaussian noise distribution")?;
        image_chw
            .iter_mut()
            .for_each(|value| *value += normal.sample(rng) as f32);
    }

    image_chw
        .iter_mut()
        .for_each(|value| *value = value.clamp(0.0, 1.0));
    Ok(())
}

fn adjust_brightness(image: &mut [f32], factor: f32) {
    image.iter_mut().for_each(|value| *value *= factor);
}

fn adjust_contrast(image: &mut [f32], factor: f32) {
    let mean = image.iter().sum::<f32>() / image.len() as f32;
    image
        .iter_mut()
        .for_each(|value| *value = (*value - mean) * factor + mean);
}

fn adjust_saturation(image: &mut [f32], h: usize, w: usize, factor: f32) {
    let pixels = h * w;
    for idx in 0..pixels {
        let r = image[idx];
        let g = image[pixels + idx];
        let b = image[2 * pixels + idx];
        let gray = 0.299 * r + 0.587 * g + 0.114 * b;
        image[idx] = gray + factor * (r - gray);
        image[pixels + idx] = gray + factor * (g - gray);
        image[2 * pixels + idx] = gray + factor * (b - gray);
    }
}

fn adjust_hue(image: &mut [f32], h: usize, w: usize, hue_shift: f32) {
    if hue_shift == 0.0 {
        return;
    }

    let pixels = h * w;
    for idx in 0..pixels {
        let r = image[idx];
        let g = image[pixels + idx];
        let b = image[2 * pixels + idx];
        let (mut hh, ss, vv) = rgb_to_hsv(r, g, b);
        hh = (hh + hue_shift).rem_euclid(1.0);
        let (nr, ng, nb) = hsv_to_rgb(hh, ss, vv);
        image[idx] = nr;
        image[pixels + idx] = ng;
        image[2 * pixels + idx] = nb;
    }
}

fn adjust_gamma(image: &mut [f32], gamma: f32) {
    image
        .iter_mut()
        .for_each(|value| *value = value.max(0.0).powf(gamma));
}

fn gaussian_blur(image: &mut [f32], h: usize, w: usize, kernel_size: usize, sigma: f32) {
    let kernel = gaussian_kernel_1d(kernel_size, sigma);
    let half = (kernel_size / 2) as isize;
    let pixels = h * w;

    let mut temp = vec![0f32; image.len()];
    let mut out = vec![0f32; image.len()];

    for c in 0..3 {
        for y in 0..h {
            for x in 0..w {
                let mut acc = 0.0;
                for (k, weight) in kernel.iter().enumerate() {
                    let offset = k as isize - half;
                    let xx = (x as isize + offset).clamp(0, (w - 1) as isize) as usize;
                    let idx = c * pixels + y * w + xx;
                    acc += image[idx] * *weight;
                }
                temp[c * pixels + y * w + x] = acc;
            }
        }

        for y in 0..h {
            for x in 0..w {
                let mut acc = 0.0;
                for (k, weight) in kernel.iter().enumerate() {
                    let offset = k as isize - half;
                    let yy = (y as isize + offset).clamp(0, (h - 1) as isize) as usize;
                    let idx = c * pixels + yy * w + x;
                    acc += temp[idx] * *weight;
                }
                out[c * pixels + y * w + x] = acc;
            }
        }
    }

    image.copy_from_slice(&out);
}

fn gaussian_kernel_1d(kernel_size: usize, sigma: f32) -> Vec<f32> {
    let half = (kernel_size / 2) as isize;
    let mut kernel = Vec::with_capacity(kernel_size);
    let mut sum = 0.0;

    for i in -half..=half {
        let x = i as f32;
        let value = (-x * x / (2.0 * sigma * sigma)).exp();
        kernel.push(value);
        sum += value;
    }

    if sum > 0.0 {
        kernel.iter_mut().for_each(|value| *value /= sum);
    }
    kernel
}

fn rgb_to_hsv(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max = r.max(g.max(b));
    let min = r.min(g.min(b));
    let delta = max - min;

    let h = if delta == 0.0 {
        0.0
    } else if max == r {
        ((g - b) / delta).rem_euclid(6.0) / 6.0
    } else if max == g {
        ((b - r) / delta + 2.0) / 6.0
    } else {
        ((r - g) / delta + 4.0) / 6.0
    };

    let s = if max == 0.0 { 0.0 } else { delta / max };
    (h, s, max)
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    if s == 0.0 {
        return (v, v, v);
    }

    let hh = h * 6.0;
    let i = hh.floor() as i32;
    let f = hh - i as f32;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));

    match i.rem_euclid(6) {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    }
}

fn load_cached_sample(
    cache_file: &Path,
    target_h: usize,
    target_w: usize,
) -> anyhow::Result<Option<(Vec<f32>, Vec<f32>, Vec<f32>)>> {
    let file = match File::open(cache_file) {
        Ok(file) => file,
        Err(_) => return Ok(None),
    };
    let mut npz = match NpzReader::new(file) {
        Ok(npz) => npz,
        Err(_) => return Ok(None),
    };

    let left: Array3<u8> = match npz.by_name("left.npy") {
        Ok(arr) => arr,
        Err(_) => match npz.by_name("left") {
            Ok(arr) => arr,
            Err(_) => return Ok(None),
        },
    };
    let right: Array3<u8> = match npz.by_name("right.npy") {
        Ok(arr) => arr,
        Err(_) => match npz.by_name("right") {
            Ok(arr) => arr,
            Err(_) => return Ok(None),
        },
    };

    let disparity_f32: Vec<f32> = if let Ok(disparity) =
        npz.by_name::<ndarray::OwnedRepr<f32>, ndarray::Ix2>("disparity.npy")
    {
        disparity.iter().copied().collect()
    } else if let Ok(disparity) = npz.by_name::<ndarray::OwnedRepr<f32>, ndarray::Ix2>("disparity")
    {
        disparity.iter().copied().collect()
    } else {
        return Ok(None);
    };

    let left_shape = left.raw_dim();
    let right_shape = right.raw_dim();
    if left_shape[0] != target_h
        || left_shape[1] != target_w
        || left_shape[2] != 3
        || right_shape[0] != target_h
        || right_shape[1] != target_w
        || right_shape[2] != 3
        || disparity_f32.len() != target_h * target_w
    {
        return Ok(None);
    }

    let left_chw = hwc_u8_to_chw_f32(
        left.as_slice().context("Non-contiguous left array")?,
        target_h,
        target_w,
    );
    let right_chw = hwc_u8_to_chw_f32(
        right.as_slice().context("Non-contiguous right array")?,
        target_h,
        target_w,
    );

    Ok(Some((left_chw, right_chw, disparity_f32)))
}

fn save_cached_sample(
    cache_file: &Path,
    left_chw: &[f32],
    right_chw: &[f32],
    target_chw: &[f32],
    h: usize,
    w: usize,
) -> anyhow::Result<()> {
    if let Some(parent) = cache_file.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create cache directory: {}", parent.display()))?;
    }

    let left_u8 = chw_f32_to_hwc_u8(left_chw, h, w);
    let right_u8 = chw_f32_to_hwc_u8(right_chw, h, w);
    let left_arr =
        Array3::from_shape_vec((h, w, 3), left_u8).context("Failed to shape left cache array")?;
    let right_arr =
        Array3::from_shape_vec((h, w, 3), right_u8).context("Failed to shape right cache array")?;
    let target_arr = Array2::from_shape_vec((h, w), target_chw.to_vec())
        .context("Failed to shape disparity cache array")?;

    let file = File::create(cache_file)
        .with_context(|| format!("Failed to create cache file: {}", cache_file.display()))?;
    let mut npz = NpzWriter::new(file);
    npz.add_array("left", &left_arr)
        .context("Failed writing 'left' to cache")?;
    npz.add_array("right", &right_arr)
        .context("Failed writing 'right' to cache")?;
    npz.add_array("disparity", &target_arr)
        .context("Failed writing 'disparity' to cache")?;
    npz.finish().context("Failed finalizing cache npz")?;
    Ok(())
}

fn hwc_u8_to_chw_f32(hwc: &[u8], h: usize, w: usize) -> Vec<f32> {
    let pixels = h * w;
    let mut out = vec![0f32; 3 * pixels];
    for y in 0..h {
        for x in 0..w {
            let base_hwc = (y * w + x) * 3;
            let base_chw = y * w + x;
            out[base_chw] = f32::from(hwc[base_hwc]) / 255.0;
            out[pixels + base_chw] = f32::from(hwc[base_hwc + 1]) / 255.0;
            out[2 * pixels + base_chw] = f32::from(hwc[base_hwc + 2]) / 255.0;
        }
    }
    out
}

fn chw_f32_to_hwc_u8(chw: &[f32], h: usize, w: usize) -> Vec<u8> {
    let pixels = h * w;
    let mut out = vec![0u8; h * w * 3];
    for y in 0..h {
        for x in 0..w {
            let base_hwc = (y * w + x) * 3;
            let base_chw = y * w + x;
            let r = (chw[base_chw] * 255.0).clamp(0.0, 255.0) as u8;
            let g = (chw[pixels + base_chw] * 255.0).clamp(0.0, 255.0) as u8;
            let b = (chw[2 * pixels + base_chw] * 255.0).clamp(0.0, 255.0) as u8;
            out[base_hwc] = r;
            out[base_hwc + 1] = g;
            out[base_hwc + 2] = b;
        }
    }
    out
}
