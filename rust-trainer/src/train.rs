use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Context;
use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use chrono::Utc;
use indicatif::{ProgressBar, ProgressStyle};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::Serialize;

use crate::config::TrainConfig;
use crate::dataset::{
    discover_samples, split_samples, AugmentConfig, DatasetConfig, FoundationStereoDataset,
    StereoSample,
};
use crate::model::StereoUNet;
use crate::preview::save_preview_montage;

#[derive(Debug, Clone, Serialize)]
pub struct EpochMetrics {
    pub loss: f64,
    pub nll: f64,
    pub mae: f64,
    pub rmse: f64,
    pub sigma: f64,
}

#[derive(Debug, Clone, Serialize)]
struct EpochSummary {
    epoch: usize,
    train: EpochMetrics,
    val: EpochMetrics,
    epoch_seconds: f64,
}

pub fn run_training(cfg: &TrainConfig) -> anyhow::Result<()> {
    cfg.validate()?;

    let dataset_root = Path::new(&cfg.dataset_root).expand_home();
    let mut all_samples = discover_samples(&dataset_root)?;
    if cfg.max_samples > 0 {
        all_samples.truncate(cfg.max_samples);
    }

    if cfg.build_cache_only {
        build_cache_only(cfg, &all_samples)?;
        return Ok(());
    }

    if all_samples.len() < 2 {
        anyhow::bail!("Need at least two samples to create train/validation splits");
    }

    let (train_samples, val_samples) = split_samples(all_samples, cfg.val_fraction, cfg.seed)?;
    println!(
        "Discovered {} samples: train={}, val={}",
        train_samples.len() + val_samples.len(),
        train_samples.len(),
        val_samples.len()
    );

    if cfg.num_workers > 0 {
        println!(
            "Note: --num-workers={} is accepted for CLI parity; data loading is currently synchronous in this Rust pipeline.",
            cfg.num_workers
        );
    }

    let augment_cfg = AugmentConfig {
        brightness_jitter: cfg.brightness_jitter,
        contrast_jitter: cfg.contrast_jitter,
        saturation_jitter: cfg.saturation_jitter,
        hue_jitter: cfg.hue_jitter,
        gamma_jitter: cfg.gamma_jitter,
        noise_std_max: cfg.noise_std_max,
        blur_prob: cfg.blur_prob,
        blur_sigma_max: cfg.blur_sigma_max,
        blur_kernel_size: cfg.blur_kernel_size,
    };

    let train_dataset = FoundationStereoDataset::new(
        train_samples.clone(),
        DatasetConfig {
            image_height: cfg.height,
            image_width: cfg.width,
            augment: cfg.augment_enabled(),
            augment_cfg: augment_cfg.clone(),
            cache_root: cfg
                .cache_root
                .as_ref()
                .map(|path| PathBuf::from(path).expand_home()),
            require_cache: cfg.require_cache,
        },
    )?;

    let val_dataset = FoundationStereoDataset::new(
        if val_samples.is_empty() {
            train_samples.clone()
        } else {
            val_samples.clone()
        },
        DatasetConfig {
            image_height: cfg.height,
            image_width: cfg.width,
            augment: false,
            augment_cfg: augment_cfg.clone(),
            cache_root: cfg
                .cache_root
                .as_ref()
                .map(|path| PathBuf::from(path).expand_home()),
            require_cache: cfg.require_cache,
        },
    )?;

    let preview_source: Vec<StereoSample> = if val_samples.is_empty() {
        train_samples.clone()
    } else {
        val_samples.clone()
    };
    let preview_count = cfg.preview_samples.min(preview_source.len());
    let preview_dataset = if preview_count > 0 {
        Some(FoundationStereoDataset::new(
            preview_source.into_iter().take(preview_count).collect(),
            DatasetConfig {
                image_height: cfg.height,
                image_width: cfg.width,
                augment: false,
                augment_cfg,
                cache_root: cfg
                    .cache_root
                    .as_ref()
                    .map(|path| PathBuf::from(path).expand_home()),
                require_cache: cfg.require_cache,
            },
        )?)
    } else {
        None
    };

    let device = resolve_device(&cfg.device)?;
    println!("Using device: {device:?}");

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = StereoUNet::new(6, 1, 32, vb)?;

    let adamw_params = ParamsAdamW {
        lr: cfg.lr,
        weight_decay: cfg.weight_decay,
        ..Default::default()
    };
    let mut optimizer = AdamW::new(varmap.all_vars(), adamw_params)?;

    let run_id = cfg
        .run_name
        .clone()
        .unwrap_or_else(|| Utc::now().format("%Y%m%d-%H%M%S").to_string());
    let run_dir = PathBuf::from(&cfg.output_dir).expand_home().join(&run_id);
    let checkpoints_dir = run_dir.join("checkpoints");
    let previews_dir = run_dir.join("previews");
    fs::create_dir_all(&checkpoints_dir).with_context(|| {
        format!(
            "Failed creating checkpoint directory: {}",
            checkpoints_dir.display()
        )
    })?;
    fs::create_dir_all(&previews_dir).with_context(|| {
        format!(
            "Failed creating preview directory: {}",
            previews_dir.display()
        )
    })?;

    let config_path = run_dir.join("config.json");
    fs::write(&config_path, serde_json::to_string_pretty(cfg)?)
        .with_context(|| format!("Failed writing config: {}", config_path.display()))?;

    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let mut best_val_mae = f64::INFINITY;
    let mut best_epoch = 0usize;
    let mut epoch_summaries = Vec::new();
    let mut global_step = 0usize;

    for epoch in 1..=cfg.epochs {
        let epoch_started = Instant::now();

        let mut train_indices: Vec<usize> = (0..train_dataset.len()).collect();
        train_indices.shuffle(&mut rng);

        let train_metrics = run_epoch(
            &model,
            &train_dataset,
            &train_indices,
            cfg,
            &device,
            Some(&mut optimizer),
            &mut global_step,
            &mut rng,
            true,
        )?;

        let val_indices: Vec<usize> = (0..val_dataset.len()).collect();
        let val_metrics = run_epoch(
            &model,
            &val_dataset,
            &val_indices,
            cfg,
            &device,
            None,
            &mut global_step,
            &mut rng,
            false,
        )?;

        let epoch_seconds = epoch_started.elapsed().as_secs_f64();
        let summary = EpochSummary {
            epoch,
            train: train_metrics.clone(),
            val: val_metrics.clone(),
            epoch_seconds,
        };
        epoch_summaries.push(summary.clone());

        println!(
            "Epoch {epoch}/{}: train_mae={:.4}, val_mae={:.4}, train_rmse={:.4}, val_rmse={:.4} ({:.1}s)",
            cfg.epochs,
            train_metrics.mae,
            val_metrics.mae,
            train_metrics.rmse,
            val_metrics.rmse,
            epoch_seconds
        );

        if let Some(preview_dataset) = &preview_dataset {
            write_epoch_previews(&model, preview_dataset, &device, &previews_dir, epoch)?;
        }

        let last_ckpt = checkpoints_dir.join("last.safetensors");
        varmap
            .save(&last_ckpt)
            .with_context(|| format!("Failed writing checkpoint: {}", last_ckpt.display()))?;
        let last_meta = checkpoints_dir.join("last.json");
        fs::write(&last_meta, serde_json::to_string_pretty(&summary)?).with_context(|| {
            format!(
                "Failed writing checkpoint metadata: {}",
                last_meta.display()
            )
        })?;

        if val_metrics.mae < best_val_mae {
            best_val_mae = val_metrics.mae;
            best_epoch = epoch;
            let best_ckpt = checkpoints_dir.join("best.safetensors");
            varmap.save(&best_ckpt).with_context(|| {
                format!("Failed writing best checkpoint: {}", best_ckpt.display())
            })?;
            let best_meta = checkpoints_dir.join("best.json");
            fs::write(&best_meta, serde_json::to_string_pretty(&summary)?).with_context(|| {
                format!(
                    "Failed writing best checkpoint metadata: {}",
                    best_meta.display()
                )
            })?;
        }
    }

    let history_path = run_dir.join("metrics_history.json");
    fs::write(
        &history_path,
        serde_json::to_string_pretty(&epoch_summaries)?,
    )
    .with_context(|| {
        format!(
            "Failed writing training history: {}",
            history_path.display()
        )
    })?;

    println!("Run ID: {run_id}");
    println!(
        "Best validation MAE: {:.4} at epoch {}",
        best_val_mae, best_epoch
    );
    println!("Artifacts written to: {}", run_dir.display());
    println!(
        "Note: Rust checkpoints save model weights + JSON metadata. Optimizer state restore is not yet implemented."
    );

    Ok(())
}

fn run_epoch(
    model: &StereoUNet,
    dataset: &FoundationStereoDataset,
    indices: &[usize],
    cfg: &TrainConfig,
    device: &Device,
    mut optimizer: Option<&mut AdamW>,
    global_step: &mut usize,
    rng: &mut StdRng,
    is_training: bool,
) -> anyhow::Result<EpochMetrics> {
    let num_batches = indices.len().div_ceil(cfg.batch_size);
    let progress = ProgressBar::new(num_batches as u64);
    progress.set_style(progress_style());

    let mut total_nll = 0f64;
    let mut total_abs_error = 0f64;
    let mut total_sq_error = 0f64;
    let mut total_sigma = 0f64;
    let mut total_valid_pixels = 0f64;

    let mut interval_nll = 0f64;
    let mut interval_abs_error = 0f64;
    let mut interval_sq_error = 0f64;
    let mut interval_sigma = 0f64;
    let mut interval_valid_pixels = 0f64;

    for (batch_idx, batch_indices) in indices.chunks(cfg.batch_size).enumerate() {
        if is_training {
            *global_step += 1;
        }

        let (inputs, targets) =
            load_batch(dataset, batch_indices, cfg.height, cfg.width, rng, device)?;

        let (predictions, logvar) = model.forward_t(&inputs, is_training)?;

        let mask = targets.gt(0f32)?.to_dtype(DType::F32)?;
        let valid_count = f64::from(mask.sum_all()?.to_scalar::<f32>()?);
        if valid_count <= 0.0 {
            progress.inc(1);
            continue;
        }

        let diff = predictions.broadcast_sub(&targets)?;
        let abs_diff = diff.abs()?;
        let inv_scale = logvar.neg()?.exp()?;
        let nll_map = abs_diff.broadcast_mul(&inv_scale)?.broadcast_add(&logvar)?;

        let masked_nll = nll_map.broadcast_mul(&mask)?;
        let masked_nll_sum = masked_nll.sum_all()?;
        let loss = masked_nll_sum.affine(1.0 / valid_count, 0.0)?;

        if let Some(opt) = optimizer.as_deref_mut() {
            opt.backward_step(&loss)?;
        }

        let abs_sum = abs_diff
            .broadcast_mul(&mask)?
            .sum_all()?
            .to_scalar::<f32>()? as f64;
        let sq_sum = diff
            .sqr()?
            .broadcast_mul(&mask)?
            .sum_all()?
            .to_scalar::<f32>()? as f64;
        let sigma_sum = (logvar
            .affine(0.5, 0.0)?
            .exp()?
            .broadcast_mul(&mask)?
            .sum_all()?
            .to_scalar::<f32>()?) as f64;
        let nll_sum = masked_nll_sum.to_scalar::<f32>()? as f64;

        total_valid_pixels += valid_count;
        total_nll += nll_sum;
        total_abs_error += abs_sum;
        total_sq_error += sq_sum;
        total_sigma += sigma_sum;

        interval_valid_pixels += valid_count;
        interval_nll += nll_sum;
        interval_abs_error += abs_sum;
        interval_sq_error += sq_sum;
        interval_sigma += sigma_sum;

        let batch_mae = abs_sum / valid_count;
        let batch_nll = nll_sum / valid_count;
        progress.set_message(format!("mae={batch_mae:.4} nll={batch_nll:.4}"));
        progress.inc(1);

        if is_training
            && cfg.log_every_batches > 0
            && (*global_step % cfg.log_every_batches == 0)
            && interval_valid_pixels > 0.0
        {
            println!(
                "step={} train_nll={:.5} train_mae={:.5} train_rmse={:.5} train_sigma={:.5}",
                *global_step,
                interval_nll / interval_valid_pixels,
                interval_abs_error / interval_valid_pixels,
                (interval_sq_error / interval_valid_pixels).sqrt(),
                interval_sigma / interval_valid_pixels,
            );
            interval_nll = 0.0;
            interval_abs_error = 0.0;
            interval_sq_error = 0.0;
            interval_sigma = 0.0;
            interval_valid_pixels = 0.0;
        }

        if batch_idx + 1 == num_batches {
            progress.finish_and_clear();
        }
    }

    if total_valid_pixels <= 0.0 {
        anyhow::bail!("No valid target pixels found for this epoch");
    }

    let nll_mean = total_nll / total_valid_pixels;
    let mae = total_abs_error / total_valid_pixels;
    let rmse = (total_sq_error / total_valid_pixels).sqrt();
    let sigma = total_sigma / total_valid_pixels;

    Ok(EpochMetrics {
        loss: nll_mean,
        nll: nll_mean,
        mae,
        rmse,
        sigma,
    })
}

fn load_batch(
    dataset: &FoundationStereoDataset,
    batch_indices: &[usize],
    h: usize,
    w: usize,
    rng: &mut StdRng,
    device: &Device,
) -> anyhow::Result<(Tensor, Tensor)> {
    let cpu = Device::Cpu;
    let mut input_tensors = Vec::with_capacity(batch_indices.len());
    let mut target_tensors = Vec::with_capacity(batch_indices.len());

    for sample_idx in batch_indices {
        let sample = dataset.load_item(*sample_idx, rng)?;
        let input = Tensor::from_vec(sample.input, (6, h, w), &cpu)
            .context("Failed to build input tensor")?;
        let target = Tensor::from_vec(sample.target, (1, h, w), &cpu)
            .context("Failed to build target tensor")?;
        input_tensors.push(input);
        target_tensors.push(target);
    }

    let input_refs: Vec<&Tensor> = input_tensors.iter().collect();
    let target_refs: Vec<&Tensor> = target_tensors.iter().collect();
    let inputs = Tensor::stack(&input_refs, 0)?;
    let targets = Tensor::stack(&target_refs, 0)?;

    Ok((inputs.to_device(device)?, targets.to_device(device)?))
}

fn write_epoch_previews(
    model: &StereoUNet,
    preview_dataset: &FoundationStereoDataset,
    device: &Device,
    preview_root: &Path,
    epoch: usize,
) -> anyhow::Result<()> {
    let epoch_dir = preview_root.join(format!("epoch_{epoch:04}"));
    fs::create_dir_all(&epoch_dir).with_context(|| {
        format!(
            "Failed to create preview epoch dir: {}",
            epoch_dir.display()
        )
    })?;

    let mut rng = StdRng::seed_from_u64(0);
    let h = preview_dataset.cfg.image_height;
    let w = preview_dataset.cfg.image_width;

    for index in 0..preview_dataset.len() {
        let sample = preview_dataset.load_item(index, &mut rng)?;

        let input = Tensor::from_vec(sample.input.clone(), (1, 6, h, w), &Device::Cpu)?
            .to_device(device)?;
        let (pred, _) = model.forward_t(&input, false)?;
        let pred_cpu = pred.to_device(&Device::Cpu)?;
        let pred_map = pred_cpu.flatten_all()?.to_vec1::<f32>()?;

        let save_path = epoch_dir.join(format!("sample_{index:03}.png"));
        save_preview_montage(&save_path, &sample.input, &sample.target, &pred_map, h, w)?;
    }

    Ok(())
}

fn resolve_device(device_arg: &str) -> anyhow::Result<Device> {
    match device_arg {
        "auto" => match Device::cuda_if_available(0) {
            Ok(device) => Ok(device),
            Err(_) => Ok(Device::Cpu),
        },
        "cpu" => Ok(Device::Cpu),
        "cuda" => Device::cuda_if_available(0)
            .context("CUDA requested with --device cuda, but CUDA is not available"),
        other => anyhow::bail!("Unsupported --device value: {other} (expected auto|cpu|cuda)"),
    }
}

fn build_cache_only(cfg: &TrainConfig, samples: &[StereoSample]) -> anyhow::Result<()> {
    let cache_root = cfg
        .cache_root
        .as_ref()
        .context("--build-cache-only requires --cache-root")?;
    if samples.is_empty() {
        anyhow::bail!("No samples discovered under {}", cfg.dataset_root);
    }

    let dataset = FoundationStereoDataset::new(
        samples.to_vec(),
        DatasetConfig {
            image_height: cfg.height,
            image_width: cfg.width,
            augment: false,
            augment_cfg: AugmentConfig {
                brightness_jitter: 0.0,
                contrast_jitter: 0.0,
                saturation_jitter: 0.0,
                hue_jitter: 0.0,
                gamma_jitter: 0.0,
                noise_std_max: 0.0,
                blur_prob: 0.0,
                blur_sigma_max: 0.0,
                blur_kernel_size: cfg.blur_kernel_size,
            },
            cache_root: Some(PathBuf::from(cache_root).expand_home()),
            require_cache: false,
        },
    )?;

    let started = Instant::now();
    let (written, skipped) = dataset.build_cache(cfg.overwrite_cache)?;
    println!(
        "Cache build complete: total={} written={} skipped={} elapsed={:.1}s",
        samples.len(),
        written,
        skipped,
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

fn progress_style() -> ProgressStyle {
    ProgressStyle::with_template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
        .unwrap_or_else(|_| ProgressStyle::default_bar())
        .progress_chars("=>-")
}

trait ExpandHome {
    fn expand_home(&self) -> PathBuf;
}

impl ExpandHome for PathBuf {
    fn expand_home(&self) -> PathBuf {
        self.as_path().expand_home()
    }
}

trait ExpandHomePath {
    fn expand_home(&self) -> PathBuf;
}

impl ExpandHomePath for Path {
    fn expand_home(&self) -> PathBuf {
        let path_str = self.to_string_lossy();
        if !path_str.starts_with('~') {
            return self.to_path_buf();
        }

        let home = std::env::var("HOME").unwrap_or_else(|_| String::from("/"));
        if path_str == "~" {
            return PathBuf::from(home);
        }

        if let Some(stripped) = path_str.strip_prefix("~/") {
            return PathBuf::from(home).join(stripped);
        }

        self.to_path_buf()
    }
}
