use clap::Parser;
use serde::Serialize;

#[derive(Debug, Clone, Serialize, Parser)]
#[command(
    name = "foundation-stereo-depth-rs",
    about = "Train a stereo disparity model on FoundationStereo using Candle"
)]
pub struct TrainConfig {
    #[arg(
        long,
        default_value = "/home/geoffrey/Reference/OffTopic/Datasets/FoundationStereo",
        help = "Path to FoundationStereo dataset root"
    )]
    pub dataset_root: String,

    #[arg(long, default_value_t = 240)]
    pub height: usize,

    #[arg(long, default_value_t = 320)]
    pub width: usize,

    #[arg(long, default_value_t = 10)]
    pub epochs: usize,

    #[arg(long, default_value_t = 8)]
    pub batch_size: usize,

    #[arg(long, default_value_t = 1e-3)]
    pub lr: f64,

    #[arg(long, default_value_t = 1e-4)]
    pub weight_decay: f64,

    #[arg(long, default_value_t = 4)]
    pub num_workers: usize,

    #[arg(long, default_value_t = 0.1)]
    pub val_fraction: f32,

    #[arg(long, default_value_t = 0)]
    pub max_samples: usize,

    #[arg(long, default_value_t = 42)]
    pub seed: u64,

    #[arg(long, default_value = "auto")]
    pub device: String,

    #[arg(long, default_value = "./outputs-rs")]
    pub output_dir: String,

    #[arg(long)]
    pub run_name: Option<String>,

    #[arg(long)]
    pub cache_root: Option<String>,

    #[arg(long, default_value_t = false)]
    pub require_cache: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "Disable asymmetric RGB augmentations"
    )]
    pub no_augment: bool,

    #[arg(long, default_value_t = 0.25)]
    pub brightness_jitter: f32,

    #[arg(long, default_value_t = 0.25)]
    pub contrast_jitter: f32,

    #[arg(long, default_value_t = 0.25)]
    pub saturation_jitter: f32,

    #[arg(long, default_value_t = 0.05)]
    pub hue_jitter: f32,

    #[arg(long, default_value_t = 0.2)]
    pub gamma_jitter: f32,

    #[arg(long, default_value_t = 0.03)]
    pub noise_std_max: f32,

    #[arg(long, default_value_t = 0.0)]
    pub blur_prob: f32,

    #[arg(long, default_value_t = 0.0)]
    pub blur_sigma_max: f32,

    #[arg(long, default_value_t = 5)]
    pub blur_kernel_size: usize,

    #[arg(long, default_value_t = 10)]
    pub log_every_batches: usize,

    #[arg(long, default_value_t = 8)]
    pub preview_samples: usize,

    #[arg(long, default_value_t = false)]
    pub build_cache_only: bool,

    #[arg(long, default_value_t = false)]
    pub overwrite_cache: bool,
}

impl TrainConfig {
    pub fn augment_enabled(&self) -> bool {
        !self.no_augment
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        if !(0.0..1.0).contains(&self.val_fraction) {
            anyhow::bail!(
                "--val-fraction must be in [0, 1), got {}",
                self.val_fraction
            );
        }
        if self.batch_size == 0 {
            anyhow::bail!("--batch-size must be > 0");
        }
        if self.height == 0 || self.width == 0 {
            anyhow::bail!("--height and --width must be > 0");
        }
        if !(0.0..=1.0).contains(&self.blur_prob) {
            anyhow::bail!("--blur-prob must be in [0, 1], got {}", self.blur_prob);
        }
        if self.blur_kernel_size < 3 || self.blur_kernel_size % 2 == 0 {
            anyhow::bail!(
                "--blur-kernel-size must be odd and >= 3, got {}",
                self.blur_kernel_size
            );
        }
        if self.saturation_jitter < 0.0 {
            anyhow::bail!("--saturation-jitter must be >= 0");
        }
        if self.gamma_jitter < 0.0 {
            anyhow::bail!("--gamma-jitter must be >= 0");
        }
        if self.require_cache && self.cache_root.is_none() {
            anyhow::bail!("--require-cache requires --cache-root");
        }
        if self.build_cache_only && self.cache_root.is_none() {
            anyhow::bail!("--build-cache-only requires --cache-root");
        }
        Ok(())
    }
}
