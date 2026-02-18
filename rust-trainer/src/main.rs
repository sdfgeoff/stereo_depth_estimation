mod config;
mod dataset;
mod model;
mod preview;
mod train;

use clap::Parser;

fn main() -> anyhow::Result<()> {
    if cfg!(debug_assertions) {
        eprintln!(
            "Warning: running a debug build. Training can be much slower. Use `cargo run --release ...`."
        );
    }
    let cfg = config::TrainConfig::parse();
    train::run_training(&cfg)
}
