use clap::Parser;
use serde::Deserialize;
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Parser)]
#[command(name = "launcher", about = "Text Diffusion Inference launcher")]
struct Cli {
    #[arg(short, long, value_name = "FILE", default_value = "config.toml")]
    config: PathBuf,
}

#[derive(Debug, Deserialize)]
struct LaunchConfig {
    model: ModelConfig,
    router: EndpointConfig,
    worker: EndpointConfig,
}

#[derive(Debug, Deserialize)]
struct ModelConfig {
    device: String,
    mlp_ratio: u32,
    d_model: u32,
    n_heads: u32,
    rope_theta: f64,
    max_sequence_length: u32,
    vocab_size: u32,
    n_layers: u32,
    mlp_hidden_size: u32,
    ckpt_path: PathBuf,
    vocab_path: PathBuf,
    merges_path: PathBuf,
    special_tokens: PathBuf,
    repo_id: String,
}

#[derive(Debug, Deserialize)]
struct EndpointConfig {
    host: String,
    port: u16,
}

fn load_config(path: &Path) -> anyhow::Result<LaunchConfig> {
    let bytes = fs::read(path)?;
    Ok(toml::from_slice(&bytes)?)
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let cfg = load_config(&cli.config)?;
    run(cfg)
}

fn run(config: LaunchConfig) -> anyhow::Result<()> {
    println!(
        "router listening on {}:{}, worker at {}:{}, model repo: {}",
        config.router.host,
        config.router.port,
        config.worker.host,
        config.worker.port,
        config.model.repo_id
    );
    Ok(())
}