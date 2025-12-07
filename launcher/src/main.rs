use clap::Parser;
use serde::Deserialize;
use std::{
    fs,
    path::{Path, PathBuf},
};
use std::process::{Command, Child};

#[derive(Parser)]
#[command(name = "launcher", about = "Text Diffusion Inference launcher")]
struct Cli {
    #[arg(short, long, value_name = "FILE", default_value = "config.toml")]
    config: PathBuf,
}

#[derive(Debug, Deserialize)]
struct LaunchConfig {
    model: ModelConfig,
    router: RouterConfig,
    worker: WorkerConfig,
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
struct RouterConfig {
    host: String,
    port: u16,
}

#[derive(Debug, Deserialize)]
struct WorkerConfig {
    sock: String,
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

fn spawn_worker(uds_path: &str) -> std::io::Result<Child> {
    Command::new("worker")
        .spawn()
}

/// exit on worker died before ready
/// socket exists -> assume server is listening
/// timeout
fn wait_for_worker_ready(worker: &mut Child, uds_path: &str) -> anyhow::Result<()> {
    Ok(())
}

fn spawn_router(uds_path: &str, host: &str, port: u16) -> std::io::Result<Child> {
    Command::new("router")
        .spawn()
}

/// loop check worker
/// loop check router
fn supervise(mut worker: Child, mut router: Child) -> anyhow::Result<()> {
    Ok(())
}

fn run(config: LaunchConfig) -> anyhow::Result<()> {
    let uds_path = &config.worker.sock;
    let host = &config.router.host;
    let port = config.router.port;

    println!("starting worker…");
    let mut worker = spawn_worker(uds_path)?;

    println!("waiting for worker ready…");
    wait_for_worker_ready(&mut worker, uds_path)?;

    println!("starting router…");
    let router = spawn_router(uds_path, &host, port)?;

    println!(
        "router listening on {}:{}, worker at sock {}",
        config.router.host,
        config.router.port,
        config.worker.sock,
    );

    supervise(worker, router)

}