use anyhow::{self, Context};
use clap::Parser;
use serde::Deserialize;
use std::{
    fs,
    path::{Path, PathBuf},
};
use std::process::{Command, Child};
use std::thread::sleep;
use std::time::{Duration, Instant};

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
    binary: String,
    host: String,
    port: u16,
}

#[derive(Debug, Deserialize)]
struct WorkerConfig {
    command: String,
    args: Vec<String>,
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

fn spawn_worker(worker: &WorkerConfig, model: &ModelConfig) -> anyhow::Result<Child> {
    let ckpt_path = path_to_string(&model.ckpt_path)?;
    let vocab_path = path_to_string(&model.vocab_path)?;
    let merges_path = path_to_string(&model.merges_path)?;
    let special_tokens = path_to_string(&model.special_tokens)?;

    let mut command = Command::new(&worker.command);
    command.args(&worker.args);

    command
        .arg("--uds-path").arg(&worker.sock)
        .arg("--device").arg(&model.device)
        .arg("--mlp-ratio").arg(model.mlp_ratio.to_string())
        .arg("--d-model").arg(model.d_model.to_string())
        .arg("--n-heads").arg(model.n_heads.to_string())
        .arg("--rope-theta").arg(model.rope_theta.to_string())
        .arg("--max-sequence-length").arg(model.max_sequence_length.to_string())
        .arg("--vocab-size").arg(model.vocab_size.to_string())
        .arg("--n-layers").arg(model.n_layers.to_string())
        .arg("--mlp-hidden-size").arg(model.mlp_hidden_size.to_string())
        .arg("--ckpt-path").arg(ckpt_path)
        .arg("--vocab-path").arg(vocab_path)
        .arg("--merges-path").arg(merges_path)
        .arg("--special-tokens").arg(special_tokens)
        .arg("--repo-id").arg(&model.repo_id)
        .spawn()
        .with_context(|| format!("failed to spawn worker command `{}`", worker.command))
}

fn wait_for_worker_ready(worker: &mut Child, uds_path: &str) -> anyhow::Result<()> {
    let uds = Path::new(uds_path);
    let start = Instant::now();

    loop {
        // exit on worker died before ready
        if let Some(status) = worker.try_wait()? {
            anyhow::bail!("worker exited early with status: {}", status);
        }

        // socket exists -> assume server is listening
        if uds.exists() {
            println!("worker ready in {:?}", start.elapsed());
            return Ok(());
        }

        // timeout
        if start.elapsed() > Duration::from_secs(300) {
            anyhow::bail!("timed out waiting for worker to become ready");
        }

        sleep(Duration::from_millis(100));
    }
}

fn spawn_router(binary: &str, uds_path: &str, host: &str, port: &u16) -> std::io::Result<Child> {
    Command::new(binary)
        .arg("--host").arg(host)
        .arg("--port").arg(port.to_string())
        .arg("--uds-path").arg(uds_path)
        .spawn()
}

fn supervise(mut worker: Child, mut router: Child) -> anyhow::Result<()> {
    loop {
        // loop check worker
        if let Some(status) = worker.try_wait()? {
            eprintln!("worker exited with status {status}, shutting down router...");
            let _ = router.kill();
            let _ = router.wait();
            anyhow::bail!("worker exited");
        }

        // loop check router
        if let Some(status) = router.try_wait()? {
            eprintln!("router exited with status {status}, shutting down worker...");
            let _ = worker.kill();
            let _ = worker.wait();
            anyhow::bail!("router exited");
        }

        sleep(Duration::from_millis(500));
    }
}

fn run(config: LaunchConfig) -> anyhow::Result<()> {
    let worker_cfg = &config.worker;
    let router_binary = &config.router.binary;
    let uds_path = &config.worker.sock;
    let host = &config.router.host;
    let port = &config.router.port;

    println!("starting worker…");
    let mut worker = spawn_worker(worker_cfg, &config.model)?;

    println!("waiting for worker ready…");
    wait_for_worker_ready(&mut worker, uds_path)?;

    println!("starting router…");
    let router = spawn_router(router_binary, uds_path, host, port)?;

    println!(
        "router listening on {}:{}, worker at sock {}",
        config.router.host,
        config.router.port,
        config.worker.sock,
    );

    supervise(worker, router)

}

fn path_to_string(path: &Path) -> anyhow::Result<String> {
    path.to_str()
        .map(|s| s.to_owned())
        .ok_or_else(|| anyhow::anyhow!("path contains invalid UTF-8: {}", path.display()))
}
