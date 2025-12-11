use anyhow::Context;
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
    tokenizer: TokenizerConfig,
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
    repo_id: String,
    dtype: String,
}

#[derive(Debug, Deserialize)]
struct TokenizerConfig {
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

struct ChildGuard(Option<Child>);

impl ChildGuard {
    fn new(child: Child) -> Self {
        Self(Some(child))
    }

    fn child_mut(&mut self) -> &mut Child {
        self.0.as_mut().expect("child already taken")
    }

    fn into_inner(mut self) -> Child {
        self.0.take().expect("child already taken")
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        if let Some(mut child) = self.0.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
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

fn spawn_worker(worker: &WorkerConfig, model: &ModelConfig) -> anyhow::Result<ChildGuard> {
    let ckpt_path = path_to_string(&model.ckpt_path)?;

    let mut command = Command::new(&worker.command);
    command.args(&worker.args);

    let child = command
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
        .arg("--repo-id").arg(&model.repo_id)
        .arg("--dtype").arg(&model.dtype)
        .spawn()
        .with_context(|| format!("failed to spawn worker command `{}`", worker.command))?;

    Ok(ChildGuard::new(child))
}

fn wait_for_worker_ready(worker: &mut ChildGuard, uds_path: &str) -> anyhow::Result<()> {
    let uds = Path::new(uds_path);
    let start = Instant::now();

    loop {
        // exit on worker died before ready
        if let Some(status) = worker.child_mut().try_wait()? {
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

fn spawn_router(
    binary: &str,
    uds_path: &str,
    host: &str,
    port: &u16,
    repo_id: &str,
    vocab_path: &str,
    merges_path: &str,
    special_tokens: &str,
) -> std::io::Result<ChildGuard> {
    let child = Command::new(binary)
        .arg("--host").arg(host)
        .arg("--port").arg(port.to_string())
        .arg("--uds-path").arg(uds_path)
        .arg("--tokenizer-repo-id").arg(repo_id)
        .arg("--vocab-path").arg(vocab_path)
        .arg("--merges-filepath").arg(merges_path)
        .arg("--specialtokens-filepath").arg(special_tokens)
        .spawn()?;

    Ok(ChildGuard::new(child))
}

fn supervise(mut worker: ChildGuard, mut router: ChildGuard) -> anyhow::Result<()> {
    loop {
        // loop check worker
        if let Some(status) = worker.child_mut().try_wait()? {
            eprintln!("worker exited with status {status}, shutting down router...");
            let _ = router.child_mut().kill();
            let _ = router.child_mut().wait();
            anyhow::bail!("worker exited");
        }

        // loop check router
        if let Some(status) = router.child_mut().try_wait()? {
            eprintln!("router exited with status {status}, shutting down worker...");
            let _ = worker.child_mut().kill();
            let _ = worker.child_mut().wait();
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
    let tokenizer_repo = &config.tokenizer.repo_id;
    let vocab_path = path_to_string(&config.tokenizer.vocab_path)?;
    let merges_path = path_to_string(&config.tokenizer.merges_path)?;
    let special_tokens = path_to_string(&config.tokenizer.special_tokens)?;

    let uds = Path::new(uds_path);
    if uds.exists() {
        let _ = fs::remove_file(uds);
    }

    println!("starting worker…");
    let mut worker = spawn_worker(worker_cfg, &config.model)?;

    println!("waiting for worker ready…");
    wait_for_worker_ready(&mut worker, uds_path)?;

    println!("starting router…");
    let router = spawn_router(
        router_binary,
        uds_path,
        host,
        port,
        tokenizer_repo,
        &vocab_path,
        &merges_path,
        &special_tokens,
    )?;

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
