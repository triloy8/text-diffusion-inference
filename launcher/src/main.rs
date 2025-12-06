use clap::{Parser};
use serde::{Deserialize};
use std::{fs, path::{PathBuf, Path}};


#[derive(Parser)]
#[command(name = "launcher", about = "Text Diffusion Inference launcher")]
struct Cli {
    #[arg(short, long, value_name = "FILE", default_value = "config.toml")]
    config: PathBuf,
}

#[derive(Deserialize)]
struct LaunchConfig {
}

fn load_config(path: &Path) -> anyhow::Result<LaunchConfig> {
    let bytes = fs::read(path)?;
    Ok(toml::from_slice(&bytes)?)
}

fn run(config: LaunchConfig) -> anyhow::Result<()> {
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let cfg = load_config(&cli.config)?;
    run(cfg)
}