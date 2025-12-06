use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "launcher", about = "Text Diffusion Inference launcher")]
struct Cli {
    #[arg(short, long, value_name = "FILE", default_value = "config.toml")]
    config: std::path::PathBuf,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
}


fn main() -> anyhow::Result<() >{
    Ok(())
}