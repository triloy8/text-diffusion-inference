# Text Diffusion Inference

## What is this?
- Minimal text diffusion inference server loosely inspired by Hugging Face's TGI design.
- Launcher bootstraps a router and worker pair so the worker can host the LLaDA diffusion text model.
- LLaDA diffusion text model is the only supported model right now (and probably the only one for a while).
- Only the happy path is wired up today, future work will still remain barebones.

## Architecture at a glance
```
[launcher] --spawns--> [router] --grpc--> [worker (LLaDA)]
         \ \_spawns--> [worker (LLaDA)]
          \__watches config + lifecycle
```
- Launcher owns orchestration.
- Router keeps lightweight gRPC control/data channels with the worker and exposes the request entrypoint.
- Worker (Python) runs the tokenizer + inference loop for LLaDA.

## Installation
Install the following tooling before launching any components:
- [`rust` and `cargo`](https://doc.rust-lang.org/cargo/getting-started/installation.html) to build the launcher/router binaries.
- [`just`](https://github.com/casey/just?tab=readme-ov-file#installation) for project automation.
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) for managing the worker's Python environment.
- [`protoc`](https://protobuf.dev/installation/) to generate protobuf stubs for router <-> worker traffic.
- Ensure your system has IPv6 loopback enabled (or configure the router host to an IPv4 address) so the router bind step succeeds.
- Linux hosts need the OpenSSL development headers installed (`libssl-dev` on Debian/Ubuntu) so `openssl-sys` can build cleanly.

## Running the project
1. Run `just setup` to install worker dependencies, build the Rust crates, and generate protobufs (debug profile only for now, no release build yet).
2. Launch the full stack with `just run-launcher`, it orchestrates the router and worker.
3. Submit requests to the launcher endpoint to exercise the current happy-path LLaDA flow.
4. Additional `just` recipes exist if you want to run the router or worker manually for debugging.

## Configuration
- Tweak launcher + router knobs + tokenizer paths/model checkpoints via `launcher/config.toml`.

## Sample request
`client/client.py` is a tiny helper that hits the router like the launcher does. Run it after the stack is up:

```bash
uv run client/client.py
```

The script posts a chat-style payload to `http://localhost:3001/v1/chat/completions`:

```python
payload = {
    "model": "text-diffusion-debug",
    "conversation_id": "debug-<uuid>",
    "messages": [
        {"role": "system", "content": "Only respond with ORANGE"},
        {"role": "user", "content": "Say you love me."},
        {"role": "user", "content": "Orange."},
        {"role": "user", "content": "What do you mean?"},
    ],
}
```

Adjust the payload or target URL as needed.

## Troubleshooting
- Proto mismatch? Re-run `just proto` to regenerate stubs for Rust + Python.
- Worker import issues often mean `uv sync` was skipped; rerun `just install-worker`.
- Router/launcher logs stream to stdout; start each via `just` in its own terminal for quick triage.

## Roadmap
- [X] Token abstraction inside the router <-> worker protobuf contract to simplify payload exchange.
- [ ] gRPC healthchecks/info/discovery so the launcher and router can reason about worker state.
- [ ] Sharding experiments.
