# ==========================================
# Proto generation
# ==========================================

# Compile proto for Worker only
proto-worker:
    @echo "Compiling protobuf for Python..."
    cd worker && \
    uv run -m grpc_tools.protoc \
        -I ../proto \
        --python_out=. \
        --grpc_python_out=. \
        ../proto/textdiffusion/v1/textdiffusion.proto
    @echo "Python protobuf successfully generated."

# Rust compilation triggers proto generation via build.rs
proto-rust:
    @echo "Building Rust router..."
    cd router && cargo build

# Compile both Python + Rust
proto: proto-worker proto-rust
    @echo "All protobufs compiled."


# ==========================================
# Worker
# ==========================================

install-worker:
    @echo "Installing Python worker deps from pyproject.toml..."
    cd worker && uv sync

# Run Python worker
run-worker:
    @echo "Starting Python worker..."
    cd worker && uv run server.py


# ==========================================
# Rust Router
# ==========================================

build-router:
    @echo "Building Rust router..."
    cd router && cargo build

run-router:
    @echo "Running Rust router..."
    cd router && cargo run

router-fmt:
    cd router && cargo fmt

router-lint:
    cd router && cargo clippy


# ==========================================
# Rust Launcher
# ==========================================

build-launcher:
    @echo "Building Rust launcher..."
    cd launcher && cargo build

run-launcher:
    @echo "Running Rust launcher..."
    cd launcher && cargo run

launcher-fmt:
    cd launcher && cargo fmt

launcher-lint:
    cd launcher && cargo clippy


# ===========================
# Convenience / meta commands
# ===========================

# One-time setup: generate proto + install Python deps + build router
setup: install-worker build-router build-launcher proto
    @echo "Setup complete."


# ==========================================
# Clean
# ==========================================

clean:
    @echo "Cleaning Rust + Python artifacts..."
    cd router && cargo clean
    cd launcher && cargo clean
    rm -rf worker/.venv
    rm -rf worker/textdiffusion
    @echo "Clean complete."
