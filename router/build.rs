fn main() {
    tonic_prost_build::compile_protos("../proto/textdiffusion/v1/textdiffusion.proto")
        .expect("failed to compile protos");
}