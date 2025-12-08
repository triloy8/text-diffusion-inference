import argparse
import logging
from concurrent import futures
from dataclasses import dataclass
from pathlib import Path

import grpc
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from generate_llada import generate_llada
from modelling_llada import LLaDAModel
from tokenizer import Tokenizer

from textdiffusion.v1 import textdiffusion_pb2
from textdiffusion.v1 import textdiffusion_pb2_grpc


@dataclass
class ModelConfig:
    device: str
    mlp_ratio: int
    d_model: int
    n_heads: int
    rope_theta: float
    max_sequence_length: int
    vocab_size: int
    n_layers: int
    mlp_hidden_size: int
    ckpt_path: Path
    vocab_path: Path
    merges_path: Path
    special_tokens: Path
    repo_id: str


@dataclass
class WorkerConfig:
    uds_path: Path
    model: ModelConfig


class TextGenerationService(textdiffusion_pb2_grpc.TextGenerationServiceServicer):
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def Generate(self, request, context):
        print(
            "GenerateRequest(\n"
            f"  prompt={request.prompt!r},\n"
            f"  max_output_tokens={request.max_output_tokens},\n"
            f"  num_steps={request.num_steps},\n"
            f"  seed={request.seed},\n"
            f"  mask_id={request.mask_id},\n"
            f"  block_length={request.block_length},\n"
            f"  temperature={request.temperature},\n"
            f"  request_id={request.request_id!r}\n"
            ")"
        )

        output_text = generate_llada(model=self.model,
                                     tokenizer=self.tokenizer,
                                     prompt=request.prompt,
                                     mask_id=request.mask_id,
                                     steps=request.num_steps,
                                     gen_length=request.max_output_tokens,
                                     block_length=request.block_length,
                                     temperature=request.temperature)

        return textdiffusion_pb2.GenerateResponse(
            output_text=output_text,
            finish_reason=textdiffusion_pb2.FinishReason.FINISH_REASON_LENGTH,
            request_id=request.request_id,
        )

def serve(config: WorkerConfig) -> None:
    model_cfg = config.model

    model = LLaDAModel(
                mlp_ratio = model_cfg.mlp_ratio,
                d_model = model_cfg.d_model,
                n_heads = model_cfg.n_heads,
                rope_theta = model_cfg.rope_theta,
                max_sequence_length = model_cfg.max_sequence_length,
                vocab_size = model_cfg.vocab_size,
                n_layers = model_cfg.n_layers,
                mlp_hidden_size = model_cfg.mlp_hidden_size,
                device = model_cfg.device,
            )

    model_filepath = hf_hub_download(
        repo_id=model_cfg.repo_id,
        repo_type="model",
        filename=str(model_cfg.ckpt_path)
    )
    state_dict = load_file(model_filepath)
    clean_state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)

    special_tokens_filepath = hf_hub_download(
        repo_id=model_cfg.repo_id,
        repo_type="model",
        filename=str(model_cfg.special_tokens)
    )
    vocab_filepath = hf_hub_download(
        repo_id=model_cfg.repo_id,
        repo_type="model",
        filename=str(model_cfg.vocab_path)
    )
    merges_filepath = hf_hub_download(
        repo_id=model_cfg.repo_id,
        repo_type="model",
        filename=str(model_cfg.merges_path)
    )
    tokenizer = Tokenizer.from_files(
        vocab_filepath=vocab_filepath,
        merges_filepath=merges_filepath, 
        special_tokens=special_tokens_filepath
    )
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    textdiffusion_pb2_grpc.add_TextGenerationServiceServicer_to_server(
        TextGenerationService(model, tokenizer, model_cfg.device), server
    )
    server.add_insecure_port(f"unix://{config.uds_path}")
    server.start()
    logging.info(
        "TextGenerationService listening on uds socket %s",
        config.uds_path,
    )
    server.wait_for_termination()


def parse_args() -> WorkerConfig:
    parser = argparse.ArgumentParser(description="text-diffusion worker")
    parser.add_argument("--uds-path", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--mlp-ratio", type=int, required=True)
    parser.add_argument("--d-model", type=int, required=True)
    parser.add_argument("--n-heads", type=int, required=True)
    parser.add_argument("--rope-theta", type=float, required=True)
    parser.add_argument("--max-sequence-length", type=int, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--n-layers", type=int, required=True)
    parser.add_argument("--mlp-hidden-size", type=int, required=True)
    parser.add_argument("--ckpt-path", type=Path, required=True)
    parser.add_argument("--vocab-path", type=Path, required=True)
    parser.add_argument("--merges-path", type=Path, required=True)
    parser.add_argument("--special-tokens", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)

    args = parser.parse_args()
    model_cfg = ModelConfig(
        device=args.device,
        mlp_ratio=args.mlp_ratio,
        d_model=args.d_model,
        n_heads=args.n_heads,
        rope_theta=args.rope_theta,
        max_sequence_length=args.max_sequence_length,
        vocab_size=args.vocab_size,
        n_layers=args.n_layers,
        mlp_hidden_size=args.mlp_hidden_size,
        ckpt_path=args.ckpt_path,
        vocab_path=args.vocab_path,
        merges_path=args.merges_path,
        special_tokens=args.special_tokens,
        repo_id=args.repo_id,
    )

    return WorkerConfig(
        uds_path=args.uds_path,
        model=model_cfg,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = parse_args()
    serve(config)
