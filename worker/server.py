from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

from tokenizer import Tokenizer
from modelling_llada import LLaDAModel
from generate_llada import generate_llada

import logging
from concurrent import futures

import grpc

from textdiffusion.v1 import textdiffusion_pb2
from textdiffusion.v1 import textdiffusion_pb2_grpc


class TextGenerationService(textdiffusion_pb2_grpc.TextGenerationServiceServicer):
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def Generate(self, request, context):
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

def serve(host: str = "localhost", port: int = 50057) -> None:
    device="cuda"
    mlp_ratio=4
    d_model=4096
    n_heads=32
    rope_theta=10000.0
    max_sequence_length=1024
    vocab_size=126464
    n_layers=32
    ckpt_path="./model.safetensors"
    vocab_path="./vocab.json"
    merges_path="./merges.txt"
    special_tokens="./special_tokens.json"
    
    model = LLaDAModel(
                mlp_ratio = mlp_ratio,
                d_model = d_model,
                n_heads = n_heads,
                rope_theta = rope_theta,
                max_sequence_length = max_sequence_length,
                vocab_size = vocab_size,
                n_layers = n_layers,
                mlp_hidden_size = 12288,
                device = device,
            )

    model_filepath = hf_hub_download(
        repo_id="trixyL/LLaDA-8B-Instruct-merged",
        repo_type="model",
        filename=ckpt_path
    )
    state_dict = load_file(model_filepath)
    clean_state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)

    special_tokens_filepath = hf_hub_download(
        repo_id="trixyL/LLaDA-8B-Instruct-merged",
        repo_type="model",
        filename=special_tokens
    )
    vocab_filepath = hf_hub_download(
        repo_id="trixyL/LLaDA-8B-Instruct-merged",
        repo_type="model",
        filename=vocab_path
    )
    merges_filepath = hf_hub_download(
        repo_id="trixyL/LLaDA-8B-Instruct-merged",
        repo_type="model",
        filename=merges_path
    )
    tokenizer = Tokenizer.from_files(vocab_filepath=vocab_filepath, 
                                     merges_filepath=merges_filepath, 
                                     special_tokens=special_tokens_filepath)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    textdiffusion_pb2_grpc.add_TextGenerationServiceServicer_to_server(
        TextGenerationService(model, tokenizer, device), server
    )
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    logging.info("TextGenerationService listening on %s:%d", host, port)
    server.wait_for_termination()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    serve()