import logging
from concurrent import futures

import grpc

from textdiffusion.v1 import textdiffusion_pb2
from textdiffusion.v1 import textdiffusion_pb2_grpc


class TextGenerationService(textdiffusion_pb2_grpc.TextGenerationServiceServicer):
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

        return textdiffusion_pb2.GenerateResponse(
            output_text=request.prompt,
            finish_reason=textdiffusion_pb2.FinishReason.FINISH_REASON_LENGTH,
            request_id=request.request_id,
        )

def serve(host: str = "localhost", port: int = 50057) -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    textdiffusion_pb2_grpc.add_TextGenerationServiceServicer_to_server(
        TextGenerationService(), server
    )
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    logging.info("TextGenerationService listening on %s:%d", host, port)
    server.wait_for_termination()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    serve()