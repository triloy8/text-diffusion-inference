import logging
from concurrent import futures

import grpc

from textdiffusion.v1 import textdiffusion_pb2
from textdiffusion.v1 import textdiffusion_pb2_grpc


class TextGenerationService(textdiffusion_pb2_grpc.TextGenerationServiceServicer):
    def Generate(self, request, context):
        return textdiffusion_pb2.GenerateResponse(
            output_text="red",
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