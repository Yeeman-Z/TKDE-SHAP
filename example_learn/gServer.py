import grpc
import logging
from concurrent import futures

import gSC_pb2_grpc
import gSC_pb2

class greeter(gSC_pb2_grpc.greeterServicer):
    def say_hello(self, request, context):
        return gSC_pb2.hello_reply(message="Hello, %s ! This is the REPLY!" % (request))


def serve():
    server=grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    gSC_pb2_grpc.add_greeterServicer_to_server(greeter(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    logging.basicConfig()
    serve()

    


