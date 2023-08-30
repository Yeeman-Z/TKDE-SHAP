from __future__ import print_function

import grpc
import logging, sys

import gSC_pb2_grpc, gSC_pb2 

def run():
    channel = grpc.insecure_channel("localhost:50051")
    stub = gSC_pb2_grpc.greeterStub(channel)
    response = stub.say_hello(gSC_pb2.hello_request(name=sys.argv[1]))
    
    print("rpc OK!" + sys.argv[1])

if __name__ == "__main__":
    logging.basicConfig()
    run()