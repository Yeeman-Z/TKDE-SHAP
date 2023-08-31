import grpc
import numpy as np
import logging
from concurrent import futures

import gSC_pb2_grpc
import gSC_pb2
import threading

class greeter(gSC_pb2_grpc.greeterServicer):
    def say_hello(self, request, context):
        # print(type(request.npdata))
        # print(request.npdata)
        byte_data = list(request.npdata)
        byte_shape = list(request.npshape)
        byte_type = list(request.nptypes)

        npdata = [np.frombuffer(data, dtype=np.dtype(rtype)).reshape(eval(shape)) for data,rtype,shape in zip(byte_data, byte_type, byte_shape)]
        
        # print(npdata)
        # mylist = [np.frombuffer()]


        # myarray = 
        # print
        # print(type(mylist), mylist)
        return gSC_pb2.hello_reply(message="Hello, server.  This is the REPLY!" )

class stop_server(gSC_pb2_grpc.stop_serverServicer):
    def __init__(self, stop_event):
        self._stop_event = stop_event
    def stop(self, request, context):
        print("The server received the stop request from "+ request.message)
        self._stop_event.set()
        # gSC_pb2.
        return gSC_pb2.stop_reply(message="Server has been stopped!")

def serve():
    stop_event = threading.Event()
    server=grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    gSC_pb2_grpc.add_greeterServicer_to_server(greeter(), server)
    gSC_pb2_grpc.add_stop_serverServicer_to_server(stop_server(stop_event), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    # server.wait_for_termination()
    stop_event.wait()
    # server.stop()

if __name__ == "__main__":
    logging.basicConfig()
    serve()

    


