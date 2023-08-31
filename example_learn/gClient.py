from __future__ import print_function
import numpy as np

import grpc
import logging, sys


import gSC_pb2_grpc, gSC_pb2 

def run():
    channel = grpc.insecure_channel("localhost:50051")
    stub = gSC_pb2_grpc.greeterStub(channel)
    # mylist = [1,2,3,4]
    array_list = [np.array([1,2,3,4]), np.array([6,5,4,3,2,1]), np.array([[1,3],[1,2],[1,3],[1,4]])]
    byte_array_list = [x.tobytes() for x in array_list]
    byte_array_type = [str(x.dtype) for x in array_list]
    byte_array_shape = [str(x.shape) for x in array_list]
    print(array_list)
    print(byte_array_shape)
    print(byte_array_type)
    response = stub.say_hello(gSC_pb2.hello_request(npdata=byte_array_list, nptypes=byte_array_type, npshape=byte_array_shape))
    # print(response.np_test)
    print('--------------')
    print(response.message)
    print('--------------')


    print("rpc OK!"+str(response))


    print("Now I request to stop the server.")
    stub = gSC_pb2_grpc.stop_serverStub(channel)
    response = stub.stop(gSC_pb2.stop_request(message="simplex"))




if __name__ == "__main__":
    logging.basicConfig()
    run()