import logging
import fed_proto_pb2
import fed_proto_pb2_grpc
import grpc
from concurrent import futures


from fed_models import rpcio_to_nparray, nparray_to_rpcio
from fed_models import linear_model as LM
from fed_models import CLINT_NUM, BASIC_PORT, DATA_SHAPE, FED_ROUND

import threading
import numpy as np

class datasize_servicier(fed_proto_pb2_grpc.GetDataSizeServiceServicer):
    def get_datasize():
        pass

class grad_descent_servicer(fed_proto_pb2_grpc.GradDescentServiceServicer):
    def grad_descent():
        pass

class stop_server(fed_proto_pb2_grpc.stop_serverServicer):
    def __init__(self, stop_event, cid):
        self._stop_event = stop_event
        self._cid = cid
    def stop(self, request, context):
        print("The client %d received the stop request from " % (self._cid)+ request.message)
        self._stop_event.set()
        return fed_proto_pb2.stop_reply(message="The client %d has been stopped!" % (self._cid))

def run_fed_client(cid):
    stop_event = threading.Event()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    fed_proto_pb2_grpc.add_GetDataSizeServiceServicer_to_server(datasize_servicier(), server)
    fed_proto_pb2_grpc.add_GradDescentServiceServicer_to_server(grad_descent_servicer(), server)
    fed_proto_pb2_grpc.add_stop_serverServicer_to_server(stop_server(stop_event, cid), server)
    server.add_insecure_port("[::]:"+str(BASIC_PORT+cid))
    # pass
    server.start()
    stop_event.wait()
    # print("The client {} stop running. ".format(cid))


def create_client_threads(_c_num=CLINT_NUM):
    client_threads = [threading.Thread(target=run_fed_client, 
                                       name="fed_client_"+str(cid), 
                                       args=(cid), daemon=True)
                                       for cid in range(_c_num)]
    for cth in client_threads:
        print(cth.getName(), "is running.")
        cth.start()
        cth.join()
    # pass

if __name__ == "__main__":
    logging.basicConfig()
    create_client_threads()




