import logging
import fed_proto_pb2
import fed_proto_pb2_grpc
import grpc
from concurrent import futures

from fed_models import rpcio_to_nparray, nparray_to_rpcio
from fed_models import linear_model as LM
from fed_models import CLINT_NUM, BASIC_PORT, DATA_SHAPE, FED_ROUND, LOCAL_EPOCH, LOCAL_BATCH
import tensorflow_federated as tff

import threading
import numpy as np
import pickle as pk
import tensorflow as tf
# import 

class datasize_servicier(fed_proto_pb2_grpc.GetDataSizeServiceServicer):
    def __init__(self, _dataset):
        self.datasize = len(_dataset[1])
        # self.dataset

    def get_datasize(self, request, context):
        # pass
        return fed_proto_pb2.datasize_reply(size=self.datasize)

class grad_descent_servicer(fed_proto_pb2_grpc.GradDescentServiceServicer):
    def __init__(self, _dataset):
        self.dataset = _dataset
        self.model = LM(DATA_SHAPE)

    def grad_descent(self, request, context):

        # Download the global model by grpc (rpcio-->np.array)
        byte_data = list(request.server_grad_para_data)
        byte_shape = list(request.server_grad_para_shape)
        byte_type = list(request.server_grad_para_type)
        global_model_weights = rpcio_to_nparray(byte_data, byte_type, byte_shape)
        self.model.model_load_weights(global_model_weights)
        
        # Locally train the globally model
        self.model.model_fit(self.dataset, LOCAL_EPOCH, LOCAL_BATCH)
        client_model_weights = self.model.model_get_weights()
        
        # reply the updates model by grpc (np.array-->rpcio)
        byte_data, byte_type, byte_shape = nparray_to_rpcio(client_model_weights)
        return fed_proto_pb2.client_reply(client_grad_para_data=byte_data, 
                                          client_grad_para_type=byte_type, 
                                          client_grad_para_shape=byte_shape)

class stop_server(fed_proto_pb2_grpc.stop_serverServicer):
    def __init__(self, stop_event, cid):
        self._stop_event = stop_event
        self._cid = cid
    def stop(self, request, context):
        print("The client %d received the stop request from " % (self._cid)+ request.message)
        self._stop_event.set()
        return fed_proto_pb2.stop_reply(message="The client %d has been stopped!" % (self._cid))

def run_fed_client(cid, client_dataset):
    stop_event = threading.Event()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    fed_proto_pb2_grpc.add_GetDataSizeServiceServicer_to_server(datasize_servicier(client_dataset), server)
    fed_proto_pb2_grpc.add_GradDescentServiceServicer_to_server(grad_descent_servicer(client_dataset), server)
    fed_proto_pb2_grpc.add_stop_serverServicer_to_server(stop_server(stop_event, cid), server)
    server.add_insecure_port("[::]:"+str(BASIC_PORT+cid))
    # pass
    server.start()
    stop_event.wait()
    # print("The client {} stop running. ".format(cid))


def create_client_threads(_c_num=CLINT_NUM):

    # prepare the dataset for each client
    data_path = "./datasets/emnist/client_5_same/"
    file_client_trainX = open(data_path+'client_trainX.pk', 'rb')
    file_client_trainY = open(data_path+'client_trainY.pk', 'rb')
    client_trainX = pk.load(file_client_trainX)
    client_trainY = pk.load(file_client_trainY)

    # create threads for each client
    client_threads = [threading.Thread(target=run_fed_client, name="fed_client_"+str(cid), 
                    args=(cid, (client_trainX[cid], client_trainY[cid])), daemon=True) for cid in range(_c_num)]
    
    for cth in client_threads:
        print(cth.getName(), "is running.")
        cth.start()
        cth.join()
    # pass

if __name__ == "__main__":
    logging.basicConfig()
    create_client_threads()




