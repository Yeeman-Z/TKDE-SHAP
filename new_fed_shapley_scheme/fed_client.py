import logging
import fed_proto_pb2
import fed_proto_pb2_grpc
import grpc
from concurrent import futures

from fed_models import rpcio_to_nparray, nparray_to_rpcio
from fed_models import linear_model as LM
from fed_models import *
import tensorflow_federated as tff

import threading
import numpy as np
import pickle as pk
import tensorflow as tf
import time
SERVE_STOP_FLAG = False 

OUPUT_INFO = True
# import 

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

class datasize_servicier(fed_proto_pb2_grpc.GetDataSizeServiceServicer):
    def __init__(self, _dataset, _cid):
        self.datasize = len(_dataset[1])
        self.cid = _cid
        # self.dataset

    def get_datasize(self, request, context):
        # pass
        return fed_proto_pb2.datasize_reply(size=self.datasize)

class grad_descent_servicer(fed_proto_pb2_grpc.GradDescentServiceServicer):
    def __init__(self, _dataset, _cid):
        self.dataset = _dataset
        self.model = LM(DATA_SHAPE[0], DATA_SHAPE[1])
        self.cid = _cid

    def grad_descent(self, request, context):

        # Download the global model by grpc (rpcio-->np.array)
        print("The client %d executes the local trainning with %d epoches" %(self.cid, LOCAL_EPOCH))
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
        print("Reply Gradients to Server by client %d" %(self.cid))

        return fed_proto_pb2.client_reply(client_grad_para_data=byte_data, 
                                          client_grad_para_type=byte_type, 
                                          client_grad_para_shape=byte_shape)

class stop_server(fed_proto_pb2_grpc.stop_serverServicer):
    def __init__(self, _stop_event, _cid=0):
        self.stop_event = _stop_event
        self.cid = _cid

    def stop(self, request, context):
        print("The client %d received the stop request from " % (self.cid)+ request.message)
        global SERVE_STOP_FLAG
        SERVE_STOP_FLAG = True
        self.stop_event.set()
        return fed_proto_pb2.stop_reply(message="The client %d has been stopped!" % (self.cid))

def run_fed_client(cid, client_dataset):
    # stop_event = threading.Event()
    # print(stop_event)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    fed_proto_pb2_grpc.add_GetDataSizeServiceServicer_to_server(datasize_servicier(client_dataset, cid), server)
    fed_proto_pb2_grpc.add_GradDescentServiceServicer_to_server(grad_descent_servicer(client_dataset, cid), server)
    # fed_proto_pb2_grpc.add_stop_serverServicer_to_server(stop_server(stop_event, cid), server)
    server.add_insecure_port("[::]:"+str(BASIC_PORT+cid))
    # pass
    print("Rpc_Client_{} is created.".format(cid))
    server.start()
    while not SERVE_STOP_FLAG:
        time.sleep(60)
        print("Client %d is still running" % (cid))
    
    # stop_event.wait()
    print("The client {} have been stoped. ".format(cid))

def stop_fed_client():
    stop_event = threading.Event()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    fed_proto_pb2_grpc.add_stop_serverServicer_to_server(stop_server(stop_event), server)
    server.add_insecure_port("[::]:"+str(STOP_PORT))
    server.start()
    stop_event.wait()


def create_client_threads(_c_num=CLINT_NUM):

    # prepare the dataset for each client
    data_path = "./datasets/emnist/client_5_same/"
    file_client_trainX = open(data_path+'client_trainX.pk', 'rb')
    file_client_trainY = open(data_path+'client_trainY.pk', 'rb')
    client_trainX = pk.load(file_client_trainX)
    client_trainY = pk.load(file_client_trainY)

    if OUPUT_INFO:
        print("Loaded Trainning Dataset for {} clients.".format(_c_num))

    # create threads for each client
    client_threads = [threading.Thread(target=run_fed_client, name="fed_client_"+str(cid), 
                    args=(cid, (client_trainX[cid], client_trainY[cid])), daemon=True) for cid in range(_c_num)]
    
    # if OUPUT_INFO:
    #     print("Loaded Trainning Dataset for {} clients.".format(_c_num))
    
    for cth in client_threads:
        print(cth.getName(), "is running.")
        cth.start()
        # cth.join()
    
    # print()
    stop_thread = threading.Thread(target=stop_fed_client, name="fed_stop_")
    stop_thread.start()
    stop_thread.join()
    # for cth in client_threads:
    #     cth.join()
    #     print(cth.getName(), "joins.")

    print("All threads have been stopped.")

    # pass

if __name__ == "__main__":
    logging.basicConfig()
    create_client_threads()




