import logging
import fed_proto_pb2
import fed_proto_pb2_grpc
import grpc
import time
from copy import deepcopy

import os
import argparse as ap  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


from fed_models import rpcio_to_nparray, nparray_to_rpcio
from fed_models import *
from helper_shap import *
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D as mp3d

import numpy as np
import pickle as pk

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

# class call_grad_descent_from_client(fed_proto_pb2_grpc.GradDescentServiceServicer):
# here the fed_server calls the remote function of clients.
# _clients = [(c_id, *), (), ...]
def run_fed_server(_args, _basic_port=BASIC_PORT):

    # load the parameters of FL
    c_num = _args.client_num
    _sample_client = eval(_args.rec_sample)
    fed_model = FED_MODEL_DICT[_args.model]
    fed_round = _args.fed_round
    print("Now the combination of client is ", _sample_client)

    # build the channel to each client to execute the local trainning.
    ports = [c+_basic_port for c in range(c_num)]  
    channels= []
    for cid in range(c_num):
        if cid in _sample_client:
            channels.append(grpc.insecure_channel("localhost:"+str(ports[cid])))
    grad_stubs = [fed_proto_pb2_grpc.GradDescentServiceStub(channel) for channel in channels]
    size_stubs = [fed_proto_pb2_grpc.GetDataSizeServiceStub(channel) for channel in channels]

    print("# build the channel to each client to execute the local trainning.")

    # Loaded the test data
    print("# Loaded the test data")
    data_path = "./datasets/emnist/client_5_same/"
    file_testX = open(data_path+'testX.pk', 'rb')
    file_testY = open(data_path+'testY.pk', 'rb')
    testX = pk.load(file_testX)
    testY = pk.load(file_testY)

    # Globally train the federated model, i.e. execute the stub.grad_decendent
    print("# Globally train the federated model, i.e. execute the stub.grad_decendent.")
    data_shape = FED_SHAPE_DICT[_args.dataset]
    global_model = fed_model(data_shape[0], data_shape[1])

    # Check whether the subset of client is []
    if len(_sample_client)==0:
        test_loss, test_acc = global_model.model_get_eval(testX, testY)
        stop_stus = fed_proto_pb2_grpc.stop_serverStub(grpc.insecure_channel("localhost:"+str(STOP_PORT)))
        stop_stus.stop(fed_proto_pb2.stop_request(message="simplex"))
        return test_acc, test_loss        

    print("Next execute FL trainning!")
    global_model_weights = global_model.model_get_weights()
    datasize_response = [size_stubs[c].get_datasize(fed_proto_pb2.datasize_request(size=0)) for c in range(len(size_stubs))]
    np_datasize = np.array([r.size for r in datasize_response])
    all_datasize = np.sum(np_datasize)

    print(" ## The datasize from rpc is :", np_datasize, " ## allsize is %d"%(all_datasize))
    grad_alpha = (np_datasize/all_datasize)
    # print(" ## grad_alpha is :", grad_alpha)

    for r in range(fed_round):
        grad_data, grad_type, grad_shape = nparray_to_rpcio(global_model_weights) # data, type, shape
        rpcio_responses = [grad_stubs[c].grad_descent(
            fed_proto_pb2.server_request(server_grad_para_data=grad_data, server_grad_para_type=grad_type, 
                                         server_grad_para_shape=grad_shape)) for c in range(len(grad_stubs))]
        print("Have got the response of weights from client.")
        responses = [rpcio_to_nparray(r.client_grad_para_data, r.client_grad_para_type, r.client_grad_para_shape) for r in rpcio_responses]
        # responses = [tf.keras.get_weights], tf.keras.get_weights --> list[layer1, layer2, layer...], layer_i --> np.array[p*q] 

        #FedAVG-1
        # grad_alpha = grad_alpha.reshape((grad_alpha.shape[0],1)) 
        # responses *= grad_alpha
        # global_model_weights = list(responses.mean(axis=0))

        #FedAVG-2
        # grad_alpha = (np_datasize/all_datasize)
        temp = [np.zeros(gmw_layer.shape) for gmw_layer in global_model_weights]
        for layers in range(len(global_model_weights)):
            # for i in range(len(responses)):
            for cid in range(len(responses)):
                temp[layers] = temp[layers] + grad_alpha[cid]*responses[cid][layers]

        global_model_weights = deepcopy(temp)
        
        # for ind in range(len(global_model_weights)):
        #     # print(global_model_weights[ind].shape)
        #     if len(global_model_weights[ind].shape)==2:
        #         plt.matshow(global_model_weights[ind])
        #         plt.savefig("./datasets/figs/GMW_W"+str(ind)+"_R"+str(r)+".png")
        #     elif len(global_model_weights[ind].shape)>=3:
        #         plt.matshow(global_model_weights[ind].reshape(global_model_weights[ind].shape[0],-1).transpose())
        #         plt.savefig("./datasets/figs/GMW_W"+str(ind)+"_R"+str(r)+".png")
        #         # ax3d = mp3d(plt.figure())
        #         # ax3d.scatter(global_model_weights[ind][0],global_model_weights[ind][1], global_model_weights[ind][2], c='b', marker="*")
            

        # test the performance of global model.
        global_model.model_load_weights(global_model_weights)
        test_loss, test_acc = global_model.model_get_eval(testX, testY)
        print("Subset<-{}: Round#{}#, Acc:{}, Loss:{}".format(_sample_client, r, test_acc, test_loss))

    
    stop_stus = fed_proto_pb2_grpc.stop_serverStub(grpc.insecure_channel("localhost:"+str(STOP_PORT)))
    stop_stus.stop(fed_proto_pb2.stop_request(message="simplex"))
    
    # record the loss and acc
    test_loss, test_acc = global_model.model_get_eval(testX, testY)
    return test_acc, test_loss

if __name__ == "__main__":
    logging.basicConfig()

    # load parameter of run_fed_server
    parser = ap.ArgumentParser(description="Creating Info. for Comp. Shapley.")
    parser.add_argument("--model", type=str, default='linear')
    parser.add_argument("--client_num", type=int, default=5)
    parser.add_argument("--dataset", type=str, default="emnist")
    parser.add_argument("--rec_sample", type=str, default=str([i for i in range(5)]))
    parser.add_argument("--fed_round", type=int, default=5)
    args = parser.parse_args()

    # waiting for the build-up of client
    for i in range(5):
        print("The Server is waiting for the client until %d seconds."% (5-i))
        time.sleep(1)

    # run the server in FL & record the running time of FL
    begin_time = time.perf_counter()
    acc, loss = run_fed_server(args)
    end_time = time.perf_counter()


    # record the time of this sample in FL
    rec_sample_time(args, acc, loss, end_time-begin_time)
