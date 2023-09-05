import logging
import fed_proto_pb2
import fed_proto_pb2_grpc
import grpc
from fed_models import rpcio_to_nparray, nparray_to_rpcio
from fed_models import linear_model as LM
from fed_models import *
# from fed
import numpy as np
import pickle as pk




# class call_grad_descent_from_client(fed_proto_pb2_grpc.GradDescentServiceServicer):
# here the fed_server calls the remote function of clients.
# _clients = [(c_id, *), (), ...]
def run_fed_server(_clients=CLINT_NUM, _basic_port=BASIC_PORT):

    # build the channel to each client to execute the local trainning.
    ports = [c+_basic_port for c in range(_clients)]  
    c_num = _clients 
    channels =[ grpc.insecure_channel("localhost:"+str(ports[i])) for i in range(c_num)]
    grad_stubs = [fed_proto_pb2_grpc.GradDescentServiceStub(channels[i]) for i in range(c_num)]
    size_stubs = [fed_proto_pb2_grpc.GetDataSizeServiceStub(channels[i]) for i in range(c_num)]
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
    global_model = LM(DATA_SHAPE[0], DATA_SHAPE[1])
    global_model_weights = global_model.model_get_weights()
    datasize_response = [size_stubs[c].get_datasize(fed_proto_pb2.datasize_request(size=0)) for c in range(c_num)]
    np_datasize = np.array([r.size for r in datasize_response])
    all_datasize = np.sum(np_datasize)
    print(" ## The datasize from rpc is :", np_datasize, " ## allsize is %d"%(all_datasize))
    grad_alpha = (np_datasize/all_datasize)
    grad_alpha = grad_alpha.reshape((grad_alpha.shape[0],1)) 
    print(" ## grad_alpha is :", grad_alpha)

    for r in range(FED_ROUND):
        grad_data, grad_type, grad_shape = nparray_to_rpcio(global_model_weights) # data, type, shape
        rpcio_responses = [grad_stubs[c].grad_descent(
            fed_proto_pb2.server_request(server_grad_para_data=grad_data, server_grad_para_type=grad_type, 
                                         server_grad_para_shape=grad_shape)) for c in range(c_num)]
        print("Have got the response of weights from client.")
        responses = [rpcio_to_nparray(r.client_grad_para_data, r.client_grad_para_type, r.client_grad_para_shape) for r in rpcio_responses]
        responses *= grad_alpha
        global_model_weights = list(responses.mean(axis=0))

        # test the performance of global model.
        global_model.model_load_weights(global_model_weights)
        test_loss, test_acc = global_model.model_get_eval(testX, testY)
        print("Round#{}#, Acc:{}, Loss:{}".format(r, test_acc, test_loss))
    
    stop_stus = fed_proto_pb2_grpc.stop_serverStub(grpc.insecure_channel("localhost:"+str(STOP_PORT)))
    stop_stus.stop(fed_proto_pb2.stop_request(message="simplex"))

if __name__ == "__main__":
    logging.basicConfig()
    # load_data
    run_fed_server()
