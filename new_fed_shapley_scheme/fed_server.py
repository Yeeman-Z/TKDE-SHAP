import logging
import fed_proto_pb2
import fed_proto_pb2_grpc
import grpc
from fed_models import rpcio_to_nparray, nparray_to_rpcio
from fed_models import linear_model as LM
from fed_models import CLINT_NUM, BASIC_PORT, DATA_SHAPE, FED_ROUND
# from fed
import numpy as np




# class call_grad_descent_from_client(fed_proto_pb2_grpc.GradDescentServiceServicer):
# here the fed_server calls the remote function of clients.
# _clients = [(c_id, *), (), ...]
def run_fed_server(_clients=CLINT_NUM, _basic_port=BASIC_PORT):

    # build the channel to each client to execute the local trainning.
    ports = [c+_basic_port for c in range(_clients)]  
    c_num = len(_clients) 
    channels =[ grpc.insecure_channel("localhost:"+str(ports[i])) for i in range(c_num)]
    grad_stubs = [fed_proto_pb2_grpc.GradDescentServiceStub(channels[i]) for i in range(c_num)]
    size_stubs = [fed_proto_pb2_grpc.GetDataSizeServiceStub(channels[i]) for i in range(c_num)]

    # Globally train the federated model, i.e. execute the stub.grad_decendent
    global_model = LM(DATA_SHAPE)
    global_model_weights = global_model.model_get_weights()
    datasizes = [size_stubs[c].get_datasize(fed_proto_pb2.datasize_request(0)) for c in range(c_num)]
    all_datasize = np.sum(datasizes)
    grad_alpha = (datasizes/all_datasize).reshape((grad_alpha.shape[0],1))
    
    for r in range(FED_ROUND):
        rpcio_global_model_weights = nparray_to_rpcio(global_model_weights)
        responses = [grad_stubs[c].grad_descent(fed_proto_pb2.server_request(server_grad_para=rpcio_global_model_weights)) for c in range(c_num)]
        responses *= grad_alpha
        global_model_weights = list(responses.mean(axis=0))

    global_model.model_load_weights(global_model_weights)
    # test_acc, test_loss = global_model.model_get_eval(test_data, test_label)
    stop_stus = [fed_proto_pb2_grpc.stop_serverStub(channels[i]) for i in range(c_num)]
    for stub in stop_stus: stub.stop(fed_proto_pb2.stop_request(message="simplex"))


if __name__ == "__main__":
    logging.basicConfig()
    run_fed_server()
