import os
import argparse as ap  
import time 
from helper_shap import *
import multiprocessing
import threading


TEST_MODE = True
# exist_dataset_names = ["emnist", "gldv2", "stackoverflow", "cifa100", "shakespeare"]
exist_dataset_names = ["emnist"]
exist_model_names = ["cnn_model", "linear_model"]
# client_numbers = [3, 5, 10, 15]
client_numbers = [3]
# few_round = [1, 5]
global_round = [1]

exit_codes = {}


def run_script(os_system_command, exit_codes):
    exit_codes[os_system_command] = os.system(os_system_command)

def create_sample_data(dataset, model, client_num, rec_grad, local_round, rec_sample, fed_round):

   
    fed_client_com = ['python fed_client.py ' + 
               '--model='+ "'"+model+"'" + ' ' +
               '--dataset='+ "'"+dataset+"'"  + ' ' +
               '--client_num=' + str(client_num) + ' ' +
               '--local_round=' + str(local_round) + ' ' +
                    '--rec_sample=' + "'"+str(rec_sample)+"'" + ' ' +
               '--rec_grad=' + str(rec_grad) + ' ']
    
    # print()
    
    fed_server_com = ['python fed_server.py ' + 
                    '--model='+ "'"+model+"'" + ' ' +
                    '--dataset='+ "'"+dataset+"'"  + ' ' +
                    '--client_num=' + str(client_num) + ' ' +
                    '--rec_sample=' + "'"+str(rec_sample)+"'" + ' ' +
                    '--fed_round=' + str(fed_round)] 
      
    print("os.system-->client",'\n', fed_client_com[0])

    threads = []
    os_command = [fed_client_com[0], fed_server_com[0]] 

    threads = [threading.Thread(target=run_script, args=(osc, exit_codes), daemon=True) for osc in os_command]
    for thd in threads:
        thd.start()
    for thd in threads:
        thd.join()
    assert exit_codes[fed_client_com[0]] == 0
    assert exit_codes[fed_server_com[0]] == 0
    print('Server & Client exits with code=(%d,%d).' % (exit_codes[fed_client_com[0]], exit_codes[fed_server_com[0]]))

def exp_on_all_datasets():
    for dataset in exist_dataset_names:
        for model in exist_model_names:
            for g_round in global_round:
                for c_num in client_numbers:
                    all_set = [i for i in range(c_num)]
                    power_set = buildPowerSets(all_set)
                    for subset in power_set:
                        create_sample_data(dataset=dataset, model=model, client_num=c_num, 
                                               rec_grad=True, local_round=5, rec_sample=subset, fed_round=g_round)                        



if __name__ == "__main__":
    # exp_on_emnist()
    # exp_on_all_datasets()
    CNUM = 20
    if TEST_MODE:
        create_sample_data(dataset='emnist', model="cnn_model", client_num=CNUM, 
                        rec_grad=False, local_round=5, rec_sample=[1,2,3,4,5], fed_round=1)
    else:
        exp_on_all_datasets()

    # parser = ap.ArgumentParser(description="Creating Info. for Comp. Shapley.")
    # parser.add_argument("--model", type=str, default='linear')
    # parser.add_argument("--client_num", type=int, default=5)
    # parser.add_argument("--rec_grad", type=bool, default=False)
    # parser.add_argument("--dataset", type=str, default="emnist")
    # parser.add_argument("--few_round", type=int, default=2)
    # parser.add_argument("--rec_sample", type=bool, default=False)
    # args = parser.parse_args()
    pass
    # os.