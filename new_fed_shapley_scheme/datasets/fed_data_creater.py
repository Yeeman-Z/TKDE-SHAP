import tensorflow as tf
import tensorflow_federated as tff

import argparse as ap  
import numpy as np
import os

import pickle as pk

exist_dataset_names = ["emnist", "gldv2", "stackoverflow", "cifa100", "shakespeare"]

def create_dataset(name, c_num, c_type):
    if not (name in exist_dataset_names):
        print(name + " is not in " + str(exist_dataset_names) + "!")
        return
    
    if name == 'emnist':
        emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()  

        # create train_dataset for cnum clients.
        ids = emnist_train.client_ids
        client_ids = [ids[i:i+len(ids)//c_num] for i in range(0, len(ids)-len(ids)//c_num, len(ids)//c_num)]
        client_data = [[] for i in range(c_num)] 
        for i in range(c_num):
            for id in client_ids[i]:
                client_data[i] += list(emnist_train.create_tf_dataset_for_client(id))
        
        client_trainX, client_trainY = [[] for i in range(c_num)], [[] for i in range(c_num)]
        for i in range(c_num):
            client_trainX[i] = np.array([data_item['pixels'] for data_item in client_data[i]])
            client_trainY[i] = np.array([data_item['label']  for data_item in client_data[i]])
               
        # create test_dataset for all_users.
        test_data = list(emnist_test.create_tf_dataset_from_all_clients())
        testX = np.array([data_item['pixels'] for data_item in test_data])
        testY = np.array([data_item['label'] for data_item in test_data])

        # save the dataset as type of tf.keras.dataset.
        data_path  = "./emnist/client_"+str(c_num)+"_"+str(c_type)+'/'
        if not os.path.exists(data_path): os.mkdir(data_path)

        with open(data_path+"client_trainX.pk", 'wb')  as fout:
            pk.dump(client_trainX, fout)
        with open(data_path+"client_trainY.pk", 'wb')  as fout:
            pk.dump(client_trainY, fout)
        with open(data_path+"testX.pk", 'wb')  as fout:
            pk.dump(testX, fout)        
        with open(data_path+"testY.pk", 'wb')  as fout:
            pk.dump(testY, fout)

        print("Create" + str(c_type) + "-type dataset for " +str(c_num)+ " clients over " + str(name))

        
if __name__ == "__main__":

    #  example: python fed_data_creater.py --dataset='emnist' --c_num=10 --c_type='same'  
    parser = ap.ArgumentParser(description="Create Federated Dataset.")
    parser.add_argument('--dataset', type=str, default= 'emnist')
    parser.add_argument('--c_num', type=int, default=10)
    parser.add_argument('--c_type', type=str, default='same')
    args = parser.parse_args()
    create_dataset(args.dataset, args.c_num, args.c_type)

