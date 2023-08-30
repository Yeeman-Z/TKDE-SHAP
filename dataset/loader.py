import time
import os
import numpy as np
import sys
import random
import collections
import copy
import itertools
import cv2
import tensorflow_federated as tff
from sklearn.model_selection import train_test_split

import scipy.special as scp
import tensorflow as tf

# from utils.fedcom import federated_train, DATSETLABEL, DATSETSHAPE
from myconst.dataset_const import *
from myconst.train_const import *
from utils.fedcom import CLIENT_NUM, DATSETLABEL



def load_mnist(expType="SAME"):
    '''
        Summary of Mnist DataSet
        trainX, trainY-->(60000, 28, 28), (60000,)
        testX, testY-->(10000, 28, 28), (10000,)
    '''

    print("ExpType", expType)
    # testX, testY = [],[]
    (trainX, trainY), (testX_m, testY_m) = tf.keras.datasets.mnist.load_data()
    trainX, testX_m = trainX / 255.0, testX_m / 255.0
    trainX = trainX.reshape((trainX.shape[0], trainX.shape[1] * trainX.shape[2]))
    testX_m = testX_m.reshape((testX_m.shape[0], testX_m.shape[1] * testX_m.shape[2]))
    testX.extend(testX_m)
    testY.extend(testY_m)

    # trainSampleNumber = trainX.shape[0]
    setY = set(trainY.reshape(trainY.shape[0]))
    sampleDictY = dict.fromkeys(setY)
    # Partition the mnist dataset by label SampleDictY[k]-->{id_1, id_x, ..., id_t} with label k
    for i in range(trainX.shape[0]):
        if sampleDictY.get(trainY[i]):
            sampleDictY[trainY[i]].append(i)
        else:
            sampleDictY[trainY[i]] = [i]
    # clientTrainX, clientTrainY = [[] for i in range(CLIENT_NUM)], [[] for i in range(CLIENT_NUM)]
    if expType in ["SAME", "NoiseX", "NoiseY"]:
        for y in setY:
            dataIndex = sampleDictY[y]
            # print("There is [%d] samples with label %d"%(len(dataIndex), y))
            sampleNumber4Client = len(dataIndex) // CLIENT_NUM
            for i in range(CLIENT_NUM):
                dataIndex4Client = dataIndex[i * sampleNumber4Client:(i + 1) * sampleNumber4Client]
                clientTrainX[i].extend([trainX[s] for s in dataIndex4Client])
                clientTrainY[i].extend([trainY[s] for s in dataIndex4Client])

        if expType == "NoiseX":
            noiseRatio = [i * 0.02 for i in range(CLIENT_NUM)]
            # noiseXshape =
            for client in range(CLIENT_NUM):
                noiseX = noiseRatio[client] * np.random.normal(size=(len(clientTrainX[client]), trainX.shape[1]))
                clientTrainX[client] += np.abs(noiseX)

        if expType == "NoiseY":
            noiseRatio = [0.02 * i for i in range(CLIENT_NUM)]
            for client in range(CLIENT_NUM):
                mask = np.random.random(size=len(clientTrainY[client])) < noiseRatio[client]
                values = np.random.randint(0, 10, size=len(mask))
                clientTrainY[client] = (1 - mask) * clientTrainY[client] + mask * values

    elif expType == "VarySize":
        dataSizeRatio = [i / sum(range(1, CLIENT_NUM + 1)) for i in range(1, CLIENT_NUM + 1)]
        # print(dataSizeRatio)
        for y in setY:
            dataIndex = sampleDictY[y]
            # print("Now we create datasets in label={}".format(y))
            for i in range(CLIENT_NUM):
                sampleNumber4Client_Begin = int(sum(dataSizeRatio[:i]) * len(dataIndex))
                sampleNumber4Client_End = int(sum(dataSizeRatio[:i + 1]) * len(dataIndex))
                print("client {} [{}, {}]".format(i, sampleNumber4Client_Begin, sampleNumber4Client_End))
                dataIndex4Client = dataIndex[sampleNumber4Client_Begin:sampleNumber4Client_End]
                clientTrainX[i].extend([trainX[s] for s in dataIndex4Client])
                clientTrainY[i].extend([trainY[s] for s in dataIndex4Client])

    elif expType == "VaryDistr":
        # Here clientNumber should be 5*k, e.g. 5, 10, 15...
        if CLIENT_NUM % 5 == 0:
            # In this part, we divide this part as main-feature and other-feature,
            # for main label (MLab) with label-y, it takes 40% of data with label-y
            # for other label (OLab) with label-y, it takes 60% of data with label-y
            # each client have 2 MLab and 8 OLab, all client share the data with a specific label
            # E.g  for scenario with 5 clients,
            #   client_0: MLab={label-0 (40%) & label-5 (40%)}, OLab={...}
            #   client_i: MLab={label-k (40%): i%5=k}, OLab={label-j (15%): i%5 neq j}
            #   client_4: MLab={label-4 (40%), label-9(40%)} OFea={label-1(15%),label-2,label-3,label-5...}

            #E.g   for scenario with 20=5*t clients,
            #   client_0: MLab={label-0 (10%) & label-5 (10%)}, OLab={...}
            #   client_i: MLab={label-k (40%/t): i%5=k}, OLab={label-j (60%/t/4): i%5 neq j}
            #   client_4: MLab={label-4 (10%), label-9(10%)} OLab={label-1(3.75%),label-2,label-3,label-5...}


            dis = np.zeros((CLIENT_NUM, DATSETLABEL))
            # sharedRatio = (CLIENT_NUM - 6) / 15.0
            main_label_ratio = 0.4*5/CLIENT_NUM
            other_label_ratio = 0.6*5/CLIENT_NUM/4
            for client in range(CLIENT_NUM):
                for y in range(DATSETLABEL):
                    if (client % 5) == (y % 5): # label_y = CLIENT_NUM % 5 --> Main-Label
                        dis[client][y] = main_label_ratio
                    else: # Other-Label
                        dis[client][y] = other_label_ratio
            # print(dis)
            for y in setY:
                dataIndex = sampleDictY[y]
                print("Now we create datasets in label={}".format(y))
                for i in range(CLIENT_NUM):
                    sampleNumber4Client_Begin = int(sum(dis[:, y][:i]) * len(dataIndex))
                    sampleNumber4Client_End = int(sum(dis[:, y][:i + 1]) * len(dataIndex))
                    # print("[{},{}]".format(sampleNumber4Client_Begin, sampleNumber4Client_End))
                    print("client {} in label={}, [{}, {}]".format(i, y, sampleNumber4Client_Begin,
                                                                   sampleNumber4Client_End))
                    dataIndex4Client = dataIndex[sampleNumber4Client_Begin:sampleNumber4Client_End]
                    clientTrainX[i].extend([trainX[s] for s in dataIndex4Client])
                    clientTrainY[i].extend([trainY[s] for s in dataIndex4Client])
        else:
            print("The clientNumber={} is wrong!".format(CLIENT_NUM))
            pass
    # return testX, testY, trainBatchData

def load_age():
    '''
        Summary of AGE DataSet
        client0 13321 images
        client1 TODO
        client2 TODO 
        client3 TODO
    '''
    print("Age Dataset")
    # read All-Age-Faces -- client0  394*309
    print("loading client0 data......")
    directory_name = "dataset/age/All-Age-Faces Dataset/aglined faces"
    files = os.listdir(r"./" + directory_name)
    D_train, D_test = train_test_split(files, test_size=0.26, random_state=42)
    for filename in D_train:
        img = cv2.imread(directory_name + "/" + filename)
        img = cv2.resize(img, (28, 28))
        img = img / 255.0
        clientTrainX[0].append(img.reshape(img.shape[0] * img.shape[1] * img.shape[2]))
        clientTrainY[0].append(int(filename.split('.')[0].split('A')[1]))
    for filename in D_test:
        img = cv2.imread(directory_name + "/" + filename)
        img = cv2.resize(img, (28, 28))
        img = img / 255.0
        testX.append(img.reshape(img.shape[0] * img.shape[1] * img.shape[2]))
        testY.append(int(filename.split('.')[0].split('A')[1]))

    # read appa-real-release -- client1
    print("loading client1 data......")
    directory_name = "dataset/age/appa-real-release/age_train"
    for filename in os.listdir(r"./" + directory_name):
        img = cv2.imread(directory_name + "/" + filename)
        img = cv2.resize(img, (28, 28))
        img = img / 255.0
        clientTrainX[1].append(img.reshape(img.shape[0] * img.shape[1] * img.shape[2]))
        clientTrainY[1].append(int(filename.split('_')[0]))
    directory_name = "dataset/age/appa-real-release/age_test"
    for filename in os.listdir(r"./" + directory_name):
        img = cv2.imread(directory_name + "/" + filename)
        img = cv2.resize(img, (28, 28))
        img = img / 255.0
        testX.append(img.reshape(img.shape[0] * img.shape[1] * img.shape[2]))
        testY.append(int(filename.split('_')[0]))

    # read imdb-wiki-dataset -- client2
    print("loading client2 data......")
    directory_name = "dataset/age/imdb-wiki-dataset/train"
    for age in os.listdir(r"./" + directory_name):
        for filename in os.listdir(r"./" + directory_name + "/" + age):
            img = cv2.imread(directory_name + "/" + age + "/" + filename)
            img = cv2.resize(img, (28, 28))
            img = img / 255.0
            clientTrainX[2].append(img.reshape(img.shape[0] * img.shape[1] * img.shape[2]))
            clientTrainY[2].append(int(age))
    directory_name = "dataset/age/imdb-wiki-dataset/test"
    for age in os.listdir(r"./" + directory_name):
        for filename in os.listdir(r"./" + directory_name + "/" + age):
            img = cv2.imread(directory_name + "/" + age + "/" + filename)
            img = cv2.resize(img, (28, 28))
            img = img / 255.0
            testX.append(img.reshape(img.shape[0] * img.shape[1] * img.shape[2]))
            testY.append(int(age))

    # read UTKFace -- client3
    print("loading client3 data......")
    directory_name = "dataset/age/UTKFace"
    files = os.listdir(r"./" + directory_name)
    D_train, D_test = train_test_split(files, test_size=0.26, random_state=42)
    for filename in D_train:
        img = cv2.imread(directory_name + "/" + filename)
        img = cv2.resize(img, (28, 28))
        img = img / 255.0
        clientTrainX[3].append(img.reshape(img.shape[0] * img.shape[1] * img.shape[2]))
        clientTrainY[3].append(int(filename.split('_')[0]))
    for filename in D_test:
        img = cv2.imread(directory_name + "/" + filename)
        img = cv2.resize(img, (28, 28))
        img = img / 255.0
        testX.append(img.reshape(img.shape[0] * img.shape[1] * img.shape[2]))
        testY.append(int(filename.split('_')[0]))


def load_FEMNIST():
    pass

def loader(expType="SAME"):

    # create the Train Datasets (#client) and the Test Dataset,
    global testX, testY, clientTrainX, clientTrainY
    testX = []
    testY = []
    trainBatchData = [[] for _ in range(CLIENT_NUM)]  # Only 4 clients for the Dataset AGE
    clientTrainX = [[] for _ in range(CLIENT_NUM)] # e.g, if we have 4 clients --> clientTrainX = [[], [], [], []]
    clientTrainY = [[] for _ in range(CLIENT_NUM)]

    # Choose the datasets by var(DATASET)
    if DATASET == 'AGE':
        load_age() # real datasets have been partitioned into 4 clients.
    elif DATASET == 'MNIST':
        load_mnist(expType) # can be human partitioned.
        
    # randomly sample data item by proportion
    proportion = 1.0
    for client in range(CLIENT_NUM):
        client_size = len(clientTrainX[client]) # datasize of each client.
        client_list = [i for i in range(client_size)] # [0, 1, ..., #]
        selectList = random.sample(client_list, round(client_size * proportion)) 
        tempTrainX = []
        tempTrainY = []
        for i in selectList:
            tempTrainX.append(clientTrainX[client][i])  
            tempTrainY.append(clientTrainY[client][i]) 
        clientTrainX[client] = tempTrainX
        clientTrainY[client] = tempTrainY

    # create batches for each client with BATCH_SIZE
    for client in range(CLIENT_NUM):
        # print("Now We Create the Client #%d with [%d] train samples" %(client, len(clientTrainX[client])))
        for i in range(0, min(NUM_EXAMPLES_PER_USER, len(clientTrainX[client]) - BATCH_SIZE), BATCH_SIZE):
            trainBatchData[client].append({
                "x": np.array(clientTrainX[client][i:i + BATCH_SIZE], dtype=np.float32),
                "y": np.array(clientTrainY[client][i:i + BATCH_SIZE], dtype=np.int32)
            })
    # print("data loaded!")
    return testX, testY, trainBatchData