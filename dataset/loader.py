import time
import os
import numpy as np
import sys
import random
import collections
import copy
import itertools
import cv2
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
    (trainX, trainY), (testX, testY) = tf.keras.datasets.mnist.load_data()
    trainX, testX = trainX / 255.0, testX / 255.0
    trainX = trainX.reshape((trainX.shape[0], trainX.shape[1] * trainX.shape[2]))
    testX = testX.reshape((testX.shape[0], testX.shape[1] * testX.shape[2]))

    # trainSampleNumber = trainX.shape[0]
    setY = set(trainY.reshape(trainY.shape[0]))
    sampleDictY = dict.fromkeys(setY)

    for i in range(trainX.shape[0]):
        if sampleDictY.get(trainY[i]):
            sampleDictY[trainY[i]].append(i)
        else:
            sampleDictY[trainY[i]] = [i]

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
            dis = np.zeros((CLIENT_NUM, DATSETLABEL))
            sharedRatio = (CLIENT_NUM - 2) / 15.0
            for client in range(CLIENT_NUM):
                for y in range(DATSETLABEL):
                    if y % CLIENT_NUM == client:
                        dis[client][y] = (1 - sharedRatio)
                    else:
                        dis[client][y] = sharedRatio / (CLIENT_NUM - 1)
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


def load_age():
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


def loader(expType="SAME"):
    global testX, testY, clientTrainX, clientTrainY
    testX = []
    testY = []
    trainBatchData = [[] for _ in range(CLIENT_NUM)]  # 就4个client
    clientTrainX = [[] for _ in range(CLIENT_NUM)]
    clientTrainY = [[] for _ in range(CLIENT_NUM)]
    if DATASET == 'AGE':
        load_age()
    elif DATASET == 'MNIST':
        load_mnist(expType)
    # create batches for each client with BATCH_SIZE
    for client in range(CLIENT_NUM):
        # print("Now We Create the Client #%d with [%d] train samples" %(client, len(clientTrainX[client])))
        for i in range(0, min(NUM_EXAMPLES_PER_USER, len(clientTrainX[client]) - BATCH_SIZE), BATCH_SIZE):
            trainBatchData[client].append({
                "x": np.array(clientTrainX[client][i:i + BATCH_SIZE], dtype=np.float32),
                "y": np.array(clientTrainY[client][i:i + BATCH_SIZE], dtype=np.int32)
            })
    print("data loaded!")
    return testX, testY, trainBatchData