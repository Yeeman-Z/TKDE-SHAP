import time
import os
import numpy as np
import sys
import random
import collections
import copy
import itertools

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import scipy.special as scp
import tensorflow as tf
from tensorflow.keras.utils import to_categorical as ToOneHot

# from utils.checkup import checkup
from utils.fedcom import federated_train, DATSETLABEL, DATSETSHAPE, CLIENT_NUM
from utils.shapley_value import power2number, buildPowerSets
from utils.record import recordParamsAndLR, getGradient, clearFile, record_Model, getHistoryModels
from myconst.dataset_const import *
from myconst.train_const import *

# wolframclient calculation
from wolframclient.evaluation import SecuredAuthenticationKey, WolframCloudSession
from wolframclient.language import wlexpr

# dataloader
from dataset.loader import load_mnist, loader
# HyperParameters
ROUND_CONVERGE = 5
CONVERGE_CRITERIA = 0.1
CONVERGE_UPPER = 80
sak = SecuredAuthenticationKey(
    'eFyh2bljUDu2bU66hX/5+cQ+fd+AM9BZRwI/ll46lpY=',
    'PqFlaeRrIcm+x8kZ7I2TrWSztz04K3yVE6d7SsbQp28=')
session = WolframCloudSession(credentials=sak)



def FedML(nowsubset, recordParams=False, recordModel=False):
    def testAcc(model):
        probY = tf.nn.softmax(
            np.dot(testX, np.asarray(model['weights'])) + np.asarray(model['bias'])
        )
        predY = tf.equal(tf.argmax(probY, 1), tf.argmax(ToOneHot(testY), 1))
        accuracy = tf.reduce_mean(tf.cast(predY, tf.float32))
        return accuracy
        
    def testLoss(model):
        predicted_y = tf.nn.softmax(tf.matmul(testX, model['weights']) + model['bias'])
        loss = -tf.reduce_mean(tf.reduce_sum(tf.one_hot(testY, DATSETLABEL)*tf.math.log(predicted_y), axis=[1]))
        return loss

    nowSetSize = len(nowsubset)
    print("Now {} clients: {}".format(nowSetSize, nowsubset))

    if recordParams:
        clearFile(nowSetSize, recordModel)

    learning_rate = LEARNING_RATE
    tfInit = tf.keras.initializers.glorot_normal(seed=42)
    model = collections.OrderedDict(
        weights=tfInit(shape=[DATSETSHAPE, DATSETLABEL]),
        bias=tfInit(shape=[DATSETLABEL, ])
    )

    if nowSetSize == 0:
        # set metric 1/2
        metric = testAcc(model)
        print("accuracy:{}".format(metric))
        return metric

    # prepare the trainDataset of a subset clients
    # subsetTrainBacthData = [batchdata for batchdata in trainBatchData]
    subsetTrainBacthData = [trainBatchData[client] for client in nowsubset]

    for _ in range(EPOCHES):
        # get local models of each client
        local_models = federated_train(model, learning_rate, subsetTrainBacthData)

        if recordParams:
            for client in range(nowSetSize):
                recordParamsAndLR(local_models, learning_rate, client)

        # execute aggregations of fedAVG: model' = \sum_c |D_c|/|D|·model_c
        model_weights = tf.Variable(tf.zeros([DATSETSHAPE, DATSETLABEL]))
        model_bias = tf.Variable(tf.zeros([DATSETLABEL]))
        DataSize = tf.constant(1 / nowSetSize)

        for client in range(nowSetSize):
            model_weights = tf.add(tf.multiply(local_models[client]['weights'], DataSize), model_weights)
            model_bias = tf.add(tf.multiply(local_models[client]['bias'], DataSize), model_bias)

        model = collections.OrderedDict(
            weights=model_weights,
            bias=model_bias
        )

        if recordModel:
            record_Model(model)

        learning_rate *= 0.9
        # loss = federated_eval(model, subsetTrainBacthData)
        # print("Round {}, loss={}".format(_, loss))

    # set metric 2/2
    metric = testAcc(model)
    return metric


def LOO(CLIENT_NUM):
    # get all subset of clients : eg. [[],[0],[1],[2],[0,1]...[0,1,2]]
    nowPerm = [i for i in range(CLIENT_NUM)]
    ACCDICT = dict()
    allSetAcc = FedML(nowPerm).numpy()

    looValue = [0 for i in range(CLIENT_NUM)]
    for client in range(CLIENT_NUM):
        subset = copy.deepcopy(nowPerm)
        subset.remove(client)
        subsetAcc = FedML(subset).numpy()
        looValue[client] = float(allSetAcc)-float(subsetAcc)
    print("LOO Value of each Client:", looValue)


def DEF_SHAP(CLIENT_NUM):
    # get all subset of clients : eg. [[],[0],[1],[2],[0,1]...[0,1,2]]
    allClientsSet = buildPowerSets(list(range(CLIENT_NUM)))
    ACCDICT = dict()

    # calculate the accuracy for each subset of the clients
    for subsetClient in allClientsSet:
        subsetAcc = FedML(subsetClient)
        ACCDICT.update({power2number(subsetClient): subsetAcc.numpy()})

    print(ACCDICT)
    # calculate the shapley value according to definition
    shapleyValue = [0 for i in range(CLIENT_NUM)]
    for client in range(CLIENT_NUM):
        sv = 0.0
        for subset in allClientsSet:
            # print()
            if not (client in subset):
                delta = ACCDICT.get(power2number(subset + [client])) - ACCDICT.get(power2number(subset))
                sv = sv + delta / scp.comb(CLIENT_NUM - 1, len(subset))
        shapleyValue[client] = float(sv)
    print("Shapley Value of each Client:", shapleyValue)


def TMC_SHAP(CLIENT_NUM):
    nowPerm = [i for i in range(CLIENT_NUM)]
    #  np.random.shuffle(allSet)
    allSetAcc = FedML(nowPerm).numpy()

    print(allSetAcc)

    ACCDICT = dict()
    ACCDICT.update({power2number(nowPerm): allSetAcc})
    cachedSV = [0, 0, 0, 0, 0]
    tempU = [0 for _ in range(CLIENT_NUM + 1)]  # [1,2, ..., CLIENT_NUM]
    TOLE_DELTA = 0.004
    ConvergenCecriteria = 0.01

    shapleyValue = [0.0 for _ in range(CLIENT_NUM + 1)]
    notConvergence = True
    round = 0
    print("=========Now We Start TMC SHAP==============")
    while notConvergence:
        np.random.shuffle(nowPerm)
        round = round + 1
        lastSV = np.copy(shapleyValue)
        # logPerm.append(nowPerm)
        tempU[0] = FedML([]).numpy()

        print("Perm:{}".format(nowPerm))
        for j in range(0, CLIENT_NUM):

            subPerm = nowPerm[:j + 1]  # j+1 is the now_id of client, j is the pre_id client
            print("Now #{}, with {} clients:{}".format(j + 1, len(subPerm), subPerm), end=" ")

            if np.abs(allSetAcc - tempU[j]) < TOLE_DELTA:
                tempU[j + 1] = tempU[j]
            else:
                if ACCDICT.__contains__(power2number(subPerm)):
                    tempU[j + 1] = ACCDICT.get(power2number(subPerm))
                else:
                    tempU[j + 1] = FedML(subPerm).numpy()
                    ACCDICT.update({power2number(subPerm): tempU[j + 1]})

            print("Acc is {}".format(tempU[j + 1]))
            # in subPerm[j]  j should be 0,1,...,CLIENT_NUM, i.e. just take j as the index
            # in termU[j+1], j denotes the number of clients, i.e. take j+1 as the number
            shapleyValue[subPerm[j]] = (round - 1.0) / round * shapleyValue[subPerm[j]] + (1.0 / round) * (
                        tempU[j + 1] - tempU[j])
            # print("shapleyValue[subPerm[j]]={:.2f}".format(shapleyValue[subPerm[j]]))

        cachedSV.pop(0)
        cachedSV.append(sum([abs(shapleyValue[_] - lastSV[_]) for _ in range(len(lastSV))]))
        if sum(cachedSV) < ConvergenCecriteria or round > 100:
            notConvergence = False

        print("Shapley Value of each Client:", shapleyValue)

    print("Shapley Value of each Client:", shapleyValue)  # [_.numpy() for _ in shapleyValue])


def isNotConverge(last_u, u, CLIENT_NUM):
    if len(last_u) <= ROUND_CONVERGE:
        return True
    for i in range(0, ROUND_CONVERGE):
        ele = last_u[i]
        delta = np.sum(np.abs(u - ele), axis=(0, 1)) / CLIENT_NUM
        if delta > CONVERGE_CRITERIA:
            return True
    return False


def solveFeasible(agentNum, U_D, U):
    eps = 1 / np.sqrt(agentNum) / agentNum / 2.0
    ans = []
    result = []
    while len(result) == 0:
        expr = ""  # expr to evaluate
        for i in range(agentNum - 1):
            expr = expr + "x" + str(i) + ">= 0 &&"
        expr = expr + "x" + str(agentNum - 1) + ">= 0 &&"
        for i in range(agentNum):
            for j in range(i + 1, agentNum):
                # abs(x_i - x_j) <= U_{i,j}
                expr = expr + "Abs[x" + str(i) + "-x" + str(j) + "-" + str(U[i][j]) + "]<=" + str(eps) + "&&"
        for i in range(agentNum - 1):
            expr = expr + "x" + str(i) + "+"
        expr = expr + "x" + str(agentNum - 1) + ">=" + str(U_D) + "&&"
        for i in range(agentNum - 1):
            expr = expr + "x" + str(i) + "+"
        expr = expr + "x" + str(agentNum - 1) + "<=" + str(U_D)
        expr = expr + ", {"
        for i in range(agentNum - 1):
            expr = expr + "x" + str(i) + ","
        expr = expr + "x" + str(agentNum - 1) + "}, Reals"
        expr = "N[FindInstance[" + expr + "]]"

        result = session.evaluate(wlexpr(expr))
        if len(result) > 0:
            ans = [result[0][i][1] for i in range(agentNum)]
        eps = eps * 1.1
        print(eps)
    print(ans)
    return ans


def GTB_SHAP(CLIENT_NUM):
    last_u = []
    rec_u = []
    Z = 0
    for n in range(1, CLIENT_NUM):
        Z += 1 / n
    Z *= 2
    U = np.zeros([CLIENT_NUM, CLIENT_NUM], dtype=np.float32)
    round_t = 0
    while isNotConverge(last_u, U, CLIENT_NUM) and round_t < CONVERGE_UPPER:
        round_t += 1
        round_k = CLIENT_NUM - 1
        random_q = random.random()
        presum = 0
        for k in range(1, CLIENT_NUM):
            presum += CLIENT_NUM / k / (CLIENT_NUM - k) / Z
            if random_q <= presum:
                round_k = k
                break
        random_seq = random.sample(range(0, CLIENT_NUM), round_k)
        U = (round_t - 1) / round_t * U
        u_t = FedML(random_seq).numpy()
        for i in range(0, CLIENT_NUM):
            for j in range(0, CLIENT_NUM):
                delta_beta = random_seq.count(i) - random_seq.count(j)
                if delta_beta != 0:
                    U[i][j] += delta_beta * u_t * Z / round_t
        if len(last_u) > ROUND_CONVERGE:
            del (last_u[0])
        last_u.append(U)
        rec_u.append(U)
    U_D = FedML(range(0, CLIENT_NUM)).numpy()
    shapleyValue = solveFeasible(CLIENT_NUM, U_D, U)
    print("Shapley Value of each Client:", shapleyValue)


def train_with_gradient(nowsubset, grad_w, grad_b, lr, iter_num=0, model_history=None):
    def testAcc(model):
        probY = tf.nn.softmax(
            np.dot(testX, np.asarray(model['weights'])) + np.asarray(model['bias'])
        )
        predY = tf.equal(tf.argmax(probY, 1), tf.argmax(ToOneHot(testY), 1))
        accuracy = tf.reduce_mean(tf.cast(predY, tf.float32))
        return accuracy

    nowSetSize = len(nowsubset)
    print("Now {} clients: {}".format(nowSetSize, nowsubset))

    tfInit = tf.keras.initializers.glorot_normal(seed=42)
    model = collections.OrderedDict(
        weights=tfInit(shape=[DATSETSHAPE, DATSETLABEL]),
        bias=tfInit(shape=[DATSETLABEL, ])
    )

    if nowSetSize == 0:
        accuracy = testAcc(model)
        print("accuracy:{}".format(accuracy))
        return accuracy

    if model_history == None:
        for _ in range(EPOCHES):
            # execute aggregations of fedAVG: model' = \sum_c |D_c|/|D|·model_c
            model_weights = tf.Variable(tf.zeros([DATSETSHAPE, DATSETLABEL]))
            model_bias = tf.Variable(tf.zeros([DATSETLABEL]))
            DataSize = tf.constant(1 / nowSetSize)

            for client in nowsubset:
                model_weights = tf.add(tf.multiply(grad_w[client][_], DataSize), model_weights)
                model_bias = tf.add(tf.multiply(grad_b[client][_], DataSize), model_bias)

            model_weights = np.subtract(model['weights'], np.multiply(lr[0][_], model_weights))
            model_bias = np.subtract(model['bias'], np.multiply(lr[0][_], model_bias))
            model = collections.OrderedDict(
                weights=model_weights,
                bias=model_bias
            )
            # loss = federated_eval(model, subsetTrainBacthData)
            # print("Round {}, loss={}".format(_, loss))
    else:
        model_weights = tf.Variable(tf.zeros([DATSETSHAPE, DATSETLABEL]))
        model_bias = tf.Variable(tf.zeros([DATSETLABEL]))
        DataSize = tf.constant(1 / nowSetSize)

        for client in nowsubset:
            model_weights = tf.add(tf.multiply(grad_w[client][iter_num - 1], DataSize), model_weights)
            model_bias = tf.add(tf.multiply(grad_b[client][iter_num - 1], DataSize), model_bias)

        model_weights = np.subtract(model_history['weights'], np.multiply(lr[0][iter_num - 1], model_weights))
        model_bias = np.subtract(model_history['bias'], np.multiply(lr[0][iter_num - 1], model_bias))
        model = collections.OrderedDict(
            weights=model_weights,
            bias=model_bias
        )

    accuracy = testAcc(model)
    # print("accuracy:{}".format(accuracy))
    return accuracy


def OR_SHAP(CLIENT_NUM):
    # Record the params and LR of the participants
    nowPerm = [i for i in range(CLIENT_NUM)]
    FedML(nowPerm, recordParams=True)

    # Get the gradients of the participant
    gradient_weights = []
    gradient_biases = []
    gradient_lrs = []
    for i in range(CLIENT_NUM):
        gradient_weights_local, gradient_biases_local, learning_rate_local = getGradient(i)
        gradient_weights.append(gradient_weights_local)
        gradient_biases.append(gradient_biases_local)
        gradient_lrs.append(learning_rate_local)

    # Calculate the accuracy for each subset of the clients
    allClientsSet = buildPowerSets(list(range(CLIENT_NUM)))
    ACCDICT = dict()
    for subsetClient in allClientsSet:
        subsetAcc = train_with_gradient(subsetClient, gradient_weights, gradient_biases, gradient_lrs)
        ACCDICT.update({power2number(subsetClient): subsetAcc.numpy()})
    shapleyValue = [0 for i in range(CLIENT_NUM)]
    for client in range(CLIENT_NUM):
        sv = 0.0
        for subset in allClientsSet:
            if not (client in subset):
                delta = ACCDICT.get(power2number(subset + [client])) - ACCDICT.get(power2number(subset))
                sv = sv + delta / scp.comb(CLIENT_NUM - 1, len(subset))
        shapleyValue[client] = float(sv)
    print("Shapley Value of each Client:", shapleyValue)


def normal(inList):
    symbolF = False
    for i in inList:
        if i < 0:
            symbolF = True
    if symbolF:
        for i in range(len(inList)):
            inList[i] = 0.1 / 5
    return inList


def MR_SHAP(CLIENT_NUM):
    # Record the params and LR of the participants
    decay = 0.8
    nowPerm = [i for i in range(CLIENT_NUM)]
    FedML(nowPerm, recordParams=True, recordModel=True)

    # Get the gradients of the participant and the models
    gradient_weights = []
    gradient_biases = []
    gradient_lrs = []
    for i in range(CLIENT_NUM):
        gradient_weights_local, gradient_biases_local, learning_rate_local = getGradient(i)
        gradient_weights.append(gradient_weights_local)
        gradient_biases.append(gradient_biases_local)
        gradient_lrs.append(learning_rate_local)

    allClientsSet = buildPowerSets(list(range(CLIENT_NUM)))
    models_history = getHistoryModels()
    client_shapley_history = []

    # C the CIs of each epoch
    for iter_num in range(1, len(gradient_weights[0]) + 1):
        ACCDICT = dict()
        for subsetClient in allClientsSet:
            subsetAcc = train_with_gradient(subsetClient, gradient_weights, gradient_biases, gradient_lrs, iter_num,
                                            models_history[iter_num - 1])
            ACCDICT.update({power2number(subsetClient): subsetAcc.numpy()})
        shapleyValue = [0 for i in range(CLIENT_NUM)]
        for client in range(CLIENT_NUM):
            sv = 0.0
            for subset in allClientsSet:
                if not (client in subset):
                    delta = ACCDICT.get(power2number(subset + [client])) - ACCDICT.get(power2number(subset))
                    sv = sv + delta / scp.comb(CLIENT_NUM - 1, len(subset))
            shapleyValue[client] = float(sv)
        client_shapley_history.append(shapleyValue)

    # Calculate the \phi_i
    CIs = np.zeros(CLIENT_NUM)
    iterNow = 0
    for svs in client_shapley_history:
        iterNow += 1
        svs = normal(svs)
        summary = sum(svs)
        mid = []
        for i in svs:
            x = i / summary
            mid.append(x * (decay ** iterNow))
        CIs = np.add(CIs, mid)
    print("Shapley Value of each Client:", CIs)


def WB_SHAP(CLIENT_NUM):
    Sets = buildPowerSets(list(range(CLIENT_NUM)))
    np.random.shuffle(Sets)
    allClientsSet = sorted(Sets, key=len)
    ACCDICT = dict()
    shapleyValue = [0 for i in range(CLIENT_NUM)]

    subsetSize = 0
    ConvergenCecriteria = 0.04
    SetsOfSize = []
    roundNum = 0
    svChange = [0.0] * 100
    # changes = [[] for i in range(CLIENT_NUM)]
    for subset in allClientsSet:
        if subsetSize != len(subset) and subsetSize > 0:
            print("start calculate!")
            # 用SetsOfSize的记录计算一次各方Shapley value
            totalChange = 0.0
            end_flag = 0
            num = 0
            for ClientsSet in SetsOfSize:
                num += 1
                change = 0.0
                setAcc = FedML(ClientsSet).numpy()
                ACCDICT.update({power2number(ClientsSet): setAcc})
                for client in ClientsSet:
                    subsetWithoutClient = copy.deepcopy(ClientsSet)
                    subsetWithoutClient.remove(client)
                    delta = setAcc - ACCDICT.get(power2number(subsetWithoutClient))
                    delta = delta / scp.comb(CLIENT_NUM - 1, len(ClientsSet) - 1)
                    change += delta
                    shapleyValue[client] = shapleyValue[client] + float(delta)
                    # changes[client].append(float(delta))
                svChange.pop(0)
                svChange.append(change)
                totalChange += change
                # print("changes:", changes)
                if num > 10 and sum(svChange) < ConvergenCecriteria:
                    print(sum(svChange))
                    end_flag = 1
                    break
            SetsOfSize = []
            print("Shapley Value of each Client:", shapleyValue)
            if totalChange < ConvergenCecriteria or end_flag or roundNum > 1000:
                break
        subsetSize = len(subset)
        roundNum += 1
        SetsOfSize.append(subset)
    print("Final Shapley Value of each Client:", shapleyValue)


def shapleyCompute(ShapType):
    print("ShapType", ShapType)
    if ShapType == "DEF":
        DEF_SHAP(CLIENT_NUM)
    elif ShapType == "OR":
        OR_SHAP(CLIENT_NUM)
    elif ShapType == "MR":
        MR_SHAP(CLIENT_NUM)
    elif ShapType == "WB":
        WB_SHAP(CLIENT_NUM)
    elif ShapType == "TMC":
        TMC_SHAP(CLIENT_NUM)
    elif ShapType == "LOO":
        LOO(CLIENT_NUM)
    elif ShapType == "GTB":
        GTB_SHAP(CLIENT_NUM)
    else:
        exit(0)


if __name__ == "__main__":
    expType = sys.argv[1] # Select the  experiment Type, e.g., SAME, VaryDistr, NoiseX, NoiseY
    shapType = sys.argv[2] # Select the Algorithms, e.g., Def, MR, OR, TMC, GTB ...
    testX, testY, trainBatchData = loader(expType) # load the input data
    # import cProfile as cpf
    a_time = time.process_time()
    shapleyCompute(shapType)
    b_time = time.process_time() # compute the running time of data valuation.
    print("SHAP_Time:", b_time - a_time)


