import os
import numpy as np
import collections
import tensorflow as tf
from utils.fedcom import federated_train, DATSETLABEL, DATSETSHAPE


def clearFile(nowSetSize, recordModel=False):
    for client in range(nowSetSize):
        f = open(os.path.join(os.path.dirname(__file__), "weights_" + str(client) + ".txt"), "w")
        f.close()
        f = open(os.path.join(os.path.dirname(__file__), "bias_" + str(client) + ".txt"), "w")
        f.close()
    if recordModel:
        f = open(os.path.join(os.path.dirname(__file__), "gradientplus_models.txt"), "w")
        f.close()


def recordParamsAndLR(local_models, learning_rate, client):
    f = open(os.path.join(os.path.dirname(__file__), "weights_" + str(client) + ".txt"), "a", encoding="utf-8")
    for i in local_models[client]['weights']:
        line = ""
        for j in list(i):
            line += (str(j) + "\t")
        print(line, file=f)
    print("***" + str(learning_rate) + "***\n" + "-" * 50, file=f)
    f.close()
    f = open(os.path.join(os.path.dirname(__file__), "bias_" + str(client) + ".txt"), "a", encoding="utf-8")
    line = ""
    for i in local_models[client]['bias']:
        line += (str(i) + "\t")
    print(line, file=f)
    print("***" + str(learning_rate) + "***\n" + "-" * 50, file=f)
    f.close()


def record_Model(model):
    f = open(os.path.join(os.path.dirname(__file__), "gradientplus_models.txt"), "a")
    g_w = list(model['weights'].numpy().reshape(-1))
    g_b = list(model['bias'].numpy().reshape(-1))
    print(g_w, file=f)
    print(g_b, file=f)
    f.close()


def getGradient(client):
    f = open(os.path.join(os.path.dirname(__file__), "weights_" + str(client) + ".txt"))
    content = f.read()
    g_ = content.split("***\n--------------------------------------------------")
    parm_local = []
    learning_rate_list = []
    for j in range(len(g_) - 1):
        line = g_[j].split("\n")
        if j == 0:
            weights_line = line[0:784]
            learning_rate_list.append(float(line[784].replace("*", "").replace("\n", "")))
        else:
            weights_line = line[1:785]
            learning_rate_list.append(float(line[785].replace("*", "").replace("\n", "")))
        valid_weights_line = []
        for l in weights_line:
            w_list = l.split("\t")
            w_list = w_list[0:len(w_list) - 1]
            w_list = [float(i) for i in w_list]
            valid_weights_line.append(w_list)
        parm_local.append(valid_weights_line)
    f.close()

    f = open(os.path.join(os.path.dirname(__file__), "bias_" + str(client) + ".txt"))
    content = f.read()
    g_ = content.split("***\n--------------------------------------------------")
    bias_local = []
    for j in range(len(g_) - 1):
        line = g_[j].split("\n")
        if j == 0:
            weights_line = line[0]
        else:
            weights_line = line[1]
        b_list = weights_line.split("\t")
        b_list = b_list[0:len(b_list) - 1]
        b_list = [float(i) for i in b_list]
        bias_local.append(b_list)
    f.close()
    ret = {
        'weights': np.asarray(parm_local),
        'bias': np.asarray(bias_local),
        'learning_rate': np.asarray(learning_rate_list)
    }
    # 计算梯度
    gradient_weights_local = []
    gradient_biases_local = []
    learning_rate_local = []
    tfInit = tf.keras.initializers.glorot_normal(seed=42)
    initial_model = collections.OrderedDict(
        weights=tfInit(shape=[DATSETSHAPE, DATSETLABEL]),
        bias=tfInit(shape=[DATSETLABEL, ])
    )
    for i in range(len(ret['learning_rate'])):
        if i == 0:
            gradient_weight = np.divide(np.subtract(initial_model['weights'], ret['weights'][i]),
                                        ret['learning_rate'][i])
            gradient_bias = np.divide(np.subtract(initial_model['bias'], ret['bias'][i]),
                                      ret['learning_rate'][i])
        else:
            gradient_weight = np.divide(np.subtract(ret['weights'][i - 1], ret['weights'][i]),
                                        ret['learning_rate'][i])
            gradient_bias = np.divide(np.subtract(ret['bias'][i - 1], ret['bias'][i]),
                                      ret['learning_rate'][i])
        gradient_weights_local.append(gradient_weight)
        gradient_biases_local.append(gradient_bias)
        learning_rate_local.append(ret['learning_rate'][i])
    return gradient_weights_local, gradient_biases_local, learning_rate_local


def getHistoryModels():
    f = open(os.path.join(os.path.dirname(__file__), "gradientplus_models.txt"))
    lines = f.readlines()
    ret_models = []

    tfInit = tf.keras.initializers.glorot_normal(seed=42)
    initial_model = collections.OrderedDict(
        weights=tfInit(shape=[DATSETSHAPE, DATSETLABEL]),
        bias=tfInit(shape=[DATSETLABEL, ])
    )

    ret_models.append(initial_model)
    temp_model= []
    for i, line in enumerate(lines):
        if i % 2 == 0:
            lis = line.strip().replace("[", "").replace("]", "").split(",")
            lis = [float(i.strip()) for i in lis]
            lis = np.array(lis).reshape([784, 10])
            temp_model = [lis]
        else:
            lis = line.strip().replace("[", "").replace("]", "").split(",")
            lis = [float(i.strip()) for i in lis]
            lis = np.array(lis)
            temp_model.append(lis)
            model = collections.OrderedDict(
                weights=temp_model[0],
                bias=temp_model[1]
            )
            ret_models.append(model)
    f.close()
    return ret_models
