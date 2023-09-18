# get the file name to pickle dump
def get_rec_file_name(model_name, client_num, dataset_name):
    return model_name + "_" + str(client_num) +"_" + dataset_name + + ".rec"



# from utils.shapley_value import power2number buildPowerSets
def power2number(itemset):
    if type(itemset) == str:
        itemset = eval(itemset)
    number = 0
    for i in itemset:
        number+= 1<<i
    return number 


def buildPowerSets(itemSet):

    number = len(itemSet)
    allSet = list(range(number))
    pwSet = [[] for i in range (1<<number)]
    for i in range(1<<number):
        # subSet = []
        for j in range(number):
            if (i>>j)%2 == 1:
                pwSet[i].append(itemSet[j])
    return pwSet


def rec_grad(pre_weights, now_weights, cid, now_args):
    pass

