# get the file name to pickle dump
import os 
import pickle as pk

def get_rec_file_name(model_name, client_num, dataset_name):
    return model_name + "_" + str(client_num) +"_" + dataset_name + ".rec"



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


def rec_grad(pre_weights, now_weights, cid, round, now_args):
    file_path = "./rec_fed_grad/" + get_rec_file_name(now_args.model, now_args.client_num, now_args.dataset)[:-4] + "/"
    if not os.path.exists(file_path): os.mkdir(file_path)
    file_name = str(cid)+ "_" +str(round)+".grad_rec"
    grad = [now-pre for (pre, now) in zip(pre_weights, now_weights)]

    # save the data in pickle form 
    with open(file_path+file_name,'wb') as fout:
        pk.dump(grad, fout)
    print("SAVE:", file_path+file_name)


def load_grad(cid, round, now_args):
    file_path = "./rec_fed_grad/" + get_rec_file_name(now_args.model, now_args.client_num, now_args.dataset)[:-4] + "/"
    file_name = str(cid)+ "_" +str(round)+".grad_rec"
    return pk.load(file_path+file_name)


    
    # file_name_rec_time = "./rec_fed_sample_time/" + get_rec_file_name(args.model, args.client_num, args.dataset)
    # now_rec = pk.load(file_name_rec_time)
    # now_rec[str(power2number(args.rec_sample))] = end_time - begin_time
    # with open(file_name_rec_time, 'wb') as fout:
    #     pk.dump(now_rec, fout)

def rec_sample_time(args, accuarcy, loss, time_slot, output_flag=True): 
    file_name_rec_time = "./rec_fed_sample_time/" + get_rec_file_name(args.model, args.client_num, args.dataset)

    if not os.path.exists(file_name_rec_time):
        print("now we create the " + file_name_rec_time)
        with open(file_name_rec_time, 'wb') as fout: 
            blank_dict = {}
            pk.dump(blank_dict, fout)

    with open(file_name_rec_time, "rb") as fin:
        now_rec = pk.load(fin)
    now_rec[str(power2number(args.rec_sample))] = {'loss':loss, 'acc': accuarcy, 'time': time_slot}
    if output_flag:
        print(now_rec)
    with open(file_name_rec_time, 'wb') as fout:
        pk.dump(now_rec, fout)

def show_rec_sample_results(file_name):
    with open(file_name, "rb") as fin:
        now_rec = pk.load(fin)
    print(now_rec)
