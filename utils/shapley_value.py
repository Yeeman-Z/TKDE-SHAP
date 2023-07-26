
# from utils.shapley_value import power2number buildPowerSets
def power2number(itemset):
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
