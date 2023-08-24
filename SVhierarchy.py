import matplotlib.pyplot as plt
import networkx as nx
import math
import sys

DEBUG = 0


nodeNum = 4
# print(len(sys.argv))
# print(sys.argv)
if len(sys.argv) >= 2: nodeNum= int(sys.argv[1])



# def PowerSetsBinary(num):
def is_subset(list1:list, list2:list):
    return set(list1).issubset(set(list2))

def L2S(mylist:list):
    return "".join([str(x) for x in mylist])

def creatSubset(num :int, subSize: int):
    #  create Full, Set,
    items = [i for i in range(1, num+1)]
    # res = [[] for i in len(subNum)]
    res, fullSet = [], []
    N = len(items)
    for i in range(2 ** N): 
        combo = []
        for j in range(N): 
            if(i>>j)%2:
                combo.append(items[j])
        fullSet.append(combo)
        # if DEBUG:
        #     print(combo)
    
    for subset in fullSet:
        if len(subset) == subSize:
            res.append(subset)
    # return res + [i for i in range(N+1)]
    return res    
    # for subsetSize in subNum:
        


def SVFigSize(nodeNum):
    height = 2*nodeNum
    # weight = int(math.comb(nodeNum, nodeNum//2))
    weight = int(math.factorial(nodeNum)/(math.factorial(nodeNum//2)**2))

    return (weight, height)
    # if nodeNum < 10:
    #     weight = math.sqrt(2/math.pi/nodeNum)*((nodeNum/2.0/math.e)**nodeNum)



# def createSubset(fullSet, s):



if __name__ == "__main__":

    
    G = nx.DiGraph()

    subsetSize = [i for i in range(nodeNum)]
    subset = [creatSubset(nodeNum, subsetSize[i]) for i in range(nodeNum)] + [[[i for i in range(1,nodeNum+1)]]]


    # print(len(subset), nodeNum)

    lenOfSubset = len(subset)

    # print(subset)

    # subsetSize1, subsetSize2 = 2, 3
    # subSet1 = creatSubset(nodeNum, subsetSize1)
    # subSet2 = creatSubset(nodeNum, subsetSize2)

    if DEBUG:
        for i in range(lenOfSubset):
            print("#%d: "%(i), subset[i], "\n")
        print(is_subset([1,2], [1,2,3]))

    # return 0
    # pass

    if DEBUG:
        print(L2S([1,2,3,4,5]))
    subsetSizesOfG = [len(subset[i]) for i in range(1, lenOfSubset)]
    # subset_color = ["gold", "violet", "darkorange"][:2]
    strSubset = [[L2S(x) for x in subset[i]] for i in range(lenOfSubset)]
    # strSubset1 = [str(x) for x in subSet1]
    # strSubset2 = [str(x) for x in subSet2]
    # strSubset1.sort()
    # strSubset2.sort()
    if DEBUG:
        for i in range(lenOfSubset):
            print('#node-%d#'%(i), strSubset[i],"\n")

        # print(strSubset1,"\n-------------\n", strSubset2)
    for node_num_i in range(1, lenOfSubset):
        for ssi in subset[node_num_i]:
            G.add_node(L2S(ssi), layer=node_num_i)    
    
        

    # for ss1 in subSet1:
    #     G.add_node(str(ss1), layer=0)    
    # for ss2 in subSet2:
    #     G.add_node(str(ss2), layer=1)    
    # G.add_nodes_from(strSubset1, layer=0)
    # G.add_edges_from(strSubset2, layer=1)

    for ssi in range(1, lenOfSubset-1):
        ssj = ssi + 1
        for ss1 in subset[ssi]:
            for ss2 in subset[ssj]:
                if is_subset(ss1, ss2):
                    # print()
        # # G.add_node(str(ss1), level=0)
        # for ss2 in subSet2:
        #     if is_subset(ss1, ss2):
                    if DEBUG:
                        print((L2S(ss1), L2S(ss2)))
                    _weight = 1
                    if (1 in ss2) and not (1 in ss1):
                        # print((L2S(ss1), L2S(ss2)))
                        _weight = 10
                    G.add_edge(L2S(ss1), L2S(ss2), length=1, weight= _weight)


    # for ss1 in subSet1:
        # G.add_node()


    plt.figure(figsize=SVFigSize(nodeNum))

    pos = nx.multipartite_layout(G, subset_key='layer', align="horizontal")
    elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > 5]
    esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= 5] 

    # for v in G.nodes():
    #     print(len(v), v)

    _node_batch_size = 250
    # so^>v<dph8
    nx.draw_networkx_nodes(G, pos, node_size=[max(len(v), 2) * _node_batch_size for v in G.nodes()], node_color="white", node_shape="o", edgecolors="black")
    
    
    # nx.draw_networkx_edges(
    #     G, pos, edgelist=esmall, width=1, alpha=0.2, edge_color="gray")
    # nx.draw_networkx_edges(G, pos, edgelist=elarge, width=2, alpha=0.5, edge_color="red")
    nx.draw_networkx_edges(
        G, pos, edgelist=esmall, width=1, alpha=0.2, edge_color="gray")
    nx.draw_networkx_edges(G, pos, edgelist=elarge, width=1, alpha=0.2, edge_color="gray")

    nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")
    



  



    # nx.draw(G, pos, with_labels=True, node_size=25*nodeNum, node_color='lightblue')

    plt.savefig("./fig/svGraph_n="+str(nodeNum)+".png")

    # nx.write_latex(G, "svGraph_n="+str(nodeNum)+".tex", pos=pos, as_document=True)





