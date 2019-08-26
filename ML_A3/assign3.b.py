import networkx as nx
import csv
import operator

dic=dict()
ls=list()
domain=list()
with open('AAAI.csv','r',encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    label=0
    for i,row in enumerate(csv_reader):
        if i != 0:
            ele=row[2].split('\n')
            ls.append(set())
            domain.append(row[3])
            for e in ele:
                if(e not in dic):
                    dic[e]=label
                    label+=1
                ls[i-1].add(dic[e])        

def build_network(ls,threshold):
    G = nx.Graph()
    for i in range(len(ls)):
        for j in range(len(ls)):
            if(i<j):#symmetric, so only one way
                sim=len(ls[i].intersection(ls[j]))/float(len(ls[i].union(ls[j])))
                if sim >=threshold:
                    G.add_edge(i, j)
    return G

def girvannewman(graph):
    while(len(list(nx.connected_component_subgraphs(graph)))<9):
        tempdic=nx.edge_betweenness_centrality(graph)
        e=max(tempdic.items(),key= operator.itemgetter(1))[0]
        #print(e)
        graph.remove_edge(*e) #unpack tupple        
    return sorted(nx.connected_components(graph),key=len,reverse=True)
    #print(gg)    

graph=build_network(ls,0.001)
cluster=girvannewman(graph)
for i,x in enumerate(cluster):
    print("cluster no: "+str(i+1)+" : number of elements in cluster is "+str(len(x)))
    for j in x:
        print(domain[j]),
    print("\n")    
        