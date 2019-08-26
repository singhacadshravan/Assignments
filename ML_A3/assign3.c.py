import math
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

graph=build_network(ls,0.01)
cluster1=girvannewman(graph)
for i,x in enumerate(cluster1):
    print("cluster no: "+str(i+1)+" : number of elements in cluster is "+str(len(x)))
    for j in x:
        print(domain[j]),
    print("\n")    

dic=dict()
ls=list()
cluster=dict()
domain=list()
dom=dict()
with open('AAAI.csv','r',encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    label=0
    for i,row in enumerate(csv_reader):
        if i != 0:
            ele=row[2].split('\n')
            #print(ele)
            ls.append(set())
            cluster[i-1]=[i-1]
            domain.append(row[3])
            if(row[3] not in dom):
                dom[row[3]]=list()
            dom[row[3]].append(i-1)
            for e in ele:
                if(e not in dic):
                    dic[e]=label
                    label+=1
                ls[i-1].add(dic[e])                    
mat=list()
def initialise(ls,linkage):
    for i in range(len(ls)):
        mat.append([])
        for j in range(len(ls)):
            sim=len(ls[i].intersection(ls[j]))/float(len(ls[i].union(ls[j])))
            sim=(2 if linkage==1 and sim==0 else sim)
            mat[i].append(sim)

def hiercluster(mat,linkage):#linkage 1=complete, 0 is single    
    while(len(cluster)>9):
        mini,minj=0,0
        maxsim=(0 if linkage==0 else 2)
        for i,x in enumerate(mat):
            for j,y in enumerate(x):
                if(linkage==0):
                    if(maxsim<y and i!=j):
                        maxsim=y
                        mini=i
                        minj=j
                else:
                    if(maxsim>y and i!=j):
                        maxsim=y
                        mini=i
                        minj=j    
        #print(mini,minj,maxsim)
        if(mini>minj):
            small,large=minj,mini
        else:
            small,large=mini,minj
        for j in range(len(mat)):
            if(linkage==0):
                mat[small][j]=max(mat[small][j],mat[large][j])
                mat[large][j]=0
            else:
                mat[small][j]=min(mat[small][j],mat[large][j])# for complete linkage
                mat[large][j]=2                
        for j in range(len(mat)):
            if(linkage==0):
                mat[j][large]=0                
            else:
                mat[j][large]=2
        
        #print(small,large,mat[small][large])
        #print(cluster[large])
        cluster[small].extend(cluster[large])
        del cluster[large]        
        

linkage=1
initialise(ls,linkage)
#print(mat)
hiercluster(mat,linkage)        
for i,x in enumerate(cluster.keys()):
    print("cluster no: "+str(i+1)+" : number of elements in cluster is "+str(len(cluster[x])))
    for j in cluster[x]:
        print(domain[j]),
    print("\n")    
cluster=[cluster[x] for x in cluster]
#print(cluster)  
                                      
def findNMI(ground,predicted):
    Wk = 0. #elements in cluster
    Cj=0. # elements in grouund community(class)
    N=len(ground) # total elements in predicted clusters
    C=N # total elements in ground community(class)
    Hp=0. #entropy of predicted
    Hc=0. #entropy of ground    
    stat=0
    lsnmi=list()
    if N > 0:        
        #for each predicted community we check in ground community
        for com1 in predicted:
            NMI=0.    
            Wk=len(com1)
            Hp=Hp+Wk/N*math.log(Wk/float(N),2) # entropy of predicted
            for com2 in ground:            
                Cj=len(com2)
                com3=[val for val in com1 if val in com2] #finding intersection of predicted and ground community                
                if len(com3)==0: # when log of zero
                    val=0
                else:
                    val=len(com3)/float(Cj)*math.log(len(com3)/float(Cj) ,2)
                NMI= val + NMI
                if stat ==0: # calculate only first time the Hc
                    Hc=Hc+Cj/float(C)*math.log(Cj/float(C),2) # entropy of ground
            stat=1
            lsnmi.append(NMI/Wk)            
        Hp=-1*Hp
        Hc=-1*Hc
        NMI=Hc+sum(lsnmi)        
        NMI=2*NMI/(Hp+Hc)
    else :
        NMI=0
    print ('Hp and Hc and NMI: ',Hp,Hc,NMI) 
    return NMI
ground=[dom[x] for x in dom]
#print(ground)
findNMI(ground,cluster)
#findNMI(ground,cluster)
findNMI(ground,cluster1)