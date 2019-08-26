import csv
dic=dict()
ls=list()
cluster=dict()
domain=list()
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
            for e in ele:
                if(e not in dic):
                    dic[e]=label
                    label+=1
                ls[i-1].add(dic[e])        
            #print(f'\t{row1[0]} works in the {row[1]} department, and was born in {row[2]}.')            
mat=list()
def initialise(ls,linkage):
    for i in range(len(ls)):
        mat.append([])
        for j in range(len(ls)):
            sim=len(ls[i].intersection(ls[j]))/float(len(ls[i].union(ls[j])))
            sim=(2 if linkage==1 and sim==0 else sim)
            mat[i].append(sim)

def hiercluster(mat,linkage):#linkage 1=complete, 0 is single    
    while(len(cluster)>10):
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
        

linkage=0
initialise(ls,linkage)
#print(mat)
hiercluster(mat,linkage)        
for i,x in enumerate(cluster.keys()):
    print("cluster no: "+str(i+1)+" : number of elements in cluster is "+str(len(cluster[x])))
    for j in cluster[x]:
        print(domain[j]),
    print("\n")    
#print(cluster)    
                            