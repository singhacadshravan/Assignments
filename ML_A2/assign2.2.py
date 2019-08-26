# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 15:29:38 2019

@author: shravan
"""
import _pickle as pickle
import math
import pandas as pd
def ginisplit(dataf,col):
    ginfo=0.0
    if(len(dataf[col].unique()) >1):
        for x in dataf[col].unique():
            ginfo+=len(dataf[dataf[col]==x])/len(dataf.index)*gini(dataf[dataf[col]==x])
    else:
        ginfo=0.5
    return ginfo
def gini(dataf):
    return (1- (len(dataf[dataf['label']==1])/len(dataf))**2 - (len(dataf[dataf['label']==2])/len(dataf))**2)
def callgini(dataf):
    top=list()
    for col in dataf.columns[:-1]:
        top.append(ginisplit(dataf,col))
    print(top)
    a,b=min(enumerate(top), key=lambda x: x[1]) #selecting max index and value
    print(dataf.columns[a])
    return dataf.columns[a]               
    pass
def entropy(i,j):
    if(i==0 or j==0):
        return 0;#all belong to same class
    else:
        return i/(i+j)*math.log((i+j)/i,2)+j/(i+j)*math.log((i+j)/j,2)
def IG(dataf):
    temp=dataf['label'].value_counts()
    #print(temp)
    #print(temp[0:1].item())
    return entropy(temp[0:1].item() ,temp[1:2].item() if len(temp)>1 else 0)
def callIG(dataf):
    top=list()
    for col in dataf.columns[:-1]:
        iginfo=0.0
        temp=dataf[col].unique()
        if(len(temp) >1):
            for x in temp:
                tempdf=dataf[dataf[col]==x]
                iginfo+= len(tempdf.index)/len(dataf[col].index)*IG(tempdf)
        else:
            iginfo=1
        temp=dataf['label'].value_counts()
        #print(temp[0:1])
        iginfo=entropy(temp[0:1].item() ,temp[1:2].item() if len(temp)>1 else 0)-iginfo
        top.append(iginfo)
    a,b=max(enumerate(top), key=lambda x: x[1]) #selecting max index and value
    #print(dataf.columns[a])
    return dataf.columns[a]
    pass

def bestattrib(dataf,choice):
    if choice==1:
        return callgini(dataf)
    elif choice ==2:
        return callIG(dataf)
#[col,list of childs,class]
def buildtree(dataf,depth):
    tree=list()
    #print(dataf['label'].unique())    
    if(len(dataf['label'].unique()) >1 and depth>0): # only if more than one unique value in label, otherwise same class            
        col=bestattrib(dataf,2)
        print(col)
        if(len(dataf[col].unique()) >1): # only if more than one unique value in label, otherwise same class                
            tree.append(col)        
            tree.append([])        
            for x in dataf[col].unique():
                tree[1].append([x,buildtree(dataf[dataf[col]==x],depth-1)]) # column value and child node, for prediction
            dataf=dataf.drop(col,1)    
        else:
            tree.append('end')        
            tree.append([])        
            temp=dataf['label'].value_counts()
            if (1 in temp.index) and (2 in temp.index):
                tree.append( 1 if temp[1]>temp[2] else 2)#majority class
            else:
                if (1 in temp.index):
                    tree.append( 1 )#majority class
                else:
                    tree.append( 2 )
    else:
        tree.append('end')
        tree.append([])
        temp=dataf['label'].value_counts()
        if (1 in temp.index) and (2 in temp.index):
                tree.append( 1 if temp[1]>temp[2] else 2)#majority class
        else:
            if (1 in temp.index):
                tree.append( 1 )#majority class
            else:
                tree.append( 2 )
    return tree
#depth first traversal of tree
def printtree(tree,depth):    
    if(len(tree[1])>0):
        for x in tree[1]:
            print(depth*' '+tree[0]+" = "+str(x[0]))
            printtree(x[1],depth+1)
    else:
        print(depth*' '+tree[2])
def recurs(tree,row):
    if(len(tree[1])>0):
        for x in tree[1]:
            if(x[0]==row[tree[0]]):
                return recurs(x[1],row)                
    else:
        return tree[2]
    
def predict(tree,df):    
    return [recurs(tree,row) for index,row in df.iterrows()]    
    pass
    
def accuracy(y,y_hat):
    tot=0
    for a,b in zip(y,y_hat):
        if a==b:
            tot+=1
    print(float(tot)/len(y))
    return float(tot)/len(y)        
    pass
data=[[0 for j in range(3567)] for i in range(1061)] # one extra for label
with open("./traindata.txt",'r') as f:
    for x in f.readlines():
        #print(int(x.split('\t')[0]),int(x.split('\t')[1]))
        data[int(x.split('\t')[0])-1][int(x.split('\t')[1])-1]=1
with open("./trainlabel.txt",'r') as f:    
    for i,x in enumerate(f.readlines()):
        data[i][3566]=int(x)
df=pd.DataFrame.from_records(data,columns=[i for i in range(3567)])        
df.rename(columns={3566:'label'},inplace=True)

num_lines = sum(1 for line in open('./testlabel.txt'))
data1=[[0 for j in range(3566)] for i in range(num_lines)] # one extra for label
with open("./testdata.txt",'r') as f:
    for x in f.readlines():
        data1[int(x.split('\t')[0])-1][int(x.split('\t')[1])-1]=1
dftest=pd.DataFrame.from_records(data1,columns=[i for i in range(3566)])        
with open("./testlabel.txt",'r') as f:
    ytrue=[x for x in f.readlines()]

for depth in [5,10,15]:
    tree=buildtree(df,depth)
    with open("tree"+str(depth)+".bin","wb") as f:
        pickle.dump(tree,f)    
    with open("tree"+str(depth)+".bin","rb") as f:
        tree=pickle.load(f)    
    
    y_hat=predict(tree,df) 
    print("accuracy training:")   
    accuracy(df['label'],y_hat)
    y_hat=predict(tree,dftest) 
    print("accuracy test:")       
    accuracy(ytrue,y_hat)
 
########################using scikit package######################
from sklearn.tree import DecisionTreeClassifier  
from sklearn.metrics import accuracy_score
classifier = DecisionTreeClassifier(criterion='entropy')  
classifier.fit(df.loc[:,df.columns!='label'], df['label'])  
y_pred = classifier.predict(dftest)  
print(y_pred)
print(accuracy_score(ytrue,y_pred))
#print(classifier.tree_.node_count)
#print(classifier.tree_.impurity[0])