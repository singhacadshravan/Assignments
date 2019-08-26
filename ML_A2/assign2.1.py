# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 15:29:38 2019

@author: shravan
"""
import math
import pandas as pd
def ginisplit(dataf,col):
    ginfo=0.0
    if(len(dataf[col].unique()) >1):
        for x in dataf[col].unique():
            ginfo+=len(dataf[dataf[col]==x])/len(dataf)*gini(dataf[dataf[col]==x])
    else:
        ginfo=0.5
    return ginfo
def gini(dataf):
    return (1- (len(dataf[dataf['profitable']=="yes"])/len(dataf))**2 - (len(dataf[dataf['profitable']=="no"])/len(dataf))**2)
def callgini(dataf):
    top=list()
    for col in dataf.columns[:-1]:
        top.append(ginisplit(dataf,col))
    print(top)
    a,b=min(enumerate(top), key=lambda x: x[1]) #selecting max index and value
    #print(dataf.columns[a])
    return dataf.columns[a]               
    pass
def entropy(i,j):
    if(i==0 or j==0):
        return 0;#all belong to same class
    else:
        return i/(i+j)*math.log((i+j)/i,2)+j/(i+j)*math.log((i+j)/j,2)
def IG(dataf):
    return entropy(len(dataf[dataf['profitable']=="yes"]),len(dataf[dataf['profitable']=="no"]))
def callIG(dataf):
    top=list()
    for col in dataf.columns[:-1]:
        iginfo=0.0
        if(len(dataf[col].unique()) >1):
            for x in dataf[col].unique():
                iginfo+= len(dataf[dataf[col]==x])/len(dataf[col])*IG(dataf[dataf[col]==x])
        else:
            iginfo=1
        iginfo=entropy(len(dataf[dataf['profitable']=="yes"]),len(dataf[dataf['profitable']=="no"]))-iginfo
        top.append(iginfo)
    a,b=max(enumerate(top), key=lambda x: x[1]) #selecting max index and value
    print(b)
    return dataf.columns[a]
    pass

def bestattrib(dataf,choice):
    if choice==1:
        return callgini(dataf)
    elif choice ==2:
        return callIG(dataf)
#[col,list of childs,class]
def buildtree(dataf):
    tree=list()
    #print(dataf['profitable'].unique())
    if(len(dataf['profitable'].unique()) >1): # only if more than one unique value in label, otherwise same class            
        col=bestattrib(dataf,1)
        #print(col)
        #print(dataf[col].unique())
        if(len(dataf[col].unique()) >1): # only if more than one unique value in label, otherwise same class                
            tree.append(col)        
            tree.append([])        
            for x in dataf[col].unique():
                tree[1].append([x,buildtree(dataf[dataf[col]==x])]) # column value and child node, for prediction
        else:
            tree.append('end')        
            tree.append([])        
            tree.append( 'yes' if len(dataf[dataf['profitable']=='yes'])>len(dataf[dataf['profitable']=='no']) else 'no')#majority class
    else:
        tree.append('end')
        tree.append([])
        tree.append( 'yes' if len(dataf[dataf['profitable']=='yes'])>len(dataf[dataf['profitable']=='no']) else 'no')#majority class
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
xlfile = pd.ExcelFile('./dataset for part 1.xlsx')
df = xlfile.parse('Training Data')
#print(df)
tree=buildtree(df)
printtree(tree,0)
y_hat=predict(tree,df) 
#print(y_hat)   
accuracy(df['profitable'],y_hat)
dftest = xlfile.parse('Test Data')
y_hat=predict(tree,dftest)
accuracy(dftest['profitable'],y_hat)
print(y_hat)

########################using scikit package######################
from sklearn.tree import DecisionTreeClassifier  
from sklearn.metrics import accuracy_score
classifier = DecisionTreeClassifier(criterion='entropy')  
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
for x in df.columns:
    df[x]=le.fit_transform( df[x])
    dftest[x]=le.transform( dftest[x])
#print(df)
classifier.fit(df.loc[:,df.columns!='profitable'], df['profitable'])  
y_pred = classifier.predict(dftest.loc[:,dftest.columns!='profitable'])  
print(y_pred)
print(accuracy_score(dftest['profitable'],y_pred))
#print(classifier.tree_.node_count)
#print(classifier.tree_.impurity[0])