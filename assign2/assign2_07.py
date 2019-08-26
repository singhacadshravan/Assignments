import numpy as np
from sklearn.metrics import f1_score

train=np.array([i.split() for i in open("train.txt","r").read().split("\n")])
refinedtrain=np.array([l for l in train if len(l)!=0]) 
print(refinedtrain[:,1])
testd=np.array([i.split() for i in open("test.txt","r").read().split("\n")])
refinedtest=np.array([l for l in testd if len(l)!=0])
#baseline method
uniqpos=np.unique(refinedtrain[:,1]) #unique pos tags
#a function to calculate majority classifier chunk tag for each pos tag
def maxcount(arr):
    a,b=np.unique(arr,return_counts=True)
    return(a[np.argmax(b)])
tagdict={i:maxcount(refinedtrain[refinedtrain[:,1]==i,2]) for i in uniqpos}
def score(refinedtest):
    return(f1_score(refinedtest[:,2],np.array([tagdict[i[1]] for i in refinedtest]),average="micro"))
f_score=score(refinedtest)
print("test f1 score of baseline method:",f_score)


#prepare training data per sentence i.e. a list of sentences with pos tags and chunk tags
temp=[];trainset=[]
for i in train:
    if(len(i)!=0):
        temp.append(i)
    elif(len(temp)!=0):
        trainset.append(temp)
        temp=[]

temp=[];testset=[]
for i in testd:
    if(len(i)!=0):
        temp.append(i)
    elif(len(temp)!=0):
        testset.append(temp)
        temp=[]

maxlensens=np.max([len(i) for i in trainset])#maximum number of words in a sentence.

vocabf=list(np.unique(np.unique(refinedtest[:,1]).tolist()+np.unique(refinedtrain[:,1]).tolist()))
vocabt=list(np.unique(np.unique(refinedtest[:,2]).tolist()+np.unique(refinedtrain[:,2]).tolist()))
#print(len(vocabf)) #length of vocabulary of all pos tags
#print(len(vocabt)) #length of vocabulary of all chunk tags
dicvocabf={vocabf[i]:i for i in range(len(vocabf))} #a dictionary to convert all pos tags to pos_indices
dicvocabt={vocabt[i]:i for i in range(len(vocabt))} #a dictionary to convert all chunk tags to tag_indices

#selects the most occuring entry in the array as the output
def label_maker(x):
    xz,c=np.unique(x,return_counts=True)
    return(xz[np.argmax(c)])

#Creates the feature vs output dictionary for trigrams and 1-grams
def dict_creator(x):
    ke=np.unique(x[:,0])
    return({k:label_maker(x[x[:,0]==k,1]) for k in ke})

#calculates combination of features as concatenation of those indices
def modifier(x):
    x=np.array(x).T
    #print(x.shape) #2 * words in sentence
    res=[]
    for i in range(len(x[0])):
        if(len(x[0])==1):
            #print(x[0])
            res.append([0,x[0][i],x[1][i]])
        elif(i==0):
            res.append([44*10000+x[0][i]*100+x[0][i+1],x[0][i],x[1][i]])
        elif(i==len(x[0])-1):
            res.append([88*10000+x[0][i-1]*100+x[0][i],x[0][i],x[1][i]])
        else:
            res.append([x[0][i-1]*10000+x[0][i]*100+x[0][i+1],x[0][i],x[1][i]])
    return(res)

#Creates test_labels given the input
def testm(x):
    try:
        return(tridic[x[0]])
    except:
        return(monodic[x[1]])
x_train=[]
for i in trainset:
    xt=[]
    for ii in i: #each word from sentence i
        xt.append([dicvocabf[ii[1]],dicvocabt[ii[2]]]) #postag, chunk tag indices
    #print(xt)
    x_train=x_train+modifier(xt) #modifier called for each sentence
#print(x_train)
x_train=np.array(x_train)
tridic=dict_creator(x_train[:,np.array([0,2])])
monodic=dict_creator(x_train[:,1:])

x_test=[]
for i in testset:
    xt=[]
    for ii in i:
        xt.append([dicvocabf[ii[1]],dicvocabt[ii[2]]])
    x_test=x_test+modifier(xt)
x_test=np.array(x_test)
y=np.array([testm(i) for i in x_test[:,:2]])
print("f1_score of non-baseline method:",f1_score(y_true=x_test[:,2],y_pred=y,average="micro")) #test_f_score of non-baseline method1: 0.9247314097557887
