import numpy as np
from sklearn.metrics import f1_score
alltrain=np.array([i.split() for i in open("../Downloads/train.txt","r").read().split("\n")]) #read the data
allrefinedtrain=np.array([l for l in alltrain if len(l)!=0]) #delete sentence breaks i.e. lines containing only "\n" which contributed to list of size 0 in traindata

alltest=np.array([i.split() for i in open("../Downloads/test.txt","r").read().split("\n")]) #read the data
allrefinedtest=np.array([l for l in alltest if len(l)!=0]) #delete sentence breaks i.e. lines containing only "\n" which contributed to list of size 0 in testdata
#baseline method
uniqpos=np.unique(allrefinedtrain[:,1]) #list of all unique pos tags

#a function to calculate majority classifier chunk tag for each pos tag
def countmax(arr):
    a,b=np.unique(arr,return_counts=True)
    return(a[np.argmax(b)])
#make a dictionary of chunk tags corresponding to each pos tag
baselinetagsdic={i:countmax(allrefinedtrain[allrefinedtrain[:,1]==i,2]) for i in uniqpos}

#a function to generate test chunk tags and calculate f_score of the model
def test(allrefinedtest):
    return(f1_score(allrefinedtest[:,2],np.array([baselinetagsdic[i[1]] for i in allrefinedtest]),average="micro"))
f_score=test(allrefinedtest)
print("test_f_score of baseline method:",f_score) #test_f_score of baseline method: 0.7729066846782195
#prepare training data per sentence i.e. a list of sentences with pos tags and chunk tags
temp=[];trainset=[]
for i in alltrain:
    if(len(i)!=0):
        temp.append(i)
    elif(len(temp)!=0):
        trainset.append(temp)
        temp=[]

temp=[];testset=[]
for i in alltest:
    if(len(i)!=0):
        temp.append(i)
    elif(len(temp)!=0):
        testset.append(temp)
        temp=[]

maxlensens=np.max([len(i) for i in trainset])#maximum number of words in a sentence.

vocabf=list(np.unique(np.unique(allrefinedtest[:,1]).tolist()+np.unique(allrefinedtrain[:,1]).tolist()))
vocabt=list(np.unique(np.unique(allrefinedtest[:,2]).tolist()+np.unique(allrefinedtrain[:,2]).tolist()))
print(len(vocabf)) #length of vocabulary of all pos tags
print(len(vocabt)) #length of vocabulary of all chunk tags
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
    res=[]
    for i in range(len(x[0])):
        if(len(x[0])==1):
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
    for ii in i:
        xt.append([dicvocabf[ii[1]],dicvocabt[ii[2]]])
    x_train=x_train+modifier(xt)
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
print("test_f_score of non-baseline method1:",f1_score(y_true=x_test[:,2],y_pred=y,average="micro")) #test_f_score of non-baseline method1: 0.9247314097557887





#In the following lines we are converting the traindata from a list of lists to a numpy array
x=[]
for i in trainset:
    xt=[]
    for ii in i:
        xt.append([dicvocabf[ii[1]],dicvocabt[ii[2]]])
    for j in range((maxlensens-len(xt))):
        xt.append([len(vocabf)+1,len(vocabt)+1])
    x.append(xt)
x=np.array(x)

#training a Bidirectional LSTM based model to predict final chunk tags

from keras.models import Sequential
from keras.layers import Bidirectional,LSTM,Dense,Activation,Embedding,TimeDistributed
import keras

model = Sequential()
model.add(Embedding(input_dim=len(vocabf)+2,output_dim=10,input_shape=(maxlensens,)))
model.add(Bidirectional(LSTM(10, return_sequences=True)))
model.add(Bidirectional(LSTM(10,return_sequences=True)))
model.add(TimeDistributed(Dense(len(vocabt)+2)))
model.add(Activation('softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
model.summary() #print the architecture of our model
model.fit(x[:,:,0],x[:,:,1:],validation_split=0.2,epochs=50)




x=[]
for i in testset:
    xt=[]
    for ii in i:
        xt.append([dicvocabf[ii[1]],dicvocabt[ii[2]]])
    for j in range((maxlensens-len(xt))):
        xt.append([len(vocabf)+1,len(vocabt)+1])
    x.append(xt)
x=np.array(x)


ypred=model.predict(x[:,:,0])

ans=np.array([[np.argmax(yyy) for yyy in yy] for yy in ypred])
print("test_f_score of the LSTM model:",f1_score(x[:,:,1].flatten()[x[:,:,1].flatten()!=24],ans.flatten()[x[:,:,1].flatten()!=24],average="micro")) #test_f_score of the LSTM model: 0.9304514848977352

'''
test_f_score of baseline method: 0.7729066846782195
44
23
test_f_score of non-baseline method1: 0.9247314097557887
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_3 (Embedding)      (None, 78, 10)            460       
_________________________________________________________________
bidirectional_5 (Bidirection (None, 78, 20)            1680      
_________________________________________________________________
bidirectional_6 (Bidirection (None, 78, 20)            2480      
_________________________________________________________________
time_distributed_3 (TimeDist (None, 78, 25)            525       
_________________________________________________________________
activation_3 (Activation)    (None, 78, 25)            0         
=================================================================
Total params: 5,145
Trainable params: 5,145
Non-trainable params: 0
_________________________________________________________________
Train on 7148 samples, validate on 1788 samples
Epoch 1/50
7148/7148 [==============================] - 37s 5ms/step - loss: 1.2458 - val_loss: 0.6025
Epoch 2/50
7148/7148 [==============================] - 35s 5ms/step - loss: 0.5747 - val_loss: 0.5576
Epoch 3/50
7148/7148 [==============================] - 35s 5ms/step - loss: 0.5516 - val_loss: 0.5440
Epoch 4/50
7148/7148 [==============================] - 35s 5ms/step - loss: 0.5382 - val_loss: 0.5252
Epoch 5/50
7148/7148 [==============================] - 34s 5ms/step - loss: 0.4806 - val_loss: 0.4073
Epoch 6/50
7148/7148 [==============================] - 34s 5ms/step - loss: 0.3692 - val_loss: 0.3394
Epoch 7/50
7148/7148 [==============================] - 33s 5ms/step - loss: 0.3175 - val_loss: 0.2906
Epoch 8/50
7148/7148 [==============================] - 35s 5ms/step - loss: 0.2590 - val_loss: 0.2219
Epoch 9/50
7148/7148 [==============================] - 37s 5ms/step - loss: 0.1948 - val_loss: 0.1712
Epoch 10/50
7148/7148 [==============================] - 36s 5ms/step - loss: 0.1598 - val_loss: 0.1488
Epoch 11/50
7148/7148 [==============================] - 35s 5ms/step - loss: 0.1422 - val_loss: 0.1364
Epoch 12/50
7148/7148 [==============================] - 35s 5ms/step - loss: 0.1315 - val_loss: 0.1279
Epoch 13/50
7148/7148 [==============================] - 36s 5ms/step - loss: 0.1231 - val_loss: 0.1205
Epoch 14/50
7148/7148 [==============================] - 38s 5ms/step - loss: 0.1160 - val_loss: 0.1142
Epoch 15/50
7148/7148 [==============================] - 39s 5ms/step - loss: 0.1106 - val_loss: 0.1097
Epoch 16/50
7148/7148 [==============================] - 38s 5ms/step - loss: 0.1065 - val_loss: 0.1063
Epoch 17/50
7148/7148 [==============================] - 39s 5ms/step - loss: 0.1032 - val_loss: 0.1030
Epoch 18/50
7148/7148 [==============================] - 38s 5ms/step - loss: 0.1003 - val_loss: 0.1007
Epoch 19/50
7148/7148 [==============================] - 36s 5ms/step - loss: 0.0980 - val_loss: 0.0984
Epoch 20/50
7148/7148 [==============================] - 37s 5ms/step - loss: 0.0959 - val_loss: 0.0967
Epoch 21/50
7148/7148 [==============================] - 38s 5ms/step - loss: 0.0941 - val_loss: 0.0953
Epoch 22/50
7148/7148 [==============================] - 41s 6ms/step - loss: 0.0926 - val_loss: 0.0939
Epoch 23/50
7148/7148 [==============================] - 38s 5ms/step - loss: 0.0913 - val_loss: 0.0925
Epoch 24/50
7148/7148 [==============================] - 36s 5ms/step - loss: 0.0902 - val_loss: 0.0914
Epoch 25/50
7148/7148 [==============================] - 35s 5ms/step - loss: 0.0891 - val_loss: 0.0905
Epoch 26/50
7148/7148 [==============================] - 36s 5ms/step - loss: 0.0882 - val_loss: 0.0895
Epoch 27/50
7148/7148 [==============================] - 37s 5ms/step - loss: 0.0873 - val_loss: 0.0887
Epoch 28/50
7148/7148 [==============================] - 37s 5ms/step - loss: 0.0864 - val_loss: 0.0880
Epoch 29/50
7148/7148 [==============================] - 35s 5ms/step - loss: 0.0856 - val_loss: 0.0873
Epoch 30/50
7148/7148 [==============================] - 37s 5ms/step - loss: 0.0848 - val_loss: 0.0867
Epoch 31/50
7148/7148 [==============================] - 36s 5ms/step - loss: 0.0840 - val_loss: 0.0858
Epoch 32/50
7148/7148 [==============================] - 37s 5ms/step - loss: 0.0834 - val_loss: 0.0856
Epoch 33/50
7148/7148 [==============================] - 38s 5ms/step - loss: 0.0827 - val_loss: 0.0847
Epoch 34/50
7148/7148 [==============================] - 38s 5ms/step - loss: 0.0822 - val_loss: 0.0842
Epoch 35/50
7148/7148 [==============================] - 39s 5ms/step - loss: 0.0816 - val_loss: 0.0838
Epoch 36/50
7148/7148 [==============================] - 38s 5ms/step - loss: 0.0811 - val_loss: 0.0831
Epoch 37/50
7148/7148 [==============================] - 38s 5ms/step - loss: 0.0805 - val_loss: 0.0827
Epoch 38/50
7148/7148 [==============================] - 38s 5ms/step - loss: 0.0800 - val_loss: 0.0822
Epoch 39/50
7148/7148 [==============================] - 38s 5ms/step - loss: 0.0795 - val_loss: 0.0816
Epoch 40/50
7148/7148 [==============================] - 38s 5ms/step - loss: 0.0790 - val_loss: 0.0816
Epoch 41/50
7148/7148 [==============================] - 38s 5ms/step - loss: 0.0785 - val_loss: 0.0810
Epoch 42/50
7148/7148 [==============================] - 38s 5ms/step - loss: 0.0781 - val_loss: 0.0805
Epoch 43/50
7148/7148 [==============================] - 38s 5ms/step - loss: 0.0776 - val_loss: 0.0800
Epoch 44/50
7148/7148 [==============================] - 38s 5ms/step - loss: 0.0771 - val_loss: 0.0796
Epoch 45/50
7148/7148 [==============================] - 38s 5ms/step - loss: 0.0765 - val_loss: 0.0794
Epoch 46/50
7148/7148 [==============================] - 38s 5ms/step - loss: 0.0761 - val_loss: 0.0786
Epoch 47/50
7148/7148 [==============================] - 38s 5ms/step - loss: 0.0755 - val_loss: 0.0783
Epoch 48/50
7148/7148 [==============================] - 37s 5ms/step - loss: 0.0750 - val_loss: 0.0778
Epoch 49/50
7148/7148 [==============================] - 37s 5ms/step - loss: 0.0746 - val_loss: 0.0775
Epoch 50/50
7148/7148 [==============================] - 37s 5ms/step - loss: 0.0741 - val_loss: 0.0771
test_f_score of the LSTM model: 0.9304514848977352'''
