import sys
from nltk.corpus import stopwords
import re
from nltk.stem import PorterStemmer
import numpy as np
import random
def preprocess():
    with open("Assignment_4_data.txt",'r') as f:
        data=[re.findall(r"[\w']+", x) for x in f.readlines()]        
    stwd=set(stopwords.words('english'))
    d=list()
    label=list()
    vocab=dict()
    ps = PorterStemmer()
    for i,x in enumerate(data):
        #print(x[0])
        label.append(1 if x[0]=="ham" else 0)
        d.append([])
        for tok in x[1:]:
            if(len(tok)==1):continue
            if ps.stem(tok) not in vocab:
                vocab[ps.stem(tok)]=len(vocab)
                #print(vocab[ps.stem(tok)])
            if tok not in stwd:
                d[i].append(ps.stem(tok))
    #print(d)
    npdata=np.zeros((len(data),len(vocab)),dtype=int)
    for i,x in enumerate(d):
        for y in x:
            npdata[i,vocab[y]]=1
    nptest=npdata[0:int(len(npdata)/4)]
    nptrain=npdata[int(len(npdata)/4):]
    ytrain=label[int(len(npdata)/4):]
    ytest=label[0:int(len(npdata)/4)]
    #print(len(nptrain),len(nptest)) 
    return nptrain,nptest,np.array(ytrain,dtype=int),np.array(ytest,dtype=int)         
def minibatch(train,label,size):
    index=np.array([random.randint(0,len(train)-1) for i in range(0,size)])
    #print(index)
    return train[index],label[index]
def weightinil(input_dim,hidden_1_dim,hidden_2_dim,output):
    w1 = np.random.normal(0,1.0,(input_dim,hidden_1_dim))
    w2 = np.random.normal(0,1.0,(hidden_1_dim,hidden_2_dim))
    w3 = np.random.normal(0,1.0,(hidden_2_dim,output))
    return w1,w2,w3
def softmax(x):
    ##############################write your code here #################################
    maxi=np.max(x,axis=1)
    maxi=maxi.reshape((x.shape[0], 1))
    x=x - maxi#np.expand_dims(np.max(expx, axis = axis), axis)
    #print(maxi)
    expx=np.exp(x)    
    sumx=np.sum(expx,axis=1)    
    sumx=sumx.reshape((x.shape[0],1))
    #print(sumx)    
    #print(np.sum(expx/sumx,axis=1))
    return expx/sumx    

def forward(x_batchinput):
    a1 =np.dot(x_batchinput,w1) # (200*7718)*(7718 * 100)
    #print(x_batchinput.shape,w1.shape,a1.shape)
    # implement Relu layer
    h1 = np.ones(a1.shape)/(np.ones(a1.shape) + np.exp(-a1)) # 200*100
    #print(h1.shape,(np.exp(-a1)))
    #  implement 2 hidden layer
    a2 = np.dot(h1,w2) #200*100 100*1
    #print(h1.shape,w2.shape,a2.shape)
    # implement Relu activation #not needed here 
    h2 = np.ones(a2.shape)/(np.ones(a2.shape) + np.exp(-a2)) # 200*1    
    #print(a2)
    a3 = np.dot(h2,w3) #200*100 100*1
    #print(h1.shape,w2.shape,a2.shape)
    # implement Relu activation #not needed here 
    softmax_score = softmax(a3) # 64*10     # 200*1    
    
    return a1,h1,h2,a2,softmax_score

def backprop1(output,ytrue,a2,h2,a1,h1,inputt):
    global w1,w2,w3,lrate
    #ytrue=ytrue.reshape(ytrue.shape[0],1)
    grad_out=output-ytrue#derivative of softmax #grad_out1*(output*(1-output))        
    batch_size=ytrue.shape[0]
    # gradient w.r.t W3
    #print(h1.shape,grad_out.shape)
    grad_W3 = 1/batch_size*(h2.T).dot(grad_out) #(64*256)' 100*1
    # gradient w.r.t h2
    grad_h2 = a2*(1-a2)
    # gradient w.r.t a2
    grad_a2 = np.multiply(grad_out.dot(grad_W3.T) ,grad_h2)
    # gradient w.r.t W2
    grad_W2 = 1/batch_size*(h1.T).dot(grad_a2)
    # gradient w.r.t h1
    grad_h1 = a1*(1-a1)
    # gradient w.r.t a1
    grad_a1 = np.multiply(grad_a2.dot(grad_W2.T) ,grad_h1)    
    # gradient w.r.t W1
    grad_W1 = 1/batch_size*(inputt.T).dot(grad_a1)
    ###############################################################################################
    w3 -= lrate * grad_W3
    w2 -= lrate * grad_W2
    w1 -= lrate * grad_W1
    neg_log_softmax_score = -np.log(output+0.00000001) # The small number is added to avoid 0 input to log function
    #print(ytrue.shape,neg_log_softmax_score.shape)
    loss=np.sum(-ytrue*neg_log_softmax_score)/ytrue.shape[0] # ytrue has to be 200*2 and softmax is also 200*2
    #loss=sum(-ytrue*np.log(output+0.00000001)-(1-ytrue)*np.log(1-output+0.00000001))/len(ytrue)
    #print(loss)
    return loss
def changedim(ytrue):
    #print(ytrue.shape)
    temp=np.zeros((ytrue.shape[0],2))
    #print(temp.shape)
    for i,x in enumerate(ytrue):
        if x==0:
            temp[i,0]=1
        else:
            temp[i,1]=1
    return temp        
lrate=0.1       
train,test,label,ytest=preprocess()
input_dim=len(train[0])
#print("Enter neurons in first layer:")
hidden_1_dim=int(sys.argv[1])
#print("Enter neurons in second layer:")
hidden_2_dim=int(sys.argv[2])
output=2
w1,w2,w3=weightinil(input_dim,hidden_1_dim,hidden_2_dim,output)
def training():
    loss=0
    for x in [100,200,300,400,500,600,700]:
        print("For {0} Epochs:".format(x))
        for i in range(0,x):
            mtrain,ytrain=minibatch(train,label,200)
            a1,h1,a2,h2,output=forward(mtrain)
            ytrain=changedim(ytrain)
            loss=backprop1(output,ytrain,a2,h2,a1,h1,mtrain)
        print("training loss:{0}".format(loss))    
        testing()    
def testing():
    global ytest
    a1,h1,a2,h2,output=forward(test)
    predict=[1 if x[1]>0.5 else 0 for x in output ]
    ytest1=changedim(ytest)
    neg_log_softmax_score = -np.log(output+0.00000001) # The small number is added to avoid 0 input to log function
    loss=np.sum(-ytest1*neg_log_softmax_score)/ytest1.shape[0] # ytrue has to be 200*2 and softmax is also 200*2
    print("testing loss:{0}".format(loss))
    from sklearn.metrics import accuracy_score
    print("testing Accuracy:{0}".format(accuracy_score(ytest,predict)))
    
training()