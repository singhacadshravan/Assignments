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
    nptest=npdata[0:int(len(npdata)/5)]
    nptrain=npdata[int(len(npdata)/5):]
    ytrain=label[int(len(npdata)/5):]
    ytest=label[0:int(len(npdata)/5)]
    #print(len(nptrain),len(nptest)) 
    return nptrain,nptest,np.array(ytrain,dtype=int),np.array(ytest,dtype=int)         
def minibatch(train,label,size):
    index=np.array([random.randint(0,len(train)-1) for i in range(0,size)])
    #print(index)
    return train[index],label[index]
def weightinil():
    w1 = np.random.normal(0,1.0,(input_dim,hidden_1_dim))
    w2 = np.random.normal(0,1.0,(hidden_1_dim,output))
    return w1,w2
def forward(x_batchinput):
    a1 =np.dot(x_batchinput,w1) # (200*7718)*(7718 * 100)
    #print(x_batchinput.shape,w1.shape,a1.shape)
    # implement Relu layer
    h1 = np.where(a1>0,a1,0) # 200*100
    #  implement 2 hidden layer
    a2 = np.dot(h1,w2) #200*100 100*1
    #print(h1.shape,w2.shape,a2.shape)
    # implement Relu activation #not needed here 
    h2 = np.ones(a2.shape)/(np.ones(a2.shape) + np.exp(-a2)) # 200*1    
    #print(a2)
    return a1,h1,h2

def backprop1(output,ytrue,a1,h1,inputt):
    global w1,w2,lrate
    #grad_out1=-ytrue/(output+0.00000001)+(1-ytrue)/(1-output+0.00000001)
    ytrue=ytrue.reshape(ytrue.shape[0],1)
    grad_out=output-ytrue#derivative #grad_out1*(output*(1-output))        
    batch_size=len(ytrue)
    # gradient w.r.t W3
    #print(h1.shape,grad_out.shape)
    grad_W2 = 1/batch_size*(h1.T).dot(grad_out) #(64*256)' 100*1
    # gradient w.r.t h2
    grad_h2 = np.where(a1>0,1,0)
    # gradient w.r.t a2
    grad_a2 = np.multiply(grad_out.dot(grad_W2.T) ,grad_h2)
    # gradient w.r.t W1
    grad_W1 = 1/batch_size*(inputt.T).dot(grad_a2)
    ###############################################################################################
    w2 -= lrate * grad_W2
    w1 -= lrate * grad_W1
    loss=np.sum(-ytrue*np.log(output+0.00000001)-(1-ytrue)*np.log(1-output+0.00000001))/len(ytrue)
    return loss
    
def backprop(output,ytrue,a1,h1,inputt):
    ytrue=ytrue.reshape(ytrue.shape[0],1)
    global w1,w2,lrate    
    grad_w1=np.zeros(w1.shape)
    grad_w2=np.zeros(w2.shape)
    for out,y,h,a,inp in zip(output,ytrue,h1,a1,inputt):
        grad_out1=-y/(out+0.00000001)+(1-y)/(1-out+0.00000001)
        grad_out=grad_out1*(out*(1-out))
        h=h.reshape(h.shape[0],1)
        grad_w2=grad_w2+grad_out*h
        #print("grad_w2:{0}".format(grad_w2.shape))        
        grad_h = np.where(a>0,1,0)
        grad_h=grad_h.reshape(grad_h.shape[0],1)
        #print(w2.shape,grad_h.shape)
        grad_a1=grad_out*(w2*grad_h)
        #print(grad_a1.shape,inp.shape)
        grad_w1=grad_w1+(inp.reshape(inp.shape[0],1)).dot(grad_a1.T)
        #print(grad_w1.shape)        
    grad_w2=grad_w2/len(ytrue)
    grad_w1=grad_w1/len(ytrue)
    w1=w1-lrate*grad_w1
    w2=w2-lrate*grad_w2  
    loss=np.sum(-ytrue*np.log(output+0.00000001)-(1-ytrue)*np.log(1-output+0.00000001))/len(ytrue)
    return loss
lrate=0.1       
train,test,label,ytest=preprocess()
input_dim=len(train[0])
hidden_1_dim=100
output=1
w1,w2=weightinil()
def training():
    loss=0
    for x in [100,200,300,400,500,600,700]:
        print("For {0} Epochs:".format(x))        
        for i in range(0,x):
            mtrain,ytrain=minibatch(train,label,200)
            a1,h1,output=forward(mtrain)
            loss=backprop1(output,ytrain,a1,h1,mtrain)
        print("training loss:{0}".format(loss))    
        testing()
def testing():
    global ytest
    a1,h1,output=forward(test)
    predict=[1 if x>0.5 else 0 for x in output ]
    ytest=ytest.reshape(ytest.shape[0],1)
    #print(ytest.shape,(np.log(output)).shape)
    loss=np.sum(-ytest*np.log(output)-(1-ytest)*np.log(1-output+0.00000001))/len(ytest)
    print("testing loss:{0}".format(loss))
    from sklearn.metrics import accuracy_score
    print("testing Accuracy:{0}".format(accuracy_score(ytest,predict)))
    
    #for i,j in zip(output,ytest):
    #    print(i,j)        
training()
    