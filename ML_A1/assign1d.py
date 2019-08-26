# -*- coding: utf-8 -*-
"""Created on Wed Feb 13 22:39:37 2019
@author: shravan
"""
import numpy as np
import random
import math
import matplotlib.pyplot as plt
def regression(n,alpha,trainx,trainy,testx,testy,power,mod):
    minerror,bestn,bestiter=10000,0,0    
    allw=list()
    trainerror=list()
    testerror=list()
    for i in range(1,n): # for n 1 to 9
        vecx=[[trainx[k]**j for j in range(i+1)] for k in range(len(trainx))] # creating power of x dependin upon n
        vectestx=[[testx[k]**j for j in range(i+1)] for k in range(len(testx))]# same as above for test data
        w=[random.uniform(0,1) for j in range(i+1)] #random weights initially
        count=len(trainx)
        tempw=[None]*(i+1)
        it,epsilon,error0,error=0,1,0,0
        while(epsilon>0.00005 and it <100):#either iteration more than 100 or epsilon value          
            for j in range(i+1): #for different theta(w) value do GDescent
                if(mod==False):# whether to use absolute value cost function
                    delw=power*sum([((np.dot(w,vecx[k])-trainy[k])**(power-1))*(1 if j==0 else vecx[k][j]) for k in range(len(trainx))])/(2*float(count))
                else:# finding the derivative of mod funciton each element of summunation will be evaluated for derivation
                    delw=sum([vecx[k][j] if w[j]> ((np.dot(w,vecx[k])-w[j]*vecx[k][j]-trainy[k])/float(vecx[k][j])) else (-vecx[k][j]) for k in range(len(trainx))])/(2*float(count))                    
                tempw[j]=w[j]-alpha*delw
                #print("new delta w{0} :{1} ".format(j,tempw[j])) 
            w=list(tempw)   #copying temporary thetas value to original weights(thetas) for another iteration 
            error=sum([(np.dot(w,vecx[k])-trainy[k])**power for k in range(len(trainx))])/(2*float(count))
            epsilon=abs(error0-error)
            error0=error
            it+=1
            #print("error in n={0} iteration={1} is={2}:".format(i,it,error))    
        allw.append(w)
        trainerror.append(error)
        testerror.append(sum([(np.dot(w,vectestx[k])-testy[k])**power for k in range(len(testx))])/(2*float(len(testx))))
        if(error<minerror):
            minerror=error
            bestn=i            
            bestiter=it
    print("overall min Train error {0} when n ={1} and iteration={2}".format(minerror,bestn,bestiter))    
    print("Min test error:",testerror[bestn-1])
    return bestn,allw,trainerror,testerror       
n=10 # power of x till where we want the polynomial to be evaluated
m=10 #number of examples
random.seed(123)
x=[random.uniform(0,1) for i in range(m)]
y=[math.sin(x[i]) for i in range(m)]
mu,sigma=0,0.3
noise=np.random.normal(mu,sigma,m)  
y=y+noise
trainx=x[0:int(.8*m)]
testx=x[int(.8*m):m]
trainy=y[0:int(.8*m)]
testy=y[int(.8*m):m]
#calling for different alpha value and power=4 and absolute value cost function
alphavec=[0.025,0.05,0.1,0.2,0.5]        
rmse=list()
rmsem=list()
for alpha in alphavec:
    bestn,allw,trainerror,testerror=regression(n,alpha,trainx,trainy,testx,testy,4,False)        
    vectestx=[[testx[k]**j for j in range(bestn+1)] for k in range(len(testx))]
    rmse.append(math.sqrt(sum([(np.dot(allw[bestn-1],vectestx[k])-testy[k])**2 for k in range(len(testx))])/float(len(testx))))
    bestn,allw,trainerror,testerror=regression(n,alpha,trainx,trainy,testx,testy,1,True)        
    vectestx=[[testx[k]**j for j in range(bestn+1)] for k in range(len(testx))]
    rmsem.append(math.sqrt(sum([(np.dot(allw[bestn-1],vectestx[k])-testy[k])**2 for k in range(len(testx))])/float(len(testx))))
print("minimum RMSE for power=4 cost function:",min(rmse))
print("minimum RMSE for absolute value cost function:",min(rmsem))
f,ax=plt.subplots()
ax.plot(alphavec,rmse,'gs')
ax.set_xlabel("alphas(learning rate)")
ax.set_ylabel("RMSE")
ax.set_title('(predicted-Y)^4 cost function')
plt.savefig("figure4.png",dpi=600)          
plt.clf()
f,ax=plt.subplots()
ax.plot(alphavec,rmsem,'gs')
ax.set_xlabel("alphas(learning rate)")
ax.set_ylabel("RMSE")
ax.set_title('Absolute value cost function')
plt.savefig("figure5.png",dpi=600)          
