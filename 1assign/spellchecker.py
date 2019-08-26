# -*- coding: utf-8 -*-
"""Created on Sun Jan 27 18:19:32 2019
@author: shravan
"""
import jieba
import json
import codecs
import time
import os
import _pickle as pickle
import xml.etree.ElementTree as ET
from xpinyin import Pinyin
root=None #root node of BK tree
gdict=dict()#text:count

class node:
    def __init__(self,text=None,pinyin=None,chain=None):
        self.pinyin=pinyin
        self.text=text
        self.chain=dict()
#returns the editdistance of two words
def geditdistance(a,b): 
    m = len(a)
    n = len(b); 
    dp=[[0]*(n+1) for i in range(m+1)] 
    for i in range(m+1): 
        dp[i][0] = i 
    for j in range(n+1): 
        dp[0][j] = j      
    for i in range(1,m+1): 
        for j in range(1,n+1): 
            if (a[i-1] != b[j-1]): 
                dp[i][j] = min( 1 + dp[i-1][j], 1 + dp[i][j-1],  1 + dp[i-1][j-1]) 
            else:
                dp[i][j] = dp[i-1][j-1]
    return dp[m][n]
 #add node to the BK tree
def addnode(nodeobj,text,pinyin):
    global root
    if(nodeobj==None): #if root node
        gdict[text]=1
        root=node(text,pinyin)
        return
    else:        
        dist=geditdistance(nodeobj.pinyin,pinyin)
        #print("word is {0} and parent is {1} distance is:{2} ".format(pinyin,nodeobj.pinyin,dist))
        #print(nodeobj.chain.keys())
        if(dist==0): #because two different words having same pinyin hence zero distance also
            gdict[text]=1
            #nodeobj.chain[dist]=node(text,pinyin)
            #print("should not have come here",nodeobj.text,text)        
        elif(dist not in nodeobj.chain ): #if new node needs to be added for new path
            gdict[text]=1
            nodeobj.chain[dist]=node(text,pinyin)
        else:# if child do exist at that distance then use that path
            addnode(nodeobj.chain[dist],text,pinyin)            
            
def makedict( worddir,pinyindir ):    
    global gdict
    global root
    total =0
    '''files = os.listdir(worddir)
    p=Pinyin()
    with open("dump.txt",'w',encoding='utf-8') as dump: #for adding later to jieba dictionary
        for j,filename in enumerate(files):  #looping over corpus  
            treew = ET.parse(worddir+"/"+filename)#(worddir+"/LCMC_A.XML")
            print ("Processing corpus for dictionary: "+worddir+"/"+filename)             
            for elem in treew.iter(tag='s'):  
                for x in elem:
                    if(x.tag=='w'):# consider only words
                        total+=1
                        #print(x.tag,x.attrib,x.text)
                        dump.write(x.text+'\n')
                        pinyin=p.get_pinyin(x.text, tone_marks='numbers')
                        #check if it is already in dictionary, and not digits
                        if(x.text not in gdict and not(pinyin.isdigit()) and pinyin!=x.text):
                            addnode(root,x.text,pinyin)
                        elif(pinyin in gdict):#increase the count for finding probablities(unigram model)
                            gdict[x.text]+=1
    with open("bktree.bin","wb") as f, open("gdict.bin","wb") as f1: #for fast loading
        pickle.dump(root,f)
        pickle.dump(gdict,f1)
    '''
    with open("bktree.bin","rb") as f, open("gdict.bin","rb") as f1:
        root=pickle.load(f)
        gdict=pickle.load(f1)                
    print( "total words processed: %i" % total)
    print( "total items in dictionary (corpus words): %i" % len(gdict))
 
    #for searching words in BK Tree    
def searchword(nodeobj,pinyin,correction,toler):
    dist=geditdistance(nodeobj.pinyin,pinyin)
    if(dist<=toler):
        correction.append([nodeobj.text,nodeobj.pinyin,gdict[nodeobj.text]])
        #print(len(correction),nodeobj.text,nodeobj.pinyin)
    #now consider all the paths that in range of(distance-tolerance, distance+tolerance)
    start = dist - toler; 
    if start < 0: 
        start = 1  
    while (start < dist + toler):
        if(start in nodeobj.chain):
            searchword(nodeobj.chain[start],pinyin,correction,toler)
        start+=1    
def firstchoice(correction): #find the word that is most frequent amongst all those who have same edit distance
    return sorted(correction,key=lambda x:x[2],reverse=True)[0][0]
# loop over sentences and does the word segmentation and check the correct spelling of words                
def segmenter(filename):    
    global root
    p=Pinyin()
    global gdict
    print("Processing...., and output is being generated in output.txt in current working directory")
    #jieba.load_userdict('dump.txt')
    count=0
    with codecs.open(filename, "rb") as f,open("output.txt",'w',encoding="utf-8") as out:
        data = json.load(f)
        for i,d in enumerate(data): #each json line is a dictionary here
            seg_list = jieba.cut(d["sentence"],cut_all=True)            
            correctsen=list()
            temp=list()
            temp.append(str(i))
            correctsen.append(str(i))
            for x in seg_list:
                temp.append(x)
                pinyin=p.get_pinyin(x, tone_marks='numbers')
                status=0
                if(x not in gdict and not(pinyin.isdigit()) and pinyin!=x):
                    correction=list()
                    for tol in range(1,tolerance+1):
                        searchword(root,pinyin,correction,tol)
                        #print("list is: ", correction)
                        if(len(correction)>0):# some correction words are in list
                            x=firstchoice(correction)#correction[0][0] #firstchoice(correction)
                            status=1
                            count+=1
                            break                
                correctsen.append(x)
                if(status==1):
                    temp.append("**")
                    correctsen.append("**")
            print(u"original:",u"  ".join(temp),file=out)
            print(u" correct:",u"  ".join(correctsen),file=out)
    print("total error:",count)        
st=time.time()
print("Make sure that corpus LCMC is at desired location: ./2474/2474/Lcmc/data/character")
makedict("./2474/2474/Lcmc/data/character","./2474/2474/Lcmc/data/pinyin")
tottime=time.time()-st
print(tottime)
print("Input the tolerance level(edit distance you want to cover): ")
tolerance=int(input())                      
print("Make sure that sample sentence file is named as ""training-sentences.json"" and is in desired format as provided originally, and placed in current working directory")
st=time.time()
segmenter("./training-sentences.json")            
tottime=time.time()-st
print(tottime)