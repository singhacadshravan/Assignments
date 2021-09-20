import numpy as np
import numpy.random
import matplotlib.pyplot as plt
import csv
import math # it makes the library available

def fetchdata():
    # Create a list.
    elements=[]    
    counter = 0
    #tempset=set()
    f = open('log.csv')
    csv_f = csv.reader(f)
    sender=[]
    receiver=[]
    #find the distinct senders and receivers
    for row in csv_f:        
        status=0
        if row[0] not in sender:
            sender.append(row[0]) # Add elements to empty lists.
        if row[2] not in receiver:
            receiver.append(row[2])         
               

    #creating two dimentional matrix
    h = len(sender)
    w = len(receiver)
    messmatrix = [[0 for x in range(w)] for y in range(h)]   
    f.seek(0)
    for row in csv_f:
        print sender.index(row[0])
        print receiver.index(row[2])
        messmatrix[sender.index(row[0])][receiver.index(row[2])]+=1        
    print messmatrix

    filet = open("TrafficMatrix.csv", "w") 
    filet.write("       ,")
    for r in receiver:
        if(len(receiver)-1==receiver.index(r)):
            filet.write(r+" ")
        else:
            filet.write(r+", ")    
    filet.write("\n")
    for s in sender:
        filet.write(s+",  ")
        for i in range(len(receiver)):
            if(len(receiver)-1==i):
                filet.write(str(messmatrix[sender.index(s)][i])+'')
            else:
                filet.write(str(messmatrix[sender.index(s)][i])+',    ')    
        filet.write("\n")    
    
    filet.close()
        
    ax = plt.subplot(1, 1, 1, aspect = 1)
    ax.pcolor(np.array(messmatrix), cmap=plt.cm.Oranges, alpha=0.7)
    irow = np.arange(len(sender))
    icol = np.arange(len(receiver))    
    ax.set_xticklabels(receiver)
    ax.set_yticklabels(sender)
    plt.xticks(icol + 0.5, rotation=45)
    plt.yticks(irow + 0.5)
    ax.set_xlabel("Receiver")
    ax.set_ylabel("Sender")
    plt.show()
    #plt.savefig("heatmap.png") 
    
fetchdata()	
