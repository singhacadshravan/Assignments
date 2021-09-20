import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
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
    
    for row in csv_f:        
        status=0
        if len(elements)==0:
            elements.append([]) # Add elements to empty lists.
            elements[counter].append(row[0])
            elements[counter].append(int(row[1]))
            counter+=1
        else:
            for r in elements:
                if r[0]==row[0] :
                    r[1]=int(r[1])+int(row[1]) #need to check if changing in r will change in elements
                    status=1
            if status==0: # if already not added then add new client
                elements.append([])
                elements[counter].append(row[0])
                elements[counter].append(int(row[1]))        
                counter+=1    
    print elements
    client = [each_list[0] for each_list in elements] #converting a column of a list into a list
    length = [int(each_list[1]) for each_list in elements]
    print length
    y_pos = np.arange(len(client))
    plt.bar(y_pos, length, align='center',width=.3, alpha=0.5)
    plt.xticks(y_pos, client)
    plt.ylabel('length')
    plt.title('Client message traffic')
 
    plt.show()

fetchdata()
