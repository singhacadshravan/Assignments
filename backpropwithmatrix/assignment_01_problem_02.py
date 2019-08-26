#!/usr/bin/env python
# coding: utf-8

# <h4 align="right">by <a href="http://cse.iitkgp.ac.in/~adas/">Abir Das</a> with help of <br> Ram Rakesh and Ankit Singh<br> </h4>

# ### Write the following details here
# ** Name: ** `<shravan kumar singh>`<br/>
# ** Roll Number: ** `<18cs92r07>`<br/>
# ** Department: ** `<computer science and engineering>`<br/>
# ** Email: ** `<shan.icwa@gmail.com>`

# # Problem Set 2

# ## Preamble
# 
# To run and solve this assignment, one must have a working IPython Notebook installation. The easiest way to set it up for both Windows and Linux is to install [Anaconda](https://www.continuum.io/downloads). Then save this file ([`assignment_01.ipynb`]()) to your computer, run Anaconda and choose this file in Anaconda's file explorer. Use `Python 3` version. Below statements assume that you have already followed these instructions. If you are new to Python or its scientific library, Numpy, there are some nice tutorials [here](https://www.learnpython.org/) and [here](http://www.scipy-lectures.org/).

# ### Problem: You will implement a fully connected neural network from scratch in this problem
# We marked places where you are expected to add/change your own code with **`##### write your code below #####`** comment.

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
'''You are not supposed to import any other python library to work on this assignments.'''


# In[2]:


'''data is loaded from data directory.
please don't remove the folder '''

x_train = np.load('./data/X_train.npy')
x_train = x_train.flatten().reshape(-1,28*28)
x_train = x_train / 255.0
gt_indices = np.load('./data/y_train.npy')
train_length = len(x_train)
print("Number of training examples: {:d}".format(train_length))


# In[3]:


'''Dimensions to be used for creating your model'''

batch_size = 64  # batch size
input_dim = 784  # input dimension
hidden_1_dim = 512  # hidden layer 1 dimension
hidden_2_dim = 256  # hidden layer 2 dimension
output_dim = 10   # output dimension

'''Other hyperparameters'''
learning_rate = 1e-5


# In[4]:


#creating one hot vector representation of output classification
y_train = np.zeros((train_length, output_dim))
# print(y.shape, gt_indices.shape)
for i in range(train_length):
    y_train[i,gt_indices[i]] = 1

# Number of mini-batches (as integer) in one epoch
num_minibatches = np.floor(train_length/batch_size).astype(int) #937


# In[5]:


print("No of mini-batches {:d} and total training data used in training:{}.".format(num_minibatches, num_minibatches*batch_size))


# In[6]:


'''Randomly Initialize Weights  from standard normal distribution (i.e., mean = 0 and s.d. = 1.0).
Use the dimesnions specified in the cell 3 to initialize your weights matrices. 
Use the nomenclature W1,W2 etc. (provided below) for the different weight matrices.'''

########################## write your code below ##############################################
W1 = np.random.normal(0,1.0,(input_dim,hidden_1_dim))
W2 = np.random.normal(0,1.0,(hidden_1_dim,hidden_2_dim))
W3 = np.random.normal(0,1.0,(hidden_2_dim,output_dim))
###############################################################################################


# In[7]:


# Write a function which computes the softmax where X is vector of scores computed during forward pass
def softmax1(x):
    ##############################write your code here #################################
    maxi=np.max(x,axis=0)
    x=x - maxi#np.expand_dims(np.max(expx, axis = axis), axis)
    print(maxi)
    expx=np.exp(x)    
    sumx=np.sum(expx,axis=0)    
    print(sumx)    
    return expx/sumx    
    ####################################################################################
    pass
def softmax2(x):
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
    ####################################################################################
    pass
def softmax(y):
    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = 1), 1)
    # exponentiate y
    y = np.exp(y)
    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = 1), 1)
    # finally: divide elementwise
    p = y / ax_sum
    #print(np.sum(p,axis=1))    
    return p

# In[8]:


no_of_iterations =  2
loss_list=[]
i_epoch = 0
for i_iter in range(no_of_iterations):
    
    
    batch_elem_idx = i_iter%num_minibatches
    x_batchinput = x_train[batch_elem_idx*batch_size:(batch_elem_idx+1)*batch_size]
    #print(x_batchinput.shape)
    ########################## write your code below ##############################################
    ######################### Forward Pass Block #####################################
    '''Write the code for forward block of the neural network with 2 hidden layers.
    Please stick to the notation below which follows the notation provided in the lecture slides.
    Note that you are allowed to write the right hand sides of these variables in more than
    one line if that is convenient for you.'''
    # first hidden layer implementation
    a1 =np.dot(x_batchinput,W1) # (64*784)*(784 * 512)
    print(x_batchinput.shape,W1.shape,a1.shape)
    # implement Relu layer
    h1 = np.where(a1>0,a1,0) # 64*512
    #  implement 2 hidden layer
    a2 = np.dot(h1,W2) #64*512 512*256
    print(h1.shape,W2.shape,a2.shape)
    # implement Relu activation 
    h2 = np.where(a2>0,a2,0) # 64*256
    #implement linear output layer
    a3 = np.dot(h2,W3) #64*256 256*10
    # softmax layer
    softmax_score = softmax(a3) # 64*10
    #print(softmax_score[0:3,])
    #enusre you have implemented the softmax function defined above
    ##################################################################################
    ###############################################################################################

    neg_log_softmax_score = -np.log(softmax_score+0.00000001) # The small number is added to avoid 0 input to log function
    #print(neg_log_softmax_score)
    # Compute and print loss
    if i_iter%num_minibatches == 0:
        loss = np.mean(np.diag(np.take(neg_log_softmax_score, gt_indices[batch_elem_idx*batch_size:(batch_elem_idx+1)*batch_size],axis=1)))
        #print(np.take(neg_log_softmax_score, gt_indices[batch_elem_idx*batch_size:(batch_elem_idx+1)*batch_size],axis=0))
        print(" Epoch: {:d}, iteration: {:d}, Loss: {:6.4f} ".format(i_epoch, i_iter, loss))
        loss_list.append(loss)
        i_epoch += 1
        # Each 10th epoch reduce learning rate by a factor of 10
        if i_epoch%10 == 0:
            learning_rate /= 10.0
    
    ################################### Backpropagation Code Block #####################################
    ''' Use the convention grad_{} for computing the gradients.
    for e.g 
        grad_W1 for gradients w.r.t. weight W1
        grad_w2 for gradients w.r.t. weights W2'''
    ########################## write your code below ##############################################
    # Gradient of cross-entropy loss w.r.t. preactivation of the output layer
    grad_softmax_score = softmax_score-y_train[batch_elem_idx*batch_size:(batch_elem_idx+1)*batch_size] #64*10 #10*64
    
    # gradient w.r.t W3
    grad_W3 = 1/batch_size*(h2.T).dot(grad_softmax_score) #(64*256)' 64*10
    # gradient w.r.t h2
    grad_h2 = np.where(h2>0,1,0)
    # gradient w.r.t a2
    grad_a2 = np.multiply(grad_softmax_score.dot(grad_W3.T) ,grad_h2)
    # gradient w.r.t W2
    grad_W2 = 1/batch_size*(h1.T).dot(grad_a2)
    # gradient w.r.t h1
    grad_h1 = np.where(h1>0,1,0)
    # gradient w.r.t a1
    grad_a1 = np.multiply(grad_a2.dot(grad_W2.T) ,grad_h1)
    # gradient w.r.t W1
    grad_W1 = 1/batch_size*(x_batchinput.T).dot(grad_a1)
    ###############################################################################################
    ####################################################################################################
    
    #print(np.sum(W3),np.sum(W2))
    ################################ Update Weights Block using SGD ####################################
    W3 -= learning_rate * grad_W3
    W2 -= learning_rate * grad_W2
    W1 -= learning_rate * grad_W1
    ####################################################################################################
   
#plotting the loss
plt.figure(figsize=(10,5))
plt.plot(loss_list)
plt.title('Loss vs epochs')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.show()
