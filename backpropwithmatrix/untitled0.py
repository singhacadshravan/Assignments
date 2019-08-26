
'''Loading the test data from data/X_test.npy and data/y_test.npy.'''
x_test = np.load('./data/X_test.npy')
x_test = x_test.flatten().reshape(-1,28*28)
x_test = x_test / 255.0
y_test = np.load('./data/y_test.npy')

batch_size_test = 100 # Deliberately taken 100 so that it divides the test data size
num_minibatches = len(y_test)/batch_size_test
test_correct = 0

'''#Only forward block code and compute softmax_score .'''
for i_iter in range(int(num_minibatches)):
    
    '''Get one minibatch'''
    batch_elem_idx = i_iter%num_minibatches
    x_batchinput = x_test[i_iter*batch_size_test:(i_iter+1)*batch_size_test]
    
    ######### copy only the forward pass block of your code and pass the x_batchinput to it and compute softmax_score ##########
    # first hidden layer implementation
    a1 =np.dot(x_batchinput,W1) # (64*784)*(784 * 512)
    #print(a1)
    # implement Relu layer
    h1 = np.where(a1>0,a1,0) # 512*64
    #print(h1.shape)
    #  implement 2 hidden layer
    a2 = np.dot(h1,W2) #64*512 512*256# (512*256)' * 512*64 
    # implement Relu activation 
    h2 = np.where(a2>0,a2,0) # 256*64
    #implement linear output layer
    a3 = np.dot(h2,W3) #64*256 256*10#(256*10)' * 256*64
    # softmax layer
    #print(a3)
    softmax_score = softmax(a3) #enusre you have implemented the softmax function defined above
    ##################################################################################
    
    y_batchinput = y_test[i_iter*batch_size_test:(i_iter+1)*batch_size_test]
    
    y_pred = np.argmax(softmax_score, axis=1)
    num_correct_i_iter = np.sum(y_pred == y_batchinput)
    test_correct += num_correct_i_iter
print ("Test accuracy is {:4.2f} %".format(test_correct/len(y_test)*100))
