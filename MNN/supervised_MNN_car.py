import keras
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

#import data from the file
from numpy import genfromtxt
data = genfromtxt('auto-mpg.data', skip_header=0, usecols=range(0,8)) 

# create the array of expected output
y_expect = data[:,0:1]

# create the array of input
x_input = data[:,1:8]

# rescale the input
for i in range(0,8):
    x_input[:,i:i+1] = x_input[:,i:i+1]/max(x_input[:,i:i+1])

# network structure
n1 = 7 # size of input layer, equal to the number of input attributes
n2 = 2 # size of hidden layer
n3 = 1 # size of output layer, Equal to the number of output
model = Sequential() # create the network layer by layer
keras.initializers.RandomUniform(minval=-0.1, maxval=0.1) # initialize the weights
model.add(Dense(units=n2, input_dim=n1, activation='sigmoid')) # hidden layer
model.add(Dense(units=n3, activation='linear')) # output layer

# loss function, optimization method, and metrics to evaluate
model.compile(loss='mean_squared_error', optimizer=keras.optimizers.SGD(learning_rate=0.01), metrics=['mean_squared_error'])

# train the model
train_history = model.fit(x=x_input, y=y_expect, validation_split=0.2, epochs=500, batch_size=20, verbose=2)

# plot the metrics to show the training results
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])  
    plt.plot(train_history.history[validation])
    plt.title('Train History', fontsize=20)  
    plt.tick_params(labelsize=16)
    plt.ylabel(train, fontsize=20)  
    plt.xlabel('Epoch', fontsize=20)  
    plt.legend(['Train', 'Validate'], loc='best', fontsize=16)
    plt.show()
    
show_train_history(train_history, 'mean_squared_error', 'val_mean_squared_error')
