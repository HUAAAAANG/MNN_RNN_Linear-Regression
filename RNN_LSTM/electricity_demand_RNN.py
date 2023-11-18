import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM

step = 5 # number of days considered for predicting next day's electricity demand

# load and prepare the data
data = np.genfromtxt('electricity_demand_data_processed.csv', delimiter=',', skip_header=1, usecols=range(1,10))
data[:,0] = data[:,0]/100000 # rescale the demand
N = np.size(data,0) # total number of days
attr_num = np.size(data,1) # total number of attributes

# prepare the set for training & validation
Tp = int(0.8*N) 

train = data[0:Tp,:] # first 80% of days for training 
trainX = np.array([train[i:i+step,:] for i in range(len(train)-step)]) # input
trainY = np.array(train[step:,0]) # expected output

# prepare the test set
test = data[Tp:N,:] # last 20% of days for testing
testX = np.array([test[i:i+step,:] for i in range(len(test)-step)]) # input
testY = np.array(test[step:,0]) # expected output

# create the recurrent neural network 
model = Sequential()
kk = keras.initializers.RandomUniform(minval=-0.01, maxval=0.01)
#model.add(SimpleRNN(units=20, input_shape=(step,attr_num), activation="tanh")) # simple RNN
model.add(LSTM(units=10, input_shape=(step,attr_num))) # LSTM
model.add(Dense(units=1, activation='linear')) 
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error', 'mean_absolute_percentage_error'])
model.summary()

# train the RNN
train_history = model.fit(x=trainX, y=trainY, validation_split=0.25, epochs=500, batch_size=32, verbose=2)
trainPredict = model.predict(trainX) # predicted output on the training set
testPredict = model.predict(testX) # predicted output on the test set

# plot the metrics for training performance
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train], '-')
    plt.plot(train_history.history[validation], '-')
    plt.title('Train History', fontsize=16)
    plt.tick_params(labelsize=16)
    plt.ylabel(train, fontsize=18)  
    plt.xlabel('Epoch', fontsize=18)
    plt.legend(['Train', 'Validate'], loc='best', fontsize=16)
    #plt.yscale('log')
    plt.ylim([0, 0.05])
    plt.show()
show_train_history(train_history, 'mean_squared_error', 'val_mean_squared_error')

# visualize the result by plotting the actual and predicted electricity demand
for i_plot in [1,2]:
    
    plt.figure(figsize=(40,20))
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)  
    
    # plot the actual and predicted demand for training set
    plt.plot(range(0,Tp-step), trainY, 'b')
    plt.plot(range(0,Tp-step), trainPredict, 'r')
    plt.legend(['Actual demand', 'Predicted demand'], loc='best', fontsize=14)  
    
    # plot the actual and predicted demand for test set
    plt.plot(range(Tp,N-step), testY, 'b')
    plt.plot(range(Tp,N-step), testPredict, 'r')
    plt.xlabel('Day', fontsize=18)
    plt.ylabel('Demand', fontsize=18)
    if i_plot == 2:
        plt.xlim([Tp,N])
    if i_plot==1:
        plt.title('Whole time period', fontsize=20)
    else:
        plt.title('Test period', fontsize=20)
    plt.show()

# output the MSE and MAPE for training and test sets
train_MSE = np.mean((trainPredict-trainY)**2)
print('train_MSE =', train_MSE)
train_MAPE = np.mean(np.abs(trainPredict-trainY)/trainY)
print('train_MAPE =', train_MAPE)
test_MSE = np.mean((testPredict-testY)**2)
print('test_MSE =', test_MSE)
test_MAPE = np.mean(np.abs(testPredict-testY)/testY)
print('test_MAPE =', test_MAPE)
