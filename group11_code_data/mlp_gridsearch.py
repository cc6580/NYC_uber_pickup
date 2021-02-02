

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
from mpi4py import MPI
import itertools
from math import sqrt
import time

ConvergenceWarning('ignore')


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

data=pd.read_csv('data.csv')

X = data.drop(['pickups'],axis=1)
Y = data['pickups']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8,test_size=0.2, random_state=1)

X_train, X_val, Y_train, Y_val  = train_test_split(X_train, Y_train, test_size=0.25, random_state=1)

activation = ['tanh','relu']
solver = ['lbfgs','sgd','adam']
alpha=[0.001,0.01,0.1,1]
learning_rate = [0.001,0.0001]
hidden = [(100,50,25),(200,100,100,50),(100,75,50,50)]
para = list(itertools.product(activation,solver,alpha,learning_rate))
block = len(para)//size

start = time.time()

if rank!=size-1:
    parameters = para[rank*block:(rank+1)*block]
    mse = []
    for parameter in parameters:
        mlp = MLPRegressor(hidden_layer_sizes=(100,75,50,50),activation = parameter[0],
                                solver=parameter[1],alpha=parameter[2],learning_rate_init=parameter[3],early_stopping=True,max_iter=200,learning_rate = 'adaptive')
        mlp.fit(X_train, Y_train)
        pred_y = mlp.predict(X_val)
        mse.append(mean_squared_error(pred_y,Y_val))
    index = np.argmin(mse)
    print("rank #",rank,": ",parameters[index],", mse: ",sqrt(mse[index]))
else:
    parameters = para[rank*block:len(para)]
    mse = []
    for parameter in parameters:
        mlp = MLPRegressor(hidden_layer_sizes=(100,75,50,50),activation = parameter[0],
                                solver=parameter[1],alpha=parameter[2],learning_rate_init=parameter[3],early_stopping=True,max_iter=200,learning_rate = 'adaptive')
        mlp.fit(X_train, Y_train)
        pred_y = mlp.predict(X_val)
        mse.append(mean_squared_error(pred_y,Y_val))
    index = np.argmin(mse)
    print("rank #",rank,": ",parameters[index],", mse: ",sqrt(mse[index]))
end = time.time()
print("#",rank,": ",end-start)
