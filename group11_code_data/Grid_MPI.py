
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet,ElasticNetCV
from sklearn.metrics import mean_squared_error
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

X_train=pd.read_csv('X_train.csv')
Y_train=pd.read_csv('Y_train.csv')
X_val=pd.read_csv('X_val.csv')
Y_val=pd.read_csv('Y_val.csv')
X_test=pd.read_csv('X_test.csv')
Y_test=pd.read_csv('Y_test.csv')

alphas = [0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1]

if rank==0:
    data=np.array(alphas)
    data=data.reshape(size, int(len(np.array(alphas))/size))

else:
    data=None
    
recvbuf = np.empty(int(len(np.array(alphas))/size))
comm.Scatter(data,recvbuf, root = 0)

model = ElasticNet(alpha=recvbuf).fit(X_train,Y_train)   
pred_y = model.predict(X_val)
score = model.score(X_val, Y_val)
mse = mean_squared_error(Y_val, pred_y)   
# print("Alpha:{0:.4f}, R2:{1:.2f}, MSE:{2:.2f}, RMSE:{3:.2f}"
#    .format(recvbuf, score, mse, np.sqrt(mse)))
print(rank, 'Alpha:',recvbuf, 'R2:', score, 'MSE:',mse , 'RMSE:', np.sqrt(mse))
