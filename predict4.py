from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import numpy as np
import random

def Normalization(X):
    X[:,0] = (X[:,0]-np.min(X[:,0]))/(np.max(X[:,0])-np.min(X[:,0]))
    X[:,0] = 2*X[:,0]-1
    X[:,1] = (X[:,1]-np.min(X[:,1]))/(np.max(X[:,1])-np.min(X[:,1]))
    X[:,1] = 2*X[:,1]-1
    return X


def FindMin_Max(X):
    return min(X[:,0]), np.max(X[:,0]), np.min(X[:,1]), np.max(X[:,1])


def ToRealValue(X, M):
    X0_min = M[0]
    X0_max = M[1]
    X1_min = M[2]
    X1_max = M[3]
    X[:,0] = (X[:,0]+1)/2
    X[:,0] = (X0_max-X0_min)*X[:,0]+X0_min
    X[:,1] = (X[:,1]+1)/2
    X[:,1] = (X1_max-X1_min)*X[:,1]+X1_min
    return X

def rmse(predictions, targets):
    return np.sqrt(np.mean((predictions-targets)**2))


timesteps = 6
data_dim = 2
batch_size = 64
epochs = 10

x_train = np.loadtxt('data/Re_x_t.txt', dtype=np.float32, delimiter=',')
x_test = np.loadtxt('data/Re_x_v.txt', dtype=np.float32, delimiter=',')
y_train = np.loadtxt('data/Re_y_t.txt', dtype=np.float32, delimiter=',')
y_test = np.loadtxt('data/Re_y_v.txt', dtype=np.float32, delimiter=',')

X = np.r_[x_train, x_test]
Min_Max = []
Min_Max = FindMin_Max(X)
X = Normalization(X)
x_train = X[:x_train.shape[0], :]
x_test = X[x_train.shape[0]:, :]

y_train = Normalization(y_train)

x_train.shape = -1, timesteps, data_dim
x_test.shape = -1, timesteps, data_dim

model = Sequential()
model.add(LSTM(128, return_sequences=False, input_shape=(timesteps, data_dim)))
model.add(Dropout(0.25))
model.add(Dense(64, activation='tanh'))
model.add(Dense(16, activation='tanh'))
model.add(Dense(data_dim, activation='tanh'))

model.compile(loss='mse', optimizer='sgd', metrics=['mse'])

model.summary()
# --------------------------------------------------------
# model.load_weights(filepath="test.h5")
# --------------------------------------------------------

early_stopping = EarlyStopping(monitor='val_mean_squared_error', patience=1)

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
          validation_split=0.1, callbacks=[early_stopping])

model.save_weights("test.h5")

predictions = model.predict(x_test, batch_size=64)
predictions = ToRealValue(predictions, Min_Max)
print('rmse:', rmse(predictions, y_test))