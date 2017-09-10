from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import numpy as np
import random

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

x_train.shape = -1, timesteps, data_dim
x_test.shape = -1, timesteps, data_dim

model = Sequential()
model.add(LSTM(128, return_sequences=False, input_shape=(timesteps, data_dim)))
model.add(Dropout(0.25))
model.add(Dense(64, activation='tanh'))
model.add(Dense(16, activation='tanh'))
model.add(Dense(data_dim, activation='linear'))

model.compile(loss='mse', optimizer='sgd', metrics=['mse'])

model.summary()
# --------------------------------------------------------
# model.load_weights(filepath="test2.h5")
# --------------------------------------------------------

early_stopping = EarlyStopping(monitor='val_mean_squared_error', patience=1)

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
          validation_split=0.1, callbacks=[early_stopping])

model.save_weights("test2.h5")

predictions = model.predict(x_test, batch_size=64)
print('rmse:', rmse(predictions, y_test))