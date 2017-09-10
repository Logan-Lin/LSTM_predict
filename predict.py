import random

import keras
import numpy as np
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
# from keras.utils import plot_model
from keras.utils.np_utils import to_categorical

timesteps = 6
data_dim = 1
num_classes = 26514
batch_size = 64
units = 64
epochs = 10
drop_rate = 0.1
time = 1
batch = 64

x_train = np.loadtxt('data/x_train.txt', delimiter=',')
x_val = np.loadtxt('data/x_value.txt', delimiter=',')
y_train = np.loadtxt('data/y_train.txt', delimiter=',')
y_val = np.loadtxt('data/y_value.txt', delimiter=',')

x_train = x_train[:, 1]
x_val = x_val[:, 1]

x_train.shape = -1, timesteps

data = np.c_[x_train, y_train]
random.shuffle(data)
x_train = data[:, :-1]
y_train = data[:, -1]

y = y_val
y1 = y_train
y_train = to_categorical(y_train, num_classes)
y_val = to_categorical(y_val, num_classes)

y_train.shape = -1, num_classes
x_train.shape = -1, timesteps, data_dim
x_val.shape = -1, timesteps, data_dim

model = Sequential()
model.add(LSTM(units, return_sequences=True,
               input_shape=(timesteps, data_dim)))
model.add(LSTM(units, return_sequences=True))
# model.add(LSTM(units, return_sequences=True))
# model.add(LSTM(units, return_sequences=True)
# model.add(LSTM(units, return_sequences=True))
model.add(Dropout(drop_rate))
model.add(LSTM(units))
model.add(Dense(num_classes, activation='softmax'))
# model.add(Dense(num_classes, activation='softmax', input_dim=units,
#                 kernel_regularizer=regularizers.l2(0.01),
#                 activity_regularizer=regularizers.l1(0.01)))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# --------------------------------------------------------
# model.load_weights(filepath="data5.h5")
# --------------------------------------------------------
early_stopping = keras.callbacks.EarlyStopping(monitor="val_acc", patience=50, mode="auto")

hist = model.fit(x_train, y_train,
                 validation_split=0.1,
                 callbacks=[early_stopping],
                 batch_size=batch, epochs=epochs)

model.save_weights("model/data" + str(time) + ".h5")

history_file = open("history_" + str(time) + ".txt", "w")
history_file.write("batch_size=" + str(batch) + "\n")
history_file.write(str(hist.history))
history_file.close()

model.summary()

# plot_model(model, to_file='model.png', show_shapes=True)
score = model.evaluate(x_val, y_val, batch_size=batch_size)

print("loss: %f \n" % (score[0]))
print("evaluate: %f \n" % (score[1]))
