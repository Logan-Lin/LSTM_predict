from keras.models import Sequential
from keras.layers import LSTM, Dense, SimpleRNN
from keras.utils.np_utils import to_categorical
import numpy as np
from keras.utils import plot_model

timesteps = 8
data_dim = 1
num_classes = 578

batch = 64
epoch = 20

x_train = np.loadtxt('data_0813/x_train.txt', delimiter=',')
x_val = np.loadtxt('data_0813/x_value.txt', delimiter=',')
y_train = np.loadtxt('data_0813/y_train.txt', delimiter=',')
y_val = np.loadtxt('data_0813/y_value.txt', delimiter=',')

x_train = x_train[:, [1]]
x_val = x_val[:, [1]]
y_train = y_train[:, [1]]
y_val = y_val[:, [1]]

# np.delete(x_train, [0], axis=1)
# np.delete(x_val, [0], axis=1)
# np.delete(y_train, [0], axis=1)
# np.delete(y_val, [0], axis=1)

y_train = to_categorical(y_train, num_classes)
y_val = to_categorical(y_val, num_classes)

y_train.shape = -1, num_classes
y_val.shape = -1, num_classes

x_train.shape = -1, timesteps, data_dim
x_val.shape = -1, timesteps, data_dim

# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(batch, return_sequences=True,
               input_shape=(timesteps, data_dim)))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(64))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

time = "0813_01"

hist = model.fit(x_train, y_train,
                 batch_size=batch, epochs=epoch)
model.save_weights("save/data_" + time + ".h5")

history_file = open("save/history_" + time + ".txt", "w")
history_file.write("batch_size=" + str(batch) + "\n")
history_file.write(str(hist.history))
history_file.close()

summary = str(model.to_json())
summary_file = open("save/summary_file_" + time + ".txt", "w")
summary_file.write("batch_size=" + str(batch) + "\n")
summary_file.write(summary)
summary_file.close()

# model.load_weights(filepath="data.h5")

plot_model(model, to_file='save/model_' + time + '.png', show_shapes=True)

score = model.evaluate(x_val, y_val, batch_size=64)

print("evaluate: %f" % (score[1]))
