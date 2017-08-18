from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
# from keras.utils import plot_model
from keras.utils.np_utils import to_categorical
import numpy as np
import random

timesteps = 16
data_dim = 1
num_classes = 628
batch_size = 64
units = 64
epochs = 10
drop_rate = 0.2

train = np.loadtxt('data/train.txt', dtype=np.int32, delimiter=',')
test = np.loadtxt('data/test.txt', dtype=np.int32, delimiter=',')

train.shape = -1, 17
test.shape = -1, 17

# random.shuffle(train)
x_train = train[:, :-1]
y_train = train[:, -1]
y_train = to_categorical(y_train, num_classes)

x_train.shape = -1, timesteps, data_dim
y_train.shape = -1, num_classes

x_test = test[:, :-1]
x_test.shape = -1, timesteps, data_dim
y_test = test[:, -1]
y_test1 = to_categorical(y_test, num_classes)

model = Sequential()
model.add(LSTM(units, return_sequences=True,
               input_shape=(timesteps, data_dim)))
model.add(LSTM(units, return_sequences=True))
# model.add(LSTM(units, return_sequences=True))
# model.add(LSTM(units, return_sequences=True))
# model.add(LSTM(units, return_sequences=True))
model.add(Dropout(drop_rate))
model.add(LSTM(units))
model.add(Dense(int(units * 4)))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# --------------------------------------------------------
# model.load_weights(filepath="data5.h5")
# --------------------------------------------------------

model.summary()

model.fit(x_train, y_train,
          batch_size=batch_size, epochs=epochs, shuffle=True)
# validation_data=(x_val, y_val))

model.save_weights("model/data.h5")

# plot_model(model, to_file='model.png', show_shapes=True)
score = model.evaluate(x_test, y_test1, batch_size=batch_size)

pd = model.predict(x_test, batch_size=batch_size)
m, n = pd.shape
Y = []
for i in range(m):
    Y.append(np.argmax(pd[i]))
mm = 0
for i in range(m):
    if Y[i] == y_test[i]:
        mm = mm + 1
print("predict: %f" % (mm / m))
print("evaluate: %f" % (score[1]))
