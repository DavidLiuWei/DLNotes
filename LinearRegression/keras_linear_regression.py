# coding: utf8

import numpy as np
from numpy import random
import matplotlib.pyplot as plt

from keras.layers import Input, Dense
from keras.optimizers import SGD
from keras.models import Model

X = random.uniform(0, 30, 100)  # 随机生成在[0,30]区间内服从均匀分布的100个数
y = 1.85 * X + random.normal(0, 2, 100)  # 对X乘以固定系数后加上随机扰动

plt.scatter(X, y)
plt.xlabel('X')
plt.ylabel('y')

input_shape = (1,)
input_tensor = Input(shape=input_shape)
predict = Dense(1, activation='linear', name='output')(input_tensor)
model = Model(inputs=input_tensor, outputs=predict)
print(model.summary())

model.compile(loss='mse', optimizer=SGD(lr=0.0001))
train_history = model.fit(X, y, validation_split=0.2,
                          epochs=100, batch_size=100, verbose=2)

plt.plot(train_history.history['loss'])
plt.plot(train_history.history['val_loss'])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('model loss')
plt.legend(['train', 'val'], loc='upper right')

[w, b] = model.layers[1].get_weights()
print(w, b)

plt.scatter(X, y)
plt.xlabel('X')
plt.ylabel('y')
x1 = np.linspace(0, 30, 1000)
y1 = w[0][0] * x1 + b[0]
plt.plot(x1, y1, 'r')
