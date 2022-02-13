from turtle import forward

from sklearn.model_selection import validation_curve
import data
import matplotlib.pyplot as plt
import numpy as np
import time
import tensorflow
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import keras
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras import regularizers 
from sklearn.metrics import mean_squared_error as mse
from keras.optimizer_v1 import SGD
from tqdm import tqdm


class MLP:

    def __init__(self) -> None:
        pass

    def create_model(self, nr_nodes_1, nr_nodes_2, input_shape = 5, output_shape = 1):

        model = Sequential()
        model.add(Dense(nr_nodes_1, input_dim=input_shape, kernel_initializer='random_normal', activation='relu', use_bias = True))
        model.add(Dense(nr_nodes_2, kernel_initializer='random_normal', activation='relu', use_bias = True))
        model.add(Dense(output_shape, activation='relu'))

        return model

    def mse(self, t, y):
        return np.sum(np.square(y - t)) / len(y)

# t = 301:1500;
# input = [x(t-20); x(t-15); x(t-10); x(t-5); x(t)];
# output = x(t+5);

beta = 0.2
gamma = 0.1
n = 10
tau = 25
N = 1600

x = np.zeros(N)
x[0] = 1.5

for t in range(1, N):
    if not (t - tau) < 0:
        x[t] = x[t - 1] + ((beta * x[t - tau - 1]) / (1 + x[t - tau - 1]**10)) - gamma * x[t - 1]
    else:
        x[t] = 0

t = np.arange(0, 1600)

plt.figure()
#plt.scatter(t, x)
plt.plot(x)
plt.show()

t = np.arange(300, 1500)

input = np.transpose(np.array([x[t - d] for d in range(-20, 1, 5)]))
output = x[t + 5]

training = input[:-200]
training_output = output[:-200]
validation = input[-500:-200]
validation_output = output[-500:-200]
test = input[:-200]
test_output = output[:-200]

noice_list = np.transpose(np.array([np.random.normal(size = 100, loc = 0, scale = 0.05) for d in range(-20, 1, 5)]))
noice_list_output = np.random.normal(size = 100, loc = 0, scale = 0.05)

training_with_noice = np.vstack((training, noice_list))
training_output_with_noice = np.hstack((training_output, noice_list_output))

np.random.seed(10)
np.random.shuffle(training_with_noice)
np.random.shuffle(training_output_with_noice)


THRESHHOLD = 0.005

nr_nodes_1 = [9, 3, 4, 5]
nr_nodes_2 = [6, 2, 4, 6]

lambda_param = [1e-6, 1e-3, 1e-1]

avg = 0
test_acc = 0

NOICE = True
for i in tqdm(range(10)):
        accuracys = []
        mlp = MLP()
        model = mlp.create_model(nr_nodes_1[3], nr_nodes_2[0])
        sgd = SGD(lr=0.01, decay=lambda_param[0], momentum=0.9, nesterov=True)
        model.compile(loss = 'mse', optimizer = 'sgd')
        for _ in range(10):
            if NOICE:
                model.fit(training_with_noice, training_output_with_noice, epochs = 10, verbose = 0)
            else:
                model.fit(training, training_output, epochs = 10, verbose = 0)
            pred = model.predict(validation, verbose = 0)
            accuracy = mse(validation_output, pred)
            accuracys.append(accuracy)
            #print(np.log(THRESHHOLD))
            if not NOICE:
                if np.log(accuracy) < np.log(THRESHHOLD):
                    break
        plt.figure()
        plt.plot(accuracys)
        plt.xlabel("Number of iterations")
        plt.ylabel("MSE of model with " + str(nr_nodes_1[0]) + " and " + str(nr_nodes_2[0]) + " nodes")
        plt.show()
        #avg += accuracy

#print(avg/10)
            #model.summary()
            

#         test_pred = model.predict(test, verbose = 0)
#         accuracy = mse(test_output, test_pred)
#         test_acc += accuracy
#             print("TEST: " + str(accuracy))
# print(test_acc/10)

plt.figure()
plt.plot(pred)
plt.show()

_, ax = plt.subplots(2)
ax[0].plot(training_output)
ax[1].plot(test_pred)
plt.show()