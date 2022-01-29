from re import X
import numpy as np
import data
import matplotlib.pyplot as plt
import time

class perceptron:

    def __init__(self, X_train) -> None:
        self.weights = [0.001 for i in range(len(X_train))]
        #self.weights = [np.random.rand for i in range(len(X_train))]
        #self.weights = [100 for i in range(len(X_train))]
        self.bias = 0.0
        self.sum = 0

    def perception_rule(self, learning_rate, error, x, j):
        self.weights[j] = self.weights[j] + learning_rate * error * x[j]

    def delta_rule(self, learning_rate, x, i, target):
        for j in range(2):
            self.weights[i] = self.weights[i] - learning_rate * np.dot((np.dot(self.weights[i], x[j]) - target), np.transpose(x[j]))
            self.sum += self.compute_mse((np.dot(self.weights[i], x[j])), target)
        return self.weights

    def delta_rule_batch(self, learning_rate, x, j, target):
        self.weights += -learning_rate * np.dot((np.dot(self.weights, x) - target), np.transpose(x))
        return self.weights

    def activation(self, data):
        value = np.dot(self.weights, data) + self.bias
        if perceptron_learning == True:
            return 1 if value >= 0 else 0
        else:
            return 1 if value >= 0 else -1

    def fit(self, X, y, learning_rate, epochs, batch_mode, delta):
        self.weights = [0.0 for i in range(len(X[0]))]
        self.bias = 0.0

        for epoch in range(epochs):
            self.sum = 0
            for index in range(len(y)):
                x = X[index]
                predicted = self.activation(x)
                target = y[index]
                self.sum += self.compute_mse((np.dot(self.weights, x)), target)
                if y[index] == predicted:
                    pass
                else:
                    error = target - predicted
                    for j in range(len(x)):
                        self.perception_rule(learning_rate, error, x, j)
                        self.bias += learning_rate * error
            MSE_perceptron.append(self.sum / len(y))
        print("MSE for perceptron rule: " + str(self.sum / len(y)))

    def predict(self, x_test):
        predicted = []
        for i in range(len(x_test)):
            predicted.append(self.activation(x_test[i]))
        return predicted

    def accuracy(self, predicted, original):
        correct = 0
        for i in range(len(predicted)):
            if predicted[i] == original[i]:
                correct += 1

        return (correct/len(predicted)) * 100

    def get_weights(self):
        return self.weights

    def compute_mse(self, y, t):
        return np.sum(np.square(y - t))

seqential_mode = True
batch_mode = False
delta = True
perceptron_learning = True
linearly_separable = True
new_data = False

remove_25 = False
remove_50_A = False
remove_50_B = False
a_20_80 = False

learning_rate = 0.01
iter = 10
MSE_delta = []
MSE_perceptron = []

if seqential_mode and delta:
    if linearly_separable:
        X_train, Y_train, classA, classB = data.generate_linearly_separated_data(perceptron_learning, seqential_mode, batch_mode)
    elif not linearly_separable:
        X_train, Y_train, classA, classB = data.generate__non_linearly_separated_data(perceptron_learning, seqential_mode, batch_mode)
    model = perceptron(X_train)
    for epoch in range(iter):
        model.sum = 0
        for i in range(len(Y_train)):
            weights = model.delta_rule(learning_rate, X_train[i], i, Y_train[i])
        MSE_delta.append(model.sum /len(X_train))
        print("MSE for delta rule in online mode epoch " + str(epoch) + ": " + str(model.sum /len(X_train)))

if batch_mode and delta:
    if linearly_separable:
        X_train, Y_train, classA, classB = data.generate_linearly_separated_data(perceptron_learning, seqential_mode, batch_mode)
    elif not linearly_separable and not new_data:
        X_train, Y_train, classA, classB = data.generate__non_linearly_separated_data(perceptron_learning, seqential_mode, batch_mode)
    elif not linearly_separable and new_data:
        X_train, Y_train, classA, classB = data.generate__non_linearly_separated_data2(perceptron_learning, seqential_mode, batch_mode, remove_25, remove_50_A, remove_50_B, a_20_80)
    model = perceptron(X_train)
    for _ in range(iter):
        model.sum = 0
        weights = model.delta_rule_batch(learning_rate, X_train, 10, Y_train)

if perceptron_learning:
    if linearly_separable:
        X_train, Y_train, classA, classB = data.generate_linearly_separated_data(perceptron_learning, seqential_mode, batch_mode)
    elif not linearly_separable:
        X_train, Y_train, classA, classB = data.generate__non_linearly_separated_data(perceptron_learning, seqential_mode, batch_mode)
    model = perceptron(X_train)
    #for _ in range(iter):
    model.fit(X_train, Y_train, learning_rate, 10, batch_mode, delta)
    weights = model.get_weights()

plt.figure(figsize=(10, 10))
plt.xlabel("x-coordinates of the data")
plt.ylabel("y-coordinates of the data")
plt.scatter(classA[0,:], classA[1,:], c = "red")
plt.scatter(classB[0,:], classB[1,:], c = "green")

x0_1 = np.amin(X_train[:, :])
x0_2 = np.amax(X_train[:, :])

x1_1 = (-weights[0] * x0_1 - model.bias) / weights[1]
x1_2 = (-weights[0] * x0_2 - model.bias) / weights[1]

plt.plot([x0_1, x0_2], [x1_1, x1_2], "k")
plt.show()

plt.figure(figsize=(10, 10))
plt.xlabel("Number of epochs")
plt.ylabel("Mean square error")
plt.plot(MSE_delta)
plt.plot(MSE_perceptron)
plt.show()