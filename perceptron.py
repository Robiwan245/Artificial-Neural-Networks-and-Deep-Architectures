from re import X
import numpy as np
import data
import matplotlib.pyplot as plt
import time

class perceptron:

    def __init__(self, X_train) -> None:
        self.weights = [0.001 for i in range(len(X_train))]
        self.bias = 0.0

    def perception_rule(self, learning_rate, error, x, j):
        self.weights[j] = self.weights[j]  + learning_rate * error * x[j]

    def delta_rule(self, learning_rate, x, i, target):
        for j in range(2):
            self.weights[i] = self.weights[i] - learning_rate * np.dot((np.dot(self.weights[i], x[j]) - target), np.transpose(x[j]))
        return self.weights

    def delta_rule_batch(self, learning_rate, x, j, target):
        self.weights += -learning_rate * np.dot((np.dot(self.weights, x) - target), np.transpose(x))
        return self.weights

    def activation(self, data):
        value = np.dot(self.weights, data) + self.bias
        return 1 if value >= 0 else 0

    def fit(self, X, y, learning_rate, epochs, batch_mode, delta):
        self.weights = [0.0 for i in range(len(X[0]))]
        self.bias = 0.0

        for epoch in range(epochs):
            for index in range(len(y)):
                x = X[index]
                predicted = self.activation(x)
                if y[index] == predicted:
                    pass
                else:
                    target = y[index]
                    error = target - predicted
                    for j in range(len(x)):
                        self.perception_rule(learning_rate, error, x, j)
                        self.bias += learning_rate * error

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

seqential_mode = False
batch_mode = True
delta = True
perceptron_learning = False

learning_rate = 0.5

if seqential_mode and delta:
    X_train, Y_train, classA, classB = data.generate_linearly_separated_data()
    model = perceptron(X_train)
    for i in range(len(Y_train)):
        weights = model.delta_rule(learning_rate, X_train[i], i, Y_train[i])

elif batch_mode and delta:
    X_train, Y_train, classA, classB = data.generate_linearly_separated_data_batch()
    model = perceptron(X_train)
    weights = model.delta_rule_batch(learning_rate, X_train, 1, Y_train)

elif perceptron_learning:
    X_train, Y_train, classA, classB = data.generate_linearly_separated_data()
    model = perceptron(X_train)
    model.fit(X_train, Y_train, learning_rate, 10, batch_mode, delta)
    weights = model.get_weights()


plt.figure(figsize=(10, 10))
plt.scatter(classA[0,:], classA[1,:], c = "red")
plt.scatter(classB[0,:], classB[1,:], c = "green")

x0_1 = np.amin(X_train[:, :])
x0_2 = np.amax(X_train[:, :])

x1_1 = (-weights[0] * x0_1 - model.bias) / weights[1]
x1_2 = (-weights[0] * x0_2 - model.bias) / weights[1]

plt.plot([x0_1, x0_2], [x1_1, x1_2], "k")
plt.show()