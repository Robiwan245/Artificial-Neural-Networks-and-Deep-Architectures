import numpy as np
import data
import matplotlib.pyplot as plt

class perceptron:

    def __init__(self) -> None:
        self.weights = []
        self.bias = 0.0

    def activation(self, data):
        value = np.dot(self.weights, data) + self.bias
        return 1 if value >= 0 else 0

    def fit(self, X, y, learning_rate, epochs):
        self.weights = [0.0 for i in range(len(X[0]))]
        self.bias = 0.0

        for epoch in range(epochs):
            for index in range(len(X)):
                x = X[index]
                predicted = self.activation(x)
                if y[index] == predicted:
                    pass
                else:
                    error = y[index] - predicted
                    for j in range(len(x)):
                        self.weights[j] = self.weights[j]  + learning_rate * error * x[j]
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

X_train, Y_train, classA, classB = data.generate_linearly_separated_data()

model = perceptron()
model.fit(X_train, Y_train, 0.1, 10)

weights = model.get_weights()

plt.figure(figsize=(10, 10))
plt.scatter(classA[0,:], classA[1,:], c = "red")
plt.scatter(classB[0,:], classB[1,:], c = "green")

x0_1 = np.amin(X_train[:, 0])
x0_2 = np.amax(X_train[:, 0])

x1_1 = (-weights[0] * x0_1 - model.bias) / weights[1]
x1_2 = (-weights[0] * x0_2 - model.bias) / weights[1]

plt.plot([x0_1, x0_2], [x1_1, x1_2], "k")

plt.show()