import numpy as np
import data
import matplotlib.pyplot as plt

class perceptron:

    def __init__(self) -> None:
        self.weights = []
        self.bias = 0.0

    def perception_rule(self, learning_rate, error, x, j):
        self.weights[j] = self.weights[j]  + learning_rate * error * x[j]

    def delta_rule(self, learning_rate, x, j, target):
        self.weights[j] = learning_rate * x[j] * np.dot((np.dot(self.weights[j], x[j]) - target), np.transpose(x[j]))

    def activation(self, data):
        value = np.dot(self.weights, data) + self.bias
        return 1 if value >= 0 else 0

    def fit(self, X, y, learning_rate, epochs, batch_mode, delta):
        self.weights = [0.0 for i in range(len(X[0]))]
        self.bias = 0.0

        if batch_mode:
            lent = len(X)
        else:
            lent = len(y)

        for epoch in range(epochs):
            for index in range(lent):
                x = X[index]
                predicted = self.activation(x)
                if y[index] == predicted:
                    pass
                else:
                    target = y[index]
                    error = target - predicted
                    for j in range(len(x)):
                        if delta:
                            self.delta_rule(learning_rate, x, j, target)
                        else:
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

seqential_mode = True
batch_mode = False
delta = True

if seqential_mode:
    X_train, Y_train, classA, classB = data.generate_linearly_separated_data()

elif batch_mode:
    X_train, Y_train, classA, classB = data.generate_linearly_separated_data_batch()

model = perceptron()
model.fit(X_train, Y_train, 0.5, 10, batch_mode, delta)

weights = model.get_weights()

plt.figure(figsize=(10, 10))
plt.scatter(classA[0,:], classA[1,:], c = "red")
plt.scatter(classB[0,:], classB[1,:], c = "green")

if seqential_mode:

    x0_1 = np.amin(X_train[:, :])
    x0_2 = np.amax(X_train[:, :])

    x1_1 = (-weights[0] * x0_1 - model.bias) / weights[1]
    x1_2 = (-weights[0] * x0_2 - model.bias) / weights[1]

    plt.plot([x0_1, x0_2], [x1_1, x1_2], "k")

elif batch_mode:

    for i in np.linspace(np.amin(X_train[:,:]), np.amax(X_train[:,:])):
            slope = -(weights[0]/weights[2] - model.bias)/(weights[0]/weights[1])  
            intercept = -weights[0]/weights[2]

            y = (slope*i) + intercept
            plt.plot(i, y,'ko')

    # x0_1 = np.amin(X_train[:, :])
    # x0_2 = np.amax(X_train[:, :])

    # x1_1 = (-weights[0]/weights[2] - model.bias)/(weights[0]/weights[1])
    # x1_2 = (-weights[0] - model.bias)/weights[2]

    # plt.plot([x0_1, x0_2], [x1_1, x1_2], "k")

plt.show()