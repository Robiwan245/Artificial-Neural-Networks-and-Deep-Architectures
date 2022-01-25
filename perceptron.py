import numpy as np
import data
import matplotlib.pyplot as plt

class perceptron:

    def __init__(self) -> None:
        self.weights = []

    def activation(self, data):
        value = np.dot(self.weights, data.T)
        return 1 if value >= 0 else 0

    def fit(self, X, y, learning_rate, epochs):
        self.weights = [0.0 for i in range(len(X[0]))]

        for epoch in range(epochs):
            sum_error = 0.0
            for index in range(len(X)):
                x = X[index]
                predicted = self.activation(x)
                if y[index] == predicted:
                    pass
                else:
                    error = y[index] - predicted
                    sum_error += error
                    for j in range(len(x)):
                        self.weights[j] = self.weights[j]  + learning_rate * error * x[j]

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

X_train, Y_train, X_test, Y_test = data.generate_linearly_separated_data()

model = perceptron()
model.fit(X_train, Y_train, 0.1, 10)

weights = model.get_weights()

plt.figure(figsize=(10, 10))
plt.scatter(X_train[0,:], X_train[1,:], c = "red")
plt.scatter(X_train[2,:], X_train[3,:], c = "green")

for i in np.linspace(np.amin(X_train[:,:]), np.amax(X_train[:,:])):

    slope = -(weights[0]/weights[2])/(weights[0]/weights[1])  
    intercept = -weights[0]/weights[2]

    y = (slope*i) + intercept
    
    plt.plot(i, y, 'ko')

plt.show()
