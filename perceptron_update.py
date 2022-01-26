import numpy as np
import data
import matplotlib.pyplot as plt
import time

class perceptron:

    def __init__(self) -> None:
        self.weights = []

    def activation(self, data):
        value = np.dot(self.weights, data)
        return 1 if value >= 0 else 0

    def fit(self, learning_rate, epochs):
        X_train, Y_train, classA, classB = data.generate_linearly_separated_data()      
        self.weights = [0.1 for i in range(len(X_train[0]))]

        plt.ion()
        figure, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(classA[0,:], classA[1,:], c = "red")
        ax.scatter(classB[0,:], classB[1,:], c = "green")
        x0_1 = np.amin(X_train[:, 0])
        x0_2 = np.amax(X_train[:, 0])
        x1_1 = (-self.weights[0] * x0_1) / self.weights[1]
        x1_2 = (-self.weights[0] * x0_2) / self.weights[1]
        line1, = ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")
        

        for epoch in range(epochs):
            for index in range(len(X_train)):
                x = X_train[index]
                predicted = self.activation(x)
                if Y_train[index] == predicted:
                    pass
                else:
                    error = Y_train[index] - predicted
                    for j in range(len(x)):
                        self.weights[j] = self.weights[j]  + learning_rate * error * x[j]
            
            x0_1_new = np.amin(X_train[:, 0])
            x0_2_new = np.amax(X_train[:, 0])
            x1_1_new = (-self.weights[0] * x0_1_new) / self.weights[1]
            x1_2_new = (-self.weights[0] * x0_2_new) / self.weights[1]
            line1.set_xdata([x0_1_new, x0_2_new])
            line1.set_ydata([x1_1_new, x1_2_new])
            figure.canvas.draw()
            figure.canvas.flush_events()

            time.sleep(0.01)

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

model = perceptron()
model.fit(0.001, 100)

time.sleep(100)