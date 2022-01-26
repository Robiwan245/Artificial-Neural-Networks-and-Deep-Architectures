from timeit import repeat
import numpy as np
import data
import time
import matplotlib.pyplot as plt

class perceptron:

    def __init__(self, X) -> None:
        self.weights = np.ones(len(X)+1)
        print(np.shape(self.weights))

    def activation(self, x):
        return 1 if x >= 0 else 0

    def fit(self, X, targets, learning_rate, epochs, classA, classB, delta):
        weights = model.get_weights()
        
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(classA[0,:], classA[1,:], c = "red")
        ax.scatter(classB[0,:], classB[1,:], c = "green")
        x0_1 = np.amin(X_train[:, 0])
        x0_2 = np.amax(X_train[:, 0])
        x1_1 = (-weights[1] * x0_1 - weights[0]) / weights[2]
        x1_2 = (-weights[1] * x0_2 - weights[0]) / weights[2]
        line1, = ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")

        for _ in range(epochs):
            for index in range(len(targets)):
                x = np.insert(X[index], 0, 1)
                predLabels = self.predict(x)
                if delta:
                    error = targets[index] - self.weights.T@x
                    self.weights -= learning_rate * (self.weights@x - error) * x.T
                else:
                    error = targets[index] - predLabels
                    self.weights = self.weights + learning_rate * error * x
                
            weights = model.get_weights()
            x0_1_new = np.amin(X_train[:, 0])
            x0_2_new = np.amax(X_train[:, 0])
            x1_1_new = (-weights[1] * x0_1 - weights[0]) / weights[2]
            x1_2_new = (-weights[1] * x0_2 - weights[0]) / weights[2]

            line1.set_xdata([x0_1_new, x0_2_new])
            line1.set_ydata([x1_1_new, x1_2_new])
            fig.canvas.draw()
            fig.canvas.flush_events()

            time.sleep(0.01)
              

    def predict(self, x):
        z = self.weights.T.dot(x)
        a = self.activation(z)
        return a

    def get_weights(self):
        return self.weights
    
X_train, labels_train, classA, classB = data.generate_linearly_separated_data()

model = perceptron(X_train.T)
model.fit(X_train, labels_train, 0.01, 100, classA, classB, True)



