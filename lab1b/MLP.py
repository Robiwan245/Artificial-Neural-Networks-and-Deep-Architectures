from cProfile import label
from tkinter import HIDDEN
import data
import matplotlib.pyplot as plt
import numpy as np

class MLP:
    def __init__(self) -> None:
        pass

    def init_theta(self,row, cols):
        return np.random.normal(size=(row, cols))

    def add_bias(self, input):
        n = input.shape[1]
        return np.concatenate((input, np.ones(shape=(1,n))))

    def transfer(self, x):
        return (2/(1+np.exp(-x))-1)

    def gradient_transfer(self, x):
        return ((1+self.transfer(x)*1-self.transfer(x))/2)

    def backprop(self,X, labels, theta1, theta2, num_hidden=2, alpha=0.1,momentum=0.9, epochs=10):
        X = self.add_bias(X)
        d0_theta,d1_theta = 0,0
        for i in range(epochs):
            # Forward pass
            H = self.add_bias(self.transfer(theta1@X))
            O = self.transfer(theta2@H)

            # Backward pass
            delta_o = (O-labels)*self.gradient_transfer(O)
            delta_h = (theta2.T@delta_o)*self.gradient_transfer(H)
            delta_h = delta_h[range(num_hidden),:]

            # Update weights
            d0_theta = (momentum*d0_theta) - ((1 - momentum)*delta_h@X.T)
            d1_theta = (momentum*d1_theta) - ((1 - momentum)*delta_o@H.T)
            theta1 += alpha*d0_theta
            theta2 += alpha*d1_theta
        
        return theta1, theta2

    def test_plot(self):
        classA, classB, labelsA, labelsB = data.xor()

        plt.figure(figsize=(10, 10))
        plt.xlabel("x-coordinates of the data")
        plt.ylabel("y-coordinates of the data")
        plt.scatter(classA[0,:], classA[1,:], c = "red")
        plt.scatter(classB[0,:], classB[1,:], c = "green")
        plt.plot(labelsA, labelsB, "k")

        plt.show()

MLP_model = MLP()
X, T = data.xor()
num_hidden = 2
epochs = 100
alpha = 0.1
theta1 = MLP_model.init_theta(num_hidden, X.shape[0]+1)
theta2 = MLP_model.init_theta( X.shape[0], num_hidden+1)

new_theta1, new_theta2 = MLP_model.backprop(X,T, theta1, theta2, num_hidden, alpha, epochs=epochs)

w11, w12, bias1 = new_theta1[0,0], new_theta1[0,1], new_theta1[0,2]
w21, w22, bias2 = new_theta1[1,0], new_theta1[1,1], new_theta1[1,2]

plt.scatter(X[0,:], X[1,:], c=T[:])
x = np.linspace(-2, 2,1000)
plt.plot(x, -(w11*x+bias1)/w12, label="Hidden layer 1")
plt.plot(x, -(w21*x+bias2)/w22, label="Hidden layer 2")
plt.legend()
plt.show()