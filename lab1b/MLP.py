import data
import matplotlib.pyplot as plt
import numpy as np
import time

class MLP:
    def __init__(self) -> None:
        pass

    def init_theta(self,n_output, n_input):
        np.random.seed(1)
        weights = np.random.normal(size=(n_output,n_input), loc=0, scale=1) 
    
        return(weights)

    def add_bias(self, input):
        n = input.shape[1]
        return np.concatenate((input, np.ones(shape=(1,n))))

    def transfer(self, x):
        return (2/(1+np.exp(-x)))-1

    def gradient_transfer(self, x_out):
        return ((1+x_out)*(1-x_out))/2

    def backprop(self,X, targets, theta1, theta2, num_hidden=2, alpha=0.1,momentum=0.9, epochs=10):
        X = self.add_bias(X)
        d0_theta,d1_theta = 0,0
        for i in range(epochs):
            # Forward pass
            h_in = theta1@X
            h_out = self.add_bias(self.transfer(h_in))
            o_in = theta2@h_out
            o_out = self.transfer(o_in)

            # Backward pass
            delta_o = (o_out-targets)*self.gradient_transfer(o_out)
            delta_h = (theta2.T@delta_o)*self.gradient_transfer(h_out)
            delta_h = delta_h[range(num_hidden),:]

            # Update weights
            d0_theta = (momentum*d0_theta) - ((1 - momentum)*(delta_h@X.T))
            d1_theta = (momentum*d1_theta) - ((1 - momentum)*(delta_o@h_out.T))
            theta1 += alpha*d0_theta
            theta2 += alpha*d1_theta
        
        return theta1, theta2
    
    def error_testing(self, X, labels, num_hidden_list, alpha, epochs):
        class_errors = []
        MSEs = []

        for num_hidden in num_hidden_list:
            theta1 = self.init_theta(num_hidden, X.shape[0]+1)
            theta2 = self.init_theta( X.shape[0], num_hidden+1)
            new_theta1, new_theta2 = self.backprop(X, labels, theta1, theta2, num_hidden,epochs=epochs, alpha=alpha, momentum=0.9)

            # Error testing
            predicted = self.forward(X, new_theta1, new_theta2)
            classification_error = np.mean(abs(np.sign(predicted)-labels)/2)
            class_errors.append(classification_error)
            MSE = np.mean((predicted-labels)**2)
            MSEs.append(MSE) 
        
        fig = plt.figure(figsize=(10,5))
        for num_hidden in num_hidden_list:
            fig.add_subplot(121)
            plt.plot(num_hidden_list,class_errors)
            plt.title("Classification error")
            plt.xlabel("Number of hidden Nodes")
            fig.add_subplot(122)
            plt.plot(num_hidden_list,MSEs)
            plt.title("MSE")
            plt.xlabel("Number of hidden Nodes")
            plt.legend()
            plt.show()

    def forward(self, X, theta1, theta2):
        X = self.add_bias(X)
        H = self.add_bias(self.transfer(theta1@X))
        O = self.transfer(theta2@H)

        return O

MLP_model = MLP()
X, T,a,b = data.generate_not_linearly_separable_data()
num_hidden = 2
epochs = 50
alpha = 0.001
theta1 = MLP_model.init_theta(num_hidden, X.shape[0]+1)
theta2 = MLP_model.init_theta(X.shape[0], num_hidden+1)

new_theta1, new_theta2 = MLP_model.backprop(X,T, theta1, theta2, num_hidden, alpha=alpha, epochs=epochs, momentum=0.9)

w11, w12, bias1 = new_theta1[0,0], new_theta1[0,1], new_theta1[0,2]
w21, w22, bias2 = new_theta1[1,0], new_theta1[1,1], new_theta1[1,2]

plt.scatter(X[0,:], X[1,:], c=T[:])
x = np.linspace(-2, 2, 1000)
plt.plot(x, -(w11*x+bias1)/w12, label="Hidden layer 1")
plt.plot(x, -(w21*x+bias2)/w22, label="Hidden layer 2")
plt.legend()
plt.show()

num_hidden_list = range(2,11)
MLP_model.error_testing(X,T,num_hidden_list,alpha,epochs)