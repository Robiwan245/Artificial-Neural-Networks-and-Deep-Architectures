from cProfile import label
from sympy import E
import data
import matplotlib.pyplot as plt
import numpy as np
import time

class MLP:
    def __init__(self) -> None:
        pass

    def init_theta(self,n_output, n_input):
        np.random.seed(1)
        weights = np.random.normal(size=(n_output,n_input)) 
    
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
    
    def error_testing(self, X_train, labels_train, num_hidden_list, alpha, title, epochs_list, X_validation, labels_validation):
        class_errors = {}
        MSEs = {}
        for epochs in epochs_list:
            class_errors[epochs] = []
            MSEs[epochs] = []
            for num_hidden in num_hidden_list:
                theta1 = self.init_theta(num_hidden, X_train.shape[0]+1)
                theta2 = self.init_theta( X_train.shape[0], num_hidden+1)
                new_theta1, new_theta2 = self.backprop(X_train, labels_train, theta1, theta2, num_hidden,epochs=epochs, alpha=alpha, momentum=0.9)

                # Error testing
                predicted = self.forward(X_validation, new_theta1, new_theta2)
                classification_error = np.mean(abs(np.sign(predicted)-labels_validation)/2)
                class_errors[epochs].append(classification_error)
                MSE = np.mean((predicted-labels_validation)**2)
                MSEs[epochs].append(MSE) 
            
        _, ax = plt.subplots(2)
        for epochs in epochs_list:
            ax[0].plot(num_hidden_list, class_errors[epochs], label='Epochs= ' + str(epochs))
            ax[0].title.set_text("Classification error " + title)
            ax[0].set_xlabel("Number of hidden nodes")
            ax[0].legend()
            ax[1].plot(num_hidden_list, MSEs[epochs], label='Epochs= ' + str(epochs))
            ax[1].title.set_text("MSE " + title)
            ax[1].set_xlabel("Number of hidden nodes")
            ax[1].legend()
        

    def forward(self, X, theta1, theta2):
        X = self.add_bias(X)
        H = self.add_bias(self.transfer(theta1@X))
        O = self.transfer(theta2@H)

        return O

MLP_model = MLP()
X, T,_,_ , n = data.generate_not_linearly_separable_data()
num_hidden = 200
epochs = 1000
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

## 3.1.1 quesiton 1
num_hidden_list = range(2,11)
MLP_model.error_testing(X,T,num_hidden_list,alpha,"train",[50,100,200, 500, 1000], X,T)
plt.show()

## 3.1.1 question 2
A_idx_list = np.where(T==1)[0]
B_idx_list = np.where(T==-1)[0]

# 25% from each class
# A_sample_idx_list = np.random.choice(A_idx_list, int(n*0.25), replace=False)
# B_sample_idx_list = np.random.choice(B_idx_list, int(n*0.25), replace=False)
# train_idx_list = np.concatenate((A_sample_idx_list, B_sample_idx_list))
# validation_idx_list = [i for i in range(X.shape[1]) if i not in train_idx_list]
# X_train = X[:, train_idx_list]
# T_train = T[train_idx_list]
# X_validation = X[:, validation_idx_list]
# T_validation = T[validation_idx_list]

# MLP_model.error_testing(X_train, T_train, num_hidden_list, alpha, "test", [50,100,200, 500, 1000], X_train, T_train) # Train
# plt.show()
# MLP_model.error_testing(X_train, T_train, num_hidden_list, alpha, "Validation", [50,100,200, 500, 1000], X_validation, T_validation) # Validation
# plt.show()

# 50% from class A
# A_sample_idx_list = np.random.choice(A_idx_list, int(n*0.50), replace=False)
# B_sample_idx_list = B_idx_list
# train_idx_list = np.concatenate((A_sample_idx_list, B_sample_idx_list))
# validation_idx_list = [i for i in range(X.shape[1]) if i not in train_idx_list]
# X_train = X[:, train_idx_list]
# T_train = T[train_idx_list]
# X_validation = X[:, validation_idx_list]
# T_validation = T[validation_idx_list]

# MLP_model.error_testing(X_train, T_train, num_hidden_list, alpha, "test", [50,100,200, 500, 1000], X_train, T_train) # Train
# plt.show()
# MLP_model.error_testing(X_train, T_train, num_hidden_list, alpha, "Validation", [50,100,200, 500, 1000], X_validation, T_validation) # Validation
# plt.show()

# 20% from a subset of classA for which classA(1,:)<0 and 80% from a subset of classA for which classA(1,:)>0
A_idx_list_cond1 = np.where((X[0,:] < 0) & (T==1))[0]
A_idx_list_cond2 = np.where((X[0,:] > 0) & (T==1))[0]
A_sample1_idx_list = np.random.choice(A_idx_list_cond1, int(0.2*A_idx_list_cond1.shape[0]), replace=False)
A_sample2_idx_list = np.random.choice(A_idx_list_cond2, int(0.2*A_idx_list_cond2.shape[0]), replace=False)
A_sample_idx_list = np.concatenate((A_sample1_idx_list, A_sample2_idx_list))
B_sample_idx_list = B_idx_list
train_idx_list = np.concatenate((A_sample_idx_list, B_sample_idx_list))
validation_idx_list = [i for i in range(X.shape[1]) if i not in train_idx_list]
X_train = X[:, train_idx_list]
T_train = T[train_idx_list]
X_validation = X[:, validation_idx_list]
T_validation = T[validation_idx_list]

MLP_model.error_testing(X_train, T_train, num_hidden_list, alpha, "test", [50,100,200, 500, 1000], X_train, T_train) # Train
plt.show()
MLP_model.error_testing(X_train, T_train, num_hidden_list, alpha, "Validation", [50,100,200, 500, 1000], X_validation, T_validation) # Validation
plt.show()
