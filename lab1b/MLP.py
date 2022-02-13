import data
import matplotlib.pyplot as plt
import numpy as np

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
    
    def backprop_seq(self, X, targets, theta1, theta2, num_hidden=2, alpha=0.1, momentum=0.9, epochs=10):
        X = self.add_bias(X)
        d0_theta,d1_theta = 0,0
        for _ in range(epochs):
            for i in range(X): 
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
X, T,_,_,n = data.generate_not_linearly_separable_data()
num_hidden = 8
epochs = 5000
alpha = 0.01
theta1 = MLP_model.init_theta(num_hidden, X.shape[0]+1)
theta2 = MLP_model.init_theta(X.shape[0], num_hidden+1)

new_theta1, new_theta2 = MLP_model.backprop(X,T, theta1, theta2, num_hidden, alpha=alpha, epochs=epochs, momentum=0.9)

w11, w12, bias1 = new_theta1[0,0], new_theta1[0,1], new_theta1[0,2]
w21, w22, bias2 = new_theta1[1,0], new_theta1[1,1], new_theta1[1,2]

plt.scatter(X[0,:], X[1,:], c=T[:])
x = np.linspace(np.amin(X[:,:]), np.amax(X[:,:]))
plt.plot(x, -(w11*x+bias1)/w12, label="Hidden layer 1")
plt.plot(x, -(w21*x+bias2)/w22, label="Hidden layer 2")
plt.legend()
plt.show()

# # 3.1.1 quesiton 1
# num_hidden_list = range(2,11)
# MLP_model.error_testing(X,T,num_hidden_list,alpha,"train",[1,10, 50, 100, 500], X,T)
# plt.show()

# ## 3.1.1 question 2
# A_idx_list = np.where(T==1)[0]
# B_idx_list = np.where(T==-1)[0]

# #25% from each class
# A_sample_idx_list = np.random.choice(A_idx_list, int(n*0.25), replace=False)
# B_sample_idx_list = np.random.choice(B_idx_list, int(n*0.25), replace=False)
# train_idx_list = np.concatenate((A_sample_idx_list, B_sample_idx_list))
# validation_idx_list = [i for i in range(X.shape[1]) if i not in train_idx_list]
# X_train = X[:, train_idx_list]
# T_train = T[train_idx_list]
# X_validation = X[:, validation_idx_list]
# T_validation = T[validation_idx_list]

# MLP_model.error_testing(X_train, T_train, num_hidden_list, alpha, "test", [1,10, 50, 100, 500], X_train, T_train) # Train
# plt.show()
# MLP_model.error_testing(X_train, T_train, num_hidden_list, alpha, "Validation", [1,10, 50, 100, 500], X_validation, T_validation) # Validation
# plt.show()

# #50% from class A
# A_sample_idx_list = np.random.choice(A_idx_list, int(n*0.50), replace=False)
# B_sample_idx_list = B_idx_list
# train_idx_list = np.concatenate((A_sample_idx_list, B_sample_idx_list))
# validation_idx_list = [i for i in range(X.shape[1]) if i not in train_idx_list]
# X_train = X[:, train_idx_list]
# T_train = T[train_idx_list]
# X_validation = X[:, validation_idx_list]
# T_validation = T[validation_idx_list]

# MLP_model.error_testing(X_train, T_train, num_hidden_list, alpha, "test", [1,10, 50, 100, 500], X_train, T_train) # Train
# plt.show()
# MLP_model.error_testing(X_train, T_train, num_hidden_list, alpha, "Validation", [1,10, 50, 100, 500], X_validation, T_validation) # Validation
# plt.show()

# #20% from a subset of classA for which classA(1,:)<0 and 80% from a subset of classA for which classA(1,:)>0
# A_idx_list_cond1 = np.where((X[0,:] < 0) & (T==1))[0]
# A_idx_list_cond2 = np.where((X[0,:] > 0) & (T==1))[0]
# A_sample1_idx_list = np.random.choice(A_idx_list_cond1, int(0.2*A_idx_list_cond1.shape[0]), replace=False)
# A_sample2_idx_list = np.random.choice(A_idx_list_cond2, int(0.8*A_idx_list_cond2.shape[0]), replace=False)
# A_sample_idx_list = np.concatenate((A_sample1_idx_list, A_sample2_idx_list))
# B_sample_idx_list = B_idx_list
# train_idx_list = np.concatenate((A_sample_idx_list, B_sample_idx_list))
# validation_idx_list = [i for i in range(X.shape[1]) if i not in train_idx_list]
# X_train = X[:, train_idx_list]
# T_train = T[train_idx_list]
# X_validation = X[:, validation_idx_list]
# T_validation = T[validation_idx_list]

# MLP_model.error_testing(X_train, T_train, num_hidden_list, alpha, "test", [1,10, 50, 100, 500], X_train, T_train) # Train
# plt.show()
# MLP_model.error_testing(X_train, T_train, num_hidden_list, alpha, "Validation", [1,10, 50, 100, 500], X_validation, T_validation) # Validation
# plt.show()

# Make an attempt at approximating the resulting decision boundary,i.e. where the network output is 0 (between the target labels of -1 and 1 for two classes, respectively).

# new_theta1, new_theta2 = MLP_model.backprop(X,T, theta1, theta2, num_hidden, alpha=alpha, epochs=epochs, momentum=0.9)
# fig = plt.figure(figsize=(5,5))
# x, y = np.mgrid[slice(-2,2,0.01), slice(-1,1,0.01)]
# x, y = np.concatenate(x), np.concatenate(y)
# x_plot = np.array([x,y])
# predictions_plot = MLP_model.forward(x_plot, new_theta1, new_theta2)
# plt.scatter(x_plot[0,:], x_plot[1,:], c=predictions_plot[0,:])
# plt.colorbar()

# # Plot hidden layer decision boundaries
# w11, w12, bias1 = new_theta1[0,0], new_theta1[0,1], new_theta1[0,2]
# w21, w22, bias2 = new_theta1[1,0], new_theta1[1,1], new_theta1[1,2]
# x = np.linspace(-2, 2, 1000)
# plt.plot(x, -(w11*x+bias1)/w12, c="red", label="hidden layer 1", linewidth=2)
# plt.plot(x, -(w21*x+bias2)/w22, c="cyan", label="hidden layer 2", linewidth=2)

# # Plot the true targets and rest of plot details
# plt.scatter(X[0,:], X[1,:], c=T)
# plt.title("Approximation of resulting decision boundry")
# plt.xlabel("X1")
# plt.ylabel("X2")
# plt.xlim((-2,2))
# plt.ylim((-1,1))
# plt.legend()
# plt.show()

# 3.1.3
x = np.arange(-5, 5, 0.5)
y = np.arange(-5, 5, 0.5)
xx, yy = np.meshgrid(x, y)
f = lambda x,y: np.exp(-(x**2+y**2)/10)-0.5
zz = f(xx, yy)

# 2D
fig = plt.figure(figsize=(10,5))
fig2d = fig.add_subplot(1,2,1)
fig2d.contourf(xx, yy, zz)
fig2d.set_xlabel("X1")
fig2d.set_ylabel("X2")

# 3D
fig3d = fig.add_subplot(1, 2, 2, projection='3d')
fig3d.plot_surface(xx, yy, zz)
fig3d.set_xlabel("X1")
fig3d.set_ylabel("X2")
fig3d.set_zlabel("f")
plt.show()

n = len(x)*len(y)
T = zz.reshape(1, n)
X = np.vstack((xx.reshape(1, n), yy.reshape(1, n)))
num_hidden_list = [1, 2, 4, 8, 16, 20, 25]
fig = plt.figure(figsize=(10,5))
plot = 1
for num_hidden in num_hidden_list:
    theta1 = MLP_model.init_theta(num_hidden, X.shape[0]+1)
    theta2 = MLP_model.init_theta(X.shape[0], num_hidden+1)
    new_theta1, new_theta2 = MLP_model.backprop(X,T, theta1, theta2, num_hidden, alpha=alpha, epochs=epochs, momentum=0.9)
    ax = fig.add_subplot(2,4, plot, projection='3d')
    plot += 1
    ax.plot_surface(xx,yy, MLP_model.forward(X, new_theta1, new_theta2)[1].reshape(20,20), color="green")
    ax.set_title("Representation using " + str(num_hidden) + " hidden nodes")
plt.show()

## Evaluate generalisation performance.

# Vary the number of nodes in the hidden layer from 1 to 25
train_idx = np.random.choice(range(n), int(n*0.6), replace=False)
validation_idx = [i for i in range(n) if i not in train_idx]
X_train = X[:, train_idx]
T_train = T[0,train_idx]
X_validation = X[:, validation_idx]
T_validation = T[0,validation_idx]
mse_train = []
mse_validation= []

for num_hidden in num_hidden_list:
    theta1 = MLP_model.init_theta(num_hidden, X.shape[0]+1)
    theta2 = MLP_model.init_theta(X.shape[0], num_hidden+1)
    new_theta1, new_theta2 = MLP_model.backprop(X,T, theta1, theta2, num_hidden, alpha=alpha, epochs=epochs, momentum=0.9)

    # MSE train
    predicted = MLP_model.forward(X_train,new_theta1, new_theta2)
    mse_train.append(np.mean((predicted-T_train)**2))

    # MSE Validation
    predicted = MLP_model.forward(X_validation,new_theta1, new_theta2)
    mse_validation.append(np.mean((predicted-T_validation)**2))

plt.plot(num_hidden_list, mse_train, label="Train")
plt.plot(num_hidden_list, mse_validation, label="Validation")
plt.xlabel("Num of Hidden Nodes")
plt.ylabel("MSE")
plt.title("MSE sampling using 60% of training data")
plt.legend()
plt.xticks(num_hidden_list)
plt.show()

# Find best for 8 hidden
num_hidden = 8
theta1 = MLP_model.init_theta(num_hidden, X.shape[0]+1)
theta2 = MLP_model.init_theta(X.shape[0], num_hidden+1)
alpha_list = [0.01,0.01,0.1,0.3,0.5]
epochs_list = [50,100,1000,5000]
best_alpha = -np.inf
best_epochs = -np.inf
best_error = np.inf
for alpha in alpha_list:
    for epochs  in epochs_list:
        new_theta1, new_theta2 = MLP_model.backprop(X,T, theta1, theta2, num_hidden, alpha=alpha, epochs=epochs, momentum=0.9)

        predicted = MLP_model.forward(X_validation, new_theta1, new_theta2)
        mse = np.mean((predicted-T_validation)**2)
        if mse < best_error:
            best_alpha = alpha
            best_epochs = epochs
            best_error = mse
        if best_error == 0:
            break

print("Best learning rate: " + str(best_alpha))
print("Best epcohs: " + str(best_epochs))
print("MSE: " + str(best_error))

# # training with best parameters GAV UPP PÅ DENNA DEL FÖR PLOTTEN FUNKADE INTE PALLADE INTE VI HAR PRINT AV MSE VILKEN BORDE RÄCKA
# new_theta1, new_theta2 = MLP_model.backprop(X,T, theta1, theta2, num_hidden, alpha=best_alpha, epochs=best_epochs, momentum=0.9)

# predictions = MLP_model.forward(X, new_theta1, new_theta2)

# # predictions plot
# fig = plt.figure(figsize=(10,5))
# fig3= fig.add_subplot(2,2,1, projection='3d')
# fig3.plot_surface(xx,yy,predictions[1].reshape(20,20), color="coral", alpha=0.5)
# fig3.set_xlabel("X1")
# fig3.set_ylabel("X2")
# fig3.set_zlabel("f")
# fig3.set_title("Original (blue) vs predicted (coral)")

# plt.show()

# For the selected ”best” model, run experiments with varying number of the training samples, e.g. from 80% down to 20% of all the dataset.
num_hidden = 8
nsamps = np.linspace(0.2, 0.8, 7, endpoint=True)
mse_train = []
mse_validation = []

for nsamp in nsamps:
    train_idx = np.random.choice(range(n), int(n*nsamp), replace=False)
    validation_idx = [i for i in range(n) if i not in train_idx]
    X_train = X[:, train_idx]
    T_train = T[0,train_idx]
    X_validation = X[:, validation_idx]
    T_validation = T[0,validation_idx]

    theta1 = MLP_model.init_theta(num_hidden, X.shape[0]+1)
    theta2 = MLP_model.init_theta(X.shape[0], num_hidden+1)
    new_theta1, new_theta2 = MLP_model.backprop(X_train,T_train, theta1, theta2, num_hidden, alpha=best_alpha, epochs=best_epochs, momentum=0.9)

    # MSE train
    predicted = MLP_model.forward(X_train,new_theta1, new_theta2)
    mse_train.append(np.mean((predicted-T_train)**2))

    # MSE Validation
    predicted = MLP_model.forward(X_validation,new_theta1, new_theta2)
    mse_validation.append(np.mean((predicted-T_validation)**2))

plt.plot(nsamps, mse_train, label="Train")
plt.plot(nsamps, mse_validation, label="Validation")
plt.xlabel("Num of samples (% of training data)")
plt.ylabel("MSE")
plt.title("MSE best model (with sampling)")
plt.legend()
plt.xticks(nsamps)
plt.show()

# For the ”best” model, can you speed up the convergence without compromising the generalisation performance?
nsamp = 0.8
train_idx = np.random.choice(range(n), int(n*nsamp), replace=False)
validation_idx = [i for i in range(n) if i not in train_idx]
X_train = X[:, train_idx]
T_train = T[0,train_idx]
X_validation = X[:, validation_idx]
T_validation = T[0,validation_idx]

epochs_list = [10,20,30,40,50,60,70,80,90,100]
mse_train = []
mse_validation = []

for epochs in epochs_list:
    theta1 = MLP_model.init_theta(num_hidden, X.shape[0]+1)
    theta2 = MLP_model.init_theta(X.shape[0], num_hidden+1)
    new_theta1, new_theta2 = MLP_model.backprop(X_train,T_train, theta1, theta2, num_hidden, alpha=best_alpha, epochs=epochs, momentum=0.9)

    # MSE train
    predicted = MLP_model.forward(X_train,new_theta1, new_theta2)
    mse_train.append(np.mean((predicted-T_train)**2))

    # MSE Validation
    predicted = MLP_model.forward(X_validation,new_theta1, new_theta2)
    mse_validation.append(np.mean((predicted-T_validation)**2))

plt.plot(epochs_list, mse_train, label="Train")
plt.plot(epochs_list, mse_validation, label="Validation")
plt.xlabel("Num of epochs")
plt.ylabel("MSE")
plt.title("MSE best model (with 80% sampling)")
plt.legend()
plt.xticks(epochs_list)
plt.show()