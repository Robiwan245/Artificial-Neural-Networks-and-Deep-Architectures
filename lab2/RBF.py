import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

def rbf(x, mean, variance):
    phi = np.zeros((x.shape[0], mean.shape[0]))
    for i in range(x.shape[0]):
        for j in range(mean.shape[0]):
            phi[i,j] = np.sum(np.exp(-(x[i] - mean[j]) ** 2 / (2 * variance ** 2)))

    return phi

def k_means(x, lr, k, epochs):
    rbf_nodes = x[np.random.choice(len(x), k)]

    for _ in tqdm(range(epochs)):
        point = [np.random.choice(x[np.random.randint(0, len(x))])]
        distances = []

        for node in rbf_nodes:
            distances.append(np.linalg.norm(node - point))

        winner = np.argmin(distances) 
        rbf_nodes[winner] += lr * (point - rbf_nodes[winner])

    return rbf_nodes

class RBF:

    def __init__(self, lr, variance, epochs, k) -> None:
        self.lr = lr
        self.variance = variance
        self.epochs = epochs
        self.k = k

    def fit(self, x, y):
        clusters = k_means(X.copy(), self.lr, self.k, self.epochs)

        phi = rbf(x, clusters, self.variance)
        w = np.dot(np.linalg.pinv(np.dot(phi.T, phi)), np.dot(phi.T, y))
        return w, clusters

    def predict(self, x, mean, w):
        phi = rbf(x, mean, self.variance)
        return np.dot(phi, w)

X = np.arange(0, 2 * np.pi, 0.1)[:, np.newaxis]
y = np.sin(2 * X)
X_test = np.arange(0.05, 2 * np.pi, 0.1)[:, np.newaxis]
y_test = np.sin(2 * X_test)

rbf_nodes = 100
variance = 0.1
lr = 0.2
epochs = 10000
k = 25

model = RBF(lr, variance, epochs, k)
w, clusters = model.fit(X, y)
y_predicted = model.predict(X_test, clusters, w)

plt.plot(X, y, label='Real output')
plt.plot(X_test, y_predicted, label='Prediction')
plt.legend()
plt.show()