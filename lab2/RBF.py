import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

def rbf(x, mean, variance):
    phi = np.zeros((x.shape[0], mean.shape[0]))
    for i in range(x.shape[0]):
        for j in range(mean.shape[0]):
            phi[i,j] = np.exp(-(x[i] - mean[j]) ** 2 / (2 * (variance ** 2)))

    return phi

def phi_RLS(x,clusters,variance):
    return np.fromfunction(lambda i,j: phi(x, clusters, variance),(len(clusters),0))

def phi(x, mean, variance):
    return np.exp(-(x - mean) ** 2 / (2 * (variance ** 2)))

def k_means(x, lr, k, epochs):
    clusters = x[np.random.choice(len(x), k)]

    for _ in tqdm(range(epochs)):
        point = x[np.random.randint(0, len(x))]
        distances = []

        for node in clusters:
            distances.append(np.linalg.norm(node - point))

        winner = np.argmin(distances) 
        clusters[winner] += lr * (point - clusters[winner])

    return clusters
class RBF:

    def __init__(self, lr, variance, epochs, k, competitive_learning, batch_mode) -> None:
        self.lr = lr
        self.variance = variance
        self.epochs = epochs
        self.k = k
        self.competitive_learning = competitive_learning
        self.batch_mode = batch_mode

    def fit(self, x, y):
        if self.competitive_learning:
            clusters = k_means(X.copy(), self.lr, self.k, self.epochs)
        else:
            clusters = np.array([[np.random.choice(x[np.random.randint(0, len(x))])] for _ in range(self.k)])

        phi = rbf(x, clusters, self.variance)
        if(self.batch_mode):
            w = np.dot(np.linalg.pinv(np.dot(phi.T, phi)), np.dot(phi.T, y))
        else:
            #recursive least square
            w = np.array([0.001 for _ in range(self.k)])[:,np.newaxis]
            for i in range(len(x)):
                w += self.lr * (y[i] - self.f_tilde(x[i], w, clusters, self.variance))*phi_RLS(x[i],clusters,self.variance)
        return w, clusters

    def predict(self, x, mean, w):
        phi = rbf(x, mean, self.variance)
        return np.dot(phi, w)

    def accuracy(self, pred, y):
        return np.sum((pred - y)**2) / len(y)

    def f_tilde(self,x, W, clusters, variance):
        f_approximate = 0.0
        for i in range(len(clusters)):
            f_approximate += W[i]*phi(x,clusters[i],variance)
        return f_approximate[0]

sin2x = True
square2x = not sin2x
competitive_learning = False
batch_mode = False

if sin2x:
    X = np.arange(0, 2 * np.pi, 0.1)[:, np.newaxis]
    y = np.sin(2 * X)
    X_test = np.arange(0.05, 2 * np.pi, 0.1)[:, np.newaxis]
    y_test = np.sin(2 * X_test)

elif square2x:
    X = np.arange(0, 2 * np.pi, 0.1)[:, np.newaxis]
    y = np.zeros(len(X))
    y_sin = np.sin(2*X)
    for i in range(len(y_sin)):
        if y_sin[i] >= 0:
            y[i] = 1
        else:
            y[i] = -1
    X_test = np.arange(0.05, 2 * np.pi, 0.1)[:, np.newaxis]
    y_test = np.zeros(len(X_test))
    y_sin_test = np.sin(2*X_test)
    for i in range(len(y_sin_test)):
        if y_sin_test[i] >= 0:
            y_test[i] = 1
        else:
            y_test[i] = -1

sigmas = [0.05,0.1,0.15,0.20,1]
#clusters = 100
variance = 0.1
lr = 0.1
epochs = 10000
ks= [50, 100, 200]
plt.figure(figsize=(13, 10))
for sigma in sigmas:
    model = RBF(lr, variance, epochs, 200, competitive_learning, batch_mode)
    w, clusters = model.fit(X, y)
    y_predicted = model.predict(X_test, clusters, w)
    accuracy = mae(y_test, y_predicted)
    print("Accuracy for " + str(200) + " components and sigma " + str(sigma)+ " as mean absolute error: " + str(accuracy))
    if not (sigma>0.05):
        plt.plot(X, y, label='Real output')
    plt.plot(X_test, y_predicted, label= "sigma = " + str(sigma))
    plt.legend()
plt.show()