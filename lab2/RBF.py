from cProfile import label
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

def rbf(x, mean, variance):
    phi = np.zeros((x.shape[0], mean.shape[0]))
    for i in range(x.shape[0]):
        for j in range(mean.shape[0]):
            phi[i,j] = np.sum(np.exp(-(x[i] - mean[j]) ** 2 / (2 * variance ** 2)))
    return phi

def k_means(x, lr, k, epochs, dead_node):
    clusters = x[np.random.choice(len(x), k)]

    for _ in tqdm(range(epochs)):
        point = [np.random.choice(x[np.random.randint(0, len(x))])]
        distances = []

        for node in clusters:
            distances.append(np.linalg.norm(node - point))

        winner = np.argmin(distances) 
        clusters[winner] += lr * (point - clusters[winner])

    if dead_node:
        loser = np.argmax(distances)
        clusters[loser] += lr**2 * (point - clusters[loser])

    return clusters

class RBF:

    def __init__(self, lr, variance, epochs, k, competitive_learning, batch, dead_node):
        self.lr = lr
        self.variance = variance
        self.epochs = epochs
        self.k = k
        self.competitive_learning = competitive_learning
        self.batch = batch
        self.dead_node = dead_node

    def fit(self, x, y):
        if self.competitive_learning:
            clusters = k_means(X.copy(), self.lr, self.k, self.epochs, self.dead_node)
        else:
            clusters = np.array([[np.random.choice(x[np.random.randint(0, len(x))])] for i in range(self.k)])
            #clusters = np.array([np.random.randint(0, 5, self.k)])
        phi = rbf(x, clusters, self.variance)
        w = np.dot(np.linalg.pinv(np.dot(phi.T, phi)), np.dot(phi.T, y))
        return w, clusters

    def predict(self, x, mean, w):
        phi = rbf(x, mean, self.variance)
        return np.dot(phi, w)

    def accuracy(self, pred, y):
        return np.sum((pred - y)**2) / len(y)

sin2x = False
square2x = False
competitive_learning = True
noise = True
batch = False
dead_node = True
ballist = True

if sin2x:
    X = np.arange(0, 2 * np.pi, 0.1)[:, np.newaxis]
    y = np.sin(2 * X)
    X_test = np.arange(0.05, 2 * np.pi, 0.1)[:, np.newaxis]
    y_test = np.sin(2 * X_test)
    if noise:
        noise_list = np.array([np.random.normal(0, 0.1, size = len(y))]).T
        noise_list_test = np.array([np.random.normal(0, 0.1, size = len(y))]).T
        for i in range(len(y)):
            y[i] = y[i] + noise_list[i]
            y_test[i] = y_test[i] + noise_list_test[i]

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
    if noise:
        noise_list = np.array([np.random.normal(0, 0.1, size = len(y))]).T
        noise_list_test = np.array([np.random.normal(0, 0.1, size = len(y))]).T
        for i in range(len(y)):
            y[i] = y[i] + noise_list[i]
            y_test[i] = y_test[i] + noise_list_test[i]

if ballist:
        train = np.loadtxt("data/ballist.dat")
        test = np.loadtxt("data/balltest.dat")

        X = train[:, :2]
        y = train[:, 2:]
        X_test = test[:, :2]
        y_test = test[:, 2:]

if noise and ballist:
        for i in range(y.shape[0]):
            y[i] = y[i] + np.random.normal(0, 0.1)
        for i in range(y_test.shape[0]):
            y_test[i] = y_test[i] + np.random.normal(0, 0.1)

# plt.plot(X, y, label='Real output')
# plt.legend()
# plt.show()

variance = 0.1
lr = 0.2
epochs = 10000
ks= [50, 100, 200]

fig, ax = plt.subplots()

ax.scatter(X, y, label='Real output')
acc = 0

print(X)
print(y)

for _ in range(10):
    #for k in ks:

        model = RBF(lr, variance, epochs, 200, competitive_learning, batch, dead_node)
        w, clusters = model.fit(X, y)
        y_predicted = model.predict(X_test, clusters, w)
        accuracy = mae(y_test, y_predicted)
        acc += accuracy
        print("Accuracy for " + str(200) + " components as mean absolute error: " + str(accuracy))

        ax.scatter(X_test, y_predicted, label=(str(200) + ' hidden nodes'))
        #plt.title("Approximation of sine wave by RBF without CL with " + str(k) + " clusters")
        legend = ax.legend(loc="upper right")
#print(acc / 10)
        plt.show()