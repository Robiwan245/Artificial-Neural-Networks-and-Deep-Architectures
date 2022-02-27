from audioop import reverse
from cProfile import label
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sympy import symmetric_poly
from tqdm import tqdm

class HOP:
    def __init__(self, bias = 0, rho = 0) -> None:
        self.theta = None
        self.rho = rho
        self.bias = bias

    def train(self, X):
        if gauss_weights:
            self.theta = np.array([[np.random.normal(0, 1) for _ in range(len(X[0]))] for _ in range(len(X[0]))])
        elif symmetric_weights:
            #self.theta = np.array([[0.5 for _ in range(len(X[0]))] for _ in range(len(X[0]))])
            #self.theta = self.theta * np.identity(len(self.theta))
            self.theta = np.array([[np.random.normal(0, 1) for _ in range(len(X[0]))] for _ in range(len(X[0]))])
            self.theta = 0.5 * (self.theta + self.theta.T)
        elif sparse:
            self.theta = np.sum([(x.reshape((-1,1)) - self.rho)@(x.reshape((1,-1)) - self.rho) for x in X], axis= 0)
            print(self.theta)
        else:
            self.theta = np.sum([x.reshape((-1,1))@x.reshape((1,-1)) for x in X], axis= 0)
            # self.theta = 0.5 * (self.theta + self.theta.T)
            # print(self.theta)

        count = 0
        for x in X:
            x_recalled = self.recall(x, synchronous=True)
            if np.array_equal(x, x_recalled):
                count += 1
            else:
                break
        if(count == 3):
            print("All patterns memorized")
        #else:
            #print("Memorization unsuccessful since one or more patterns aren't fixed points")

    def sign(self, x):
        return 1 if x>= 0 else -1

    def update(self, x, synchronous=False):
        if synchronous:
            x_updated = self.theta@x
            x_updated[x_updated>=0] = 1
            x_updated[x_updated < 0] = -1
        else: 
            x_updated = x.copy()
            for i in range(x.shape[0]):
                if sparse:
                    x_updated[i] = 0.5 + 0.5*self.sign(np.sum((self.theta[i,:]*x) - self.bias))
                else:
                    x_updated[i] = self.sign(np.sum(self.theta[i,:]*x))
        return x_updated

    def recall(self, x, synchronous=False, max_iter = 10):
        iter = 0
        es = []
        while iter<max_iter:
            e = self.energy(x)
            es.append(e)
            print(e)
            x_updated = self.update(x, synchronous)

            if np.array_equal(x, x_updated):
                break
            else:
                iter += 1
                x = x_updated

        return x_updated, es
    
    def check_recall(self, X, X_expected, synchronous = False):
        X_recalled = self.recall(X, synchronous)
        same = np.array_equal(X_recalled, X_expected)
        print(X_recalled, " -> ", X_expected, " ", "Successfully recalled" if same else "Failed recall" )
    
    def find_attractors(self, x=[]):
        attractors = set()
        if len(x) == 0:
            all_patterns = [list(i) for i in itertools.product([-1,1], repeat=8)]
        else:
            all_patterns = np.array([x])
        for pattern in all_patterns:
            attractors.add(tuple(self.recall(pattern, synchronous=True)))
        
        return np.array([attractor for attractor in attractors])
    
    def display_pattern(self, x):
        x = x.reshape(32,32)
        plt.imshow(x)
        plt.title("Pattern")
        plt.show()

    def energy(self, x):
        E = 0
        for i in range(len(x)):
            for j in range(len(x)):
                E += - np.dot(self.theta[i, j], np.dot(x[i], x[j]))
        return E

gauss_weights = False
symmetric_weights = False
sparse = False

# X = np.array([[-1,-1,1,-1,1,-1,-1,1],
#                 [-1,-1,-1,-1,-1,1,-1,-1],
#                 [-1,1,1,-1,-1,1,-1,1]])
model = HOP()
# model.train(X)

# X_distorted = np.array([[1, -1, 1, -1, 1, -1, -1, 1],
#                         [1, 1, -1, -1, -1, 1, -1, -1],
#                         [1, 1, 1, -1, 1, 1, -1, 1]])

# # 3.1
# print("Synchronous")
# for x_distorted, x in zip(X_distorted, X):                    
#     model.check_recall(x_distorted, x, synchronous = True)
# print("Asynchronous")
# for x_distorted, x in zip(X_distorted, X):   
#     model.check_recall(x_distorted, x, synchronous = False)

# attractors = model.find_attractors()
# print("Found {} attractors:".format(len(attractors)))
# for attractor in attractors:
#     print(attractor)
# #3.3
#     E_for_attractors = model.energy(attractor)
#     print(E_for_attractors)

# 3.2
pict = np.genfromtxt("pict.dat", delimiter = ',')
data = pict.reshape(11, 1024)

p1 = data[0]
p2 = data[1]
p3 = data[2]
p4 = data[3]
p5 = data[4]
p6 = data[5]
p7 = data[6]
p8 = data[7]
p9 = data[8]
p10 = data[9]
p11 = data[10]

#model.display_pattern(p2)
#model.display_pattern(p10)

X = np.array([p1,p2,p3, p4, p5, p6, p7, p8, p9, p10, p11])
model.train(X)
new_p10 = model.update(p1,True)
if (np.array_equal(p1, new_p10)):
    print("Same")
model.display_pattern(p1)
model.display_pattern(p10)
model.display_pattern(new_p10)

#3.3
# E_for_pattern = model.energy(new_p10)
# print(E_for_pattern)
# for i in tqdm(range(data.shape[0])):
#     E_for_pattern = model.energy(data[i])
#     print("Energy for p" + str(i) + ":" + str(E_for_pattern))

energies = [[] for i in range(data.shape[0])]
plt.figure()

for i in tqdm(range(data.shape[0])):
    print("Energy for pattern p" + str(i) + ":")
    _, es = model.recall(data[i])
    energies[i].append(es)

plt.plot(energies[0])
plt.plot(energies[1])
plt.plot(energies[2])
plt.plot(energies[3])
plt.plot(energies[4])
plt.plot(energies[5])
plt.plot(energies[6])
plt.plot(energies[7])
plt.plot(energies[8])
plt.plot(energies[9])
plt.plot(energies[10])
plt.show()

#3.4
# epochs = 1
# noice_amount_max = 1024
# able_to_recall = [[], [], []]
# atts = [p1, p2, p3]
# index = 0

# for att in atts:
#     attractor_to_test = att
#     for _ in tqdm(range(1)):
#         noice_amount = 0
#         while noice_amount <= noice_amount_max:
#             attractor_to_test_new = attractor_to_test.copy()
#             for i in range(noice_amount):
#                 if attractor_to_test[i] == 1:
#                     attractor_to_test_new[i] = -1
#                 if attractor_to_test[i] == -1:
#                     attractor_to_test_new[i] = 1
#             x_new = model.recall(attractor_to_test_new, False)
#             # attractor_new = find_attractors(x_new)
#             # attractor_real = find_attractors(attractor_to_test)
#             # print("Real: " + str(attractor_real) + " and new: " + str(attractor_new))
#             if np.array_equal(x_new, attractor_to_test):
#                 able_to_recall[index].append(1)
#             elif np.array_equal(x_new, p2):
#                 print("Converged to p2")
#             elif np.array_equal(x_new, p3):
#                 print("Converged to p3")
#             elif np.array_equal(x_new, p4):
#                 print("Converged to p4")
#             elif np.array_equal(x_new, p5):
#                 print("Converged to p5")
#             elif np.array_equal(x_new, p6):
#                 print("Converged to p6")
#             elif np.array_equal(x_new, p7):
#                 print("Converged to p7")
#             elif np.array_equal(x_new, p8):
#                 print("Converged to p8")
#             elif np.array_equal(x_new, p9):
#                 print("Converged to p9")
#             elif np.array_equal(x_new, p10):
#                 print("Converged to p10")
#             elif np.array_equal(x_new, p11):
#                 print("Converged to p11")
#             else:
#                 #print("Dropped at " + str(noice_amount))
#                 sum = 0
#                 for i in range(len(attractor_to_test)):
#                     if attractor_to_test[i] == x_new[i]:
#                         sum += 1
#                     else:
#                         sum += 0
#                 able_to_recall[index].append(sum/len(attractor_to_test))
#             # x = x_new.reshape(32,32)
#             # plt.imshow(x)
#             # plt.title("Pattern")
#             # plt.show()
#             noice_amount += 1
#         index += 1

# attractor_new = model.find_attractors(x = x_new)
# attractor_real = model.find_attractors(x = attractor_to_test)
# print("Real: " + str(attractor_real) + " and new: " + str(attractor_new))

# if np.array_equal(x_new, p2):
#     print("Converged to p2")
# elif np.array_equal(x_new, p3):
#     print("Converged to p3")
# elif np.array_equal(x_new, p4):
#     print("Converged to p4")
# elif np.array_equal(x_new, p5):
#     print("Converged to p5")
# elif np.array_equal(x_new, p6):
#     print("Converged to p6")
# elif np.array_equal(x_new, p7):
#     print("Converged to p7")
# elif np.array_equal(x_new, p8):
#     print("Converged to p8")
# elif np.array_equal(x_new, p9):
#     print("Converged to p9")
# elif np.array_equal(x_new, p10):
#     print("Converged to p10")
# elif np.array_equal(x_new, p11):
#     print("Converged to p11")

# plt.figure()
# plt.plot(able_to_recall[0], label = "p1")
# plt.plot(able_to_recall[1], label = "p2")
# plt.plot(able_to_recall[2], label = "p3")
# plt.ylabel("Part of patterns correctly recalled [percentage]")
# plt.xlabel("Amount of noise")
# plt.legend()
# plt.show()

# model.display_pattern(x_new)

#3.6
sparse = True

N = 100
amount_of_sparse_patterns = 50

sparse_patterns = np.zeros((amount_of_sparse_patterns, N))

amount_of_successes = 0
amount_of_successes_list = [[], [], []]
bias_for_success = [[], [], []]

for index in range(3):

    if index == 0:
        sparsness = 10
        rho = 0.1
    elif index == 1:
        sparsness = 20
        rho = 0.05
    elif index == 2:
        sparsness = 100
        rho = 0.01

    amount_of_successes = 0
    failed = False

    for patt in tqdm(range(1, 101)):

            amount_of_sparse_patterns = patt

            sparse_patterns = np.zeros((amount_of_sparse_patterns, N))

        #for _ in range(10):

            for i in range(amount_of_sparse_patterns):
                for j in range(N):
                    if j % sparsness == 0:
                        sparse_patterns[i,j] = 1
                    else:
                        sparse_patterns[i,j] = 0
                np.random.shuffle(sparse_patterns[i])

            success = False
            bias = 0

            model = HOP(bias = bias, rho = rho)
            model.train(sparse_patterns)
            #model.train([sparse_patterns[0], sparse_patterns[1], sparse_patterns[2], sparse_patterns[3], sparse_patterns[4], sparse_patterns[5], sparse_patterns[6], sparse_patterns[7], sparse_patterns[8], sparse_patterns[9]])
            recalled_pattern = model.recall(sparse_patterns[0])
            while success == False:
                if np.array_equal(recalled_pattern, sparse_patterns[0]):
                    #print("Success!")
                    amount_of_successes += 1
                    bias_for_success[index].append(bias)
                    success = True
                elif bias <= 1:
                    bias += 0.001
                    model = HOP(bias = bias, rho = rho)
                    model.train(sparse_patterns)
                    #model.train([sparse_patterns[0], sparse_patterns[1], sparse_patterns[2], sparse_patterns[3], sparse_patterns[4], sparse_patterns[5], sparse_patterns[6], sparse_patterns[7], sparse_patterns[8], sparse_patterns[9]])
                    recalled_pattern = model.update(sparse_patterns[0])
                else:
                    print("Unable to store")
                    failed = True
                    break
            if failed:
                break
            amount_of_successes_list[index].append(amount_of_successes)

amount_of_successes_list[0].reverse()
amount_of_successes_list[1].reverse()
amount_of_successes_list[2].reverse()

plt.figure()
plt.plot(amount_of_successes_list[0], label = "rho = 0.1")
#plt.plot(amount_of_successes_list[1], label = "rho = 0.05")
#plt.plot(amount_of_successes_list[2], label = "rho = 0.01")
plt.ylabel("Amount of successes")
plt.xlabel("Amount of patterns to store")
plt.legend()
plt.show()

plt.figure()
plt.plot(bias_for_success[0], label = "rho = 0.1")
plt.plot(bias_for_success[1], label = "rho = 0.05")
plt.plot(bias_for_success[2], label = "rho = 0.01")
plt.xlabel("Amount of patterns to store")
plt.ylabel("Bias")
plt.legend()
plt.show()

print(amount_of_successes)
print(bias_for_success)
