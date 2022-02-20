import itertools
import numpy as np
import matplotlib.pyplot as plt
from sympy import symmetric_poly
from tqdm import tqdm

class HOP:
    def __init__(self) -> None:
        self.theta = None

    def train(self, X):
        if gauss_weights:
            self.theta = np.array([[np.random.normal(0, 1) for _ in range(len(X[0]))] for _ in range(len(X[0]))])
        elif symmetric_weights:
            self.theta = np.array([[0.5 for _ in range(len(X[0]))] for _ in range(len(X[0]))])
        else:
            self.theta = np.sum([x.reshape((-1,1))@x.reshape((1,-1)) for x in X], axis= 0)

        count = 0
        for x in X:
            x_recalled = self.recall(x, synchronous=True)
            if np.array_equal(x, x_recalled):
                count += 1
            else:
                break
        if(count == 3):
            print("All patterns memorized")
        else:
            print("Memorization unsuccessful since one or more patterns aren't fixed points")

    def sign(self, x):
        return 1 if x>= 0 else -1

    def update(self, x, synchronous=True):
        if synchronous:
            x_updated = self.theta@x
            x_updated[x_updated>=0] = 1
            x_updated[x_updated < 0] = -1
        else: 
            x_updated = x.copy()
            for i in range(x.shape[0]):
                x_updated[i] = self.sign(np.sum(self.theta[i,:]*x))
        return x_updated

    def recall(self, x, synchronous=False, max_iter = 10):
        iter = 0
        while iter<max_iter:
            e = self.energy(x)
            print(e)
            x_updated = self.update(x, synchronous)

            if np.array_equal(x, x_updated):
                break
            else:
                iter += 1
                x = x_updated

        return x_updated
    
    def check_recall(self, X, X_expected, synchronous = False):
        X_recalled = self.recall(X, synchronous)
        same = np.array_equal(X_recalled, X_expected)
        print(X_recalled, " -> ", X_expected, " ", "Successfully recalled" if same else "Failed recall" )
    
    def find_attractors(self):
        attractors = set()
        all_patterns = [list(i) for i in itertools.product([-1,1], repeat=8)]
        for pattern in all_patterns:
            attractors.add(tuple(self.recall(pattern, synchronous=True)))
        print("________")
        print(attractors)
        print("________")
        
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

X = np.array([[-1,-1,1,-1,1,-1,-1,1],
                [-1,-1,-1,-1,-1,1,-1,-1],
                [-1,1,1,-1,-1,1,-1,1]])
model = HOP()
model.train(X)

X_distorted = np.array([[1, -1, 1, -1, 1, -1, -1, 1],
                        [1, 1, -1, -1, -1, 1, -1, -1],
                        [1, 1, 1, -1, 1, 1, -1, 1]])

# 3.1
print("Synchronous")
for x_distorted, x in zip(X_distorted, X):                    
    model.check_recall(x_distorted, x, synchronous = True)
print("Asynchronous")
for x_distorted, x in zip(X_distorted, X):   
    model.check_recall(x_distorted, x, synchronous = False)

attractors = model.find_attractors()
print("Found {} attractors:".format(len(attractors)))
for attractor in attractors:
    print(attractor)
#3.3
    E_for_attractors = model.energy(attractor)
    print(E_for_attractors)

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

#model.display_pattern(p1)
#model.display_pattern(p10)

X = np.array([p1,p2,p3])
model.train(X)
new_p10 = model.update(p10,True)
if (np.array_equal(p1, new_p10)):
    print("Same")
model.display_pattern(p1)
model.display_pattern(new_p10)

#3.3
# E_for_pattern = model.energy(new_p10)
# print(E_for_pattern)
# for i in tqdm(range(data.shape[0])):
#     E_for_pattern = model.energy(data[i])
#     print("Energy for p" + str(i) + ":" + str(E_for_pattern))

for i in range(data.shape[0]):
    print("Energy for pattern p" + str(i) + ":")
    model.recall(data[i])

#3.4
epochs = 1
noice_amount_max = 1024
able_to_recall = []

attractor_to_test = p1
for _ in tqdm(range(epochs)):
    noice_amount = 0
    while noice_amount <= noice_amount_max:
        attractor_to_test_new = attractor_to_test.copy()
        for i in range(noice_amount):
            if attractor_to_test[i] == 1:
                attractor_to_test_new[i] = -1
            if attractor_to_test[i] == -1:
                attractor_to_test_new[i] = 1
        x_new = model.update(attractor_to_test_new, False)
        # attractor_new = find_attractors(x_new)
        # attractor_real = find_attractors(attractor_to_test)
        # print("Real: " + str(attractor_real) + " and new: " + str(attractor_new))
        if np.array_equal(x_new, attractor_to_test):
            able_to_recall.append(1)
        else:
            sum = 0
            for i in range(len(attractor_to_test)):
                if attractor_to_test[i] == x_new[i]:
                    sum += 1
                else:
                    sum += 0
            able_to_recall.append(sum/len(attractor_to_test))
        # x = x_new.reshape(32,32)
        # plt.imshow(x)
        # plt.title("Pattern")
        # plt.show()
        noice_amount += 1

plt.figure()
plt.plot(able_to_recall)
plt.show()