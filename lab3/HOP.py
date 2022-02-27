import itertools
import numpy as np
import matplotlib.pyplot as plt
from sympy import symmetric_poly
from tqdm import tqdm
import random

class HOP:
    def __init__(self) -> None:
        self.theta = None
        self.pos_to_update = None

    def train(self, X, gauss_weights=False, symmetric_weights=False, N=3, check=True):
        if gauss_weights:
            self.theta = np.array([[np.random.normal(0, 1) for _ in range(len(X[0]))] for _ in range(len(X[0]))])
        elif symmetric_weights:
            self.theta = np.array([[0.5 for _ in range(len(X[0]))] for _ in range(len(X[0]))])
        else:
            self.theta = np.sum([x.reshape((-1,1))@x.reshape((1,-1)) for x in X], axis= 0)

        if check:
            count = 0
            for x in X:
                x_recalled, _ = self.recall(x, True,100,False,False)
                if np.array_equal(x, x_recalled):
                    count += 1
                else:
                    break
            if(count == N):
                print("All patterns memorized")
            else:
                print("Memorization unsuccessful since one or more patterns aren't fixed points")

    def sign(self, x):
        return 1 if x>= 0 else -1

    def update(self, x, synchronous=False, isRandom=False):
        if synchronous:
            x_updated = self.theta@x
            x_updated[x_updated>=0] = 1
            x_updated[x_updated < 0] = -1
        else: 
            x_updated = x.copy()
            if not self.pos_to_update:
                self.pos_to_update = [i for i in range(x.shape[0])]
            if isRandom:
                for _ in self.pos_to_update:
                    i = np.random.choice(self.pos_to_update)
                    x_updated[i] = self.sign(np.sum(self.theta[i,:]*x))
            else:
                for i in range(x.shape[0]):
                    x_updated[i] = self.sign(np.sum(self.theta[i,:]*x))
        return x_updated

    def recall(self, x, synchronous=False, max_iter = 1, isRandom=False, disp=False):
        iter = 0
        while iter<max_iter:
            x_updated = self.update(x, synchronous, isRandom=isRandom)
            if disp:
                self.display_pattern(x_updated)

            if np.array_equal(x, x_updated) and not disp:
                if iter == 0:
                    iter += 1
                break
            else:
                iter += 1
                x = x_updated

        return x_updated, iter
    
    def check_recall(self, X, X_expected, synchronous = False, max_iter=10):
        X_recalled,_ = self.recall(X, synchronous, max_iter=max_iter)
        same = np.array_equal(X_recalled, X_expected)
        print(X_recalled, " -> ", X_expected, " ", "Successfully recalled" if same else "Failed recall" )
    
    def find_attractors(self, isPatterns=False):
        attractors = set()
        if isPatterns:
            all_patterns = [list(i) for i in itertools.product([-1,1], repeat=1024)]
        else:
            all_patterns = [list(i) for i in itertools.product([-1,1], repeat=8)]
        for pattern in all_patterns:
            attractors.add(tuple(self.recall(pattern, synchronous=True)[0]))
        
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

X = np.array([[-1,-1,1,-1,1,-1,-1,1],
                [-1,-1,-1,-1,-1,1,-1,-1],
                [-1,1,1,-1,-1,1,-1,1]])
model = HOP()
model.train(X, N=3)
X_distorted = np.array([[1, -1, 1, -1, 1, -1, -1, 1],
                        [1, 1, -1, -1, -1, 1, -1, -1],
                        [1, 1, 1, -1, 1, 1, -1, 1]])

# # 3.1
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

# 5 bit error
X_5bit_err = np.array([[1,1,-1,1,-1,-1,-1,-1],
                        [-1,-1,1,1,1,1,-1,-1],
                        [-1,-1,-1,1,1,1,-1,-1]])

print("5 bit error:")
print("Synchronous")
for x_5bit_err, x in zip(X_5bit_err, X):                    
    model.check_recall(x_5bit_err, x, synchronous = True)
print("Asynchronous")
for x_5bit_err, x in zip(X_5bit_err, X):   
    model.check_recall(x_5bit_err, x, synchronous = False)

# # 3.2
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
# new_p10, iter = model.recall(p10,True, 100)
# if (np.array_equal(p1, new_p10)):
#     print("p1 and p10 same and converged after: ", str(iter), " iterations")
# model.display_pattern(p1)
# model.display_pattern(new_p10)
# new_p11, iter = model.recall(p11, True, 100)
# print("p11 cannot complete pattern because mix of two patterns but converges after: ", str(iter), " iterations")
# model.display_pattern(new_p11)

# new_p11, iter = model.recall(p11, False, 100, True)
# print("___Random Units___ \n Able to find combined pattern and converges after: ", str(iter), " iterations")
# model.display_pattern(new_p11)

# #display after each iteration
# print("___Random Units and display after iterations ___")
# new_p11, iter = model.recall(p11, False, 100, True, True)

#3.3 - Ralle my boy got my back on this one
# attractors = model.find_attractors(isPatterns = True)
# print("Found {} attractors for patterns (p1, p2, p3):".format(len(attractors)))
# for attractor in attractors:
#     E = model.energy(attractor)
#     print(attractor, " Energy: ", str(E))
# E_for_pattern = model.energy(new_p10)
# print(E_for_pattern)

# #3.4
def add_noise(pattern, noise_lvl):
    d = len(pattern)
    res=np.copy(pattern)
    random_idx = random.sample(list(np.arange(d)), int(noise_lvl * d / 100))
    for idx in random_idx:
        res[idx] = -pattern[idx]
    return res
# noice_amount_max = 1024
# able_to_recall = []

# # #3.4
# epochs = 1
# noice_amount_max = 1024
# able_to_recall = []

# attractor_to_test = p1
# for _ in tqdm(range(epochs)):
#     noice_amount = 0
#     while noice_amount <= noice_amount_max:
#         attractor_to_test_new = attractor_to_test.copy()
#         for i in range(noice_amount):
#             if attractor_to_test[i] == 1:
#                 attractor_to_test_new[i] = -1
#             if attractor_to_test[i] == -1:
#                 attractor_to_test_new[i] = 1
#         x_new = model.update(attractor_to_test_new, False)
#         # attractor_new = find_attractors(x_new)
#         # attractor_real = find_attractors(attractor_to_test)
#         # print("Real: " + str(attractor_real) + " and new: " + str(attractor_new))
#         if np.array_equal(x_new, attractor_to_test):
#             able_to_recall.append(1)
#         else:
#             sum = 0
#             for i in range(len(attractor_to_test)):
#                 if attractor_to_test[i] == x_new[i]:
#                     sum += 1
#                 else:
#                     sum += 0
#             able_to_recall.append(sum/len(attractor_to_test))
#         # x = x_new.reshape(32,32)
#         # plt.imshow(x)
#         # plt.title("Pattern")
#         # plt.show()
#         noice_amount += 1

# plt.figure()
# plt.plot(able_to_recall)
# plt.show()

# 3.5
all_patterns = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11]

patterns = np.array([p1])
errors =[]

for i in range(1,9):
    p_tmp = all_patterns[i]
    patterns = np.append(patterns, [p_tmp], axis=0)
    model.train(patterns, check=False)
    pred_p1,_ = model.recall(p1, True, 100)
    error = (p1 != pred_p1)
    error = sum(error)
    errors.append(error)
plt.plot(np.arange(2,10), errors)
plt.title("Errors in predicted pattern p1 when increasing amount of stored patterns")
plt.ylabel("Number of errors (0 - 1024")
plt.xlabel("Number of stored patterns")
plt.show()

# def generate_random_patterns():
#     rand_pattern = np.zeros((1024,))
#     flip_list = []

#     for _ in range(600):
#         tmp_int = random.randint(0,1023)
#         flip_list.append(tmp_int)
#     for j in range(600):
#         rand_pattern[flip_list[j]] = 1

#     return rand_pattern

# patterns = np.array([p1])
# all_patterns = []
# for i in range(0, 100):
#     all_patterns.append(generate_random_patterns())

# errors = []
# for i in range(1,98):
#     p_tmp = all_patterns[i]
#     patterns = np.append(patterns, [p_tmp], axis=0)
#     model.train(patterns, check=False)
#     pred_p1,_ = model.recall(p1, True, 100)
#     error = (p1 != pred_p1)
#     error = sum(error)
#     errors.append(error)
# plt.plot(np.arange(2,99), errors)
# plt.title("Errors in predicted pattern p1 when increasing amount of stored patterns (random patterns)")
# plt.ylabel("Number of errors (0 - 1024")
# plt.xlabel("Number of stored patterns")
# plt.show()

# 300 random
num_patterns = 300
size_net = 100
# all_patterns = np.array([np.random.choice([-1,1], size_net) for _ in range(num_patterns)])

# success = np.zeros(300, dtype=np.float64)
# weights = np.zeros((100,100))
# weights += np.outer(all_patterns[0], all_patterns[0])

def recall(w, x):
    return np.sign(w.dot(x))

# for i in range(1, 300):
#     weights += np.outer(all_patterns[i], all_patterns[i])
#     success_rate = 0
#     for j in range(0, i-1):
#         tmp_p = all_patterns[j]
#         pred = recall(weights, tmp_p)
#         if(np.array_equal(tmp_p, pred)):
#             success_rate += 1
#     success_rate = success_rate/i
#     success[i] = success_rate
# plt.plot(success)
# plt.xlabel("Number of patterns stored")
# plt.ylabel("Percentage of patterns that are stable")
# plt.show()

# noise 300 random
# all_patterns = np.array([np.random.choice([-1,1], size_net) for _ in range(num_patterns)])
# random_idx = [i for i in range(size_net)]
# np.random.shuffle(random_idx)
# noisy_patterns = np.array(all_patterns, copy=True)
# for i in range(num_patterns):
#     for j, idx in enumerate(random_idx):
#         noisy_patterns[i][idx] *= -1
#         if (j+1) % 40 == 0:
#             break

# success = np.zeros(300, dtype=np.float64)
# success_noisy = np.zeros(300)
# weights = np.zeros((100,100))
# weights += np.outer(all_patterns[0], all_patterns[0])

# def recall(w, x):
#     return np.sign(w.dot(x))

# for i in range(1, num_patterns):
#     weights += np.outer(all_patterns[i], all_patterns[i])
#     success_rate = 0
#     success_rate_noisy = 0
#     for j in range(0, i-1):
#         tmp_p_noisy = noisy_patterns[j]
#         tmp_p = all_patterns[j]
#         pred_noisy = recall(weights, tmp_p_noisy)
#         pred = recall(weights, tmp_p)
#         if(np.array_equal(tmp_p, pred_noisy)):
#             success_rate_noisy += 1
#         if(np.array_equal(tmp_p, pred)):
#             success_rate += 1
#     success_rate = success_rate/i
#     success_rate_noisy = success_rate_noisy/i
#     success[i] = success_rate
#     success_noisy[i] = success_rate_noisy

# line1 = plt.plot(success, color='r', label = 'no noise')
# line2 = plt.plot(success_noisy, color='b', label = 'noise')
# plt.title("Noisy vs no noise (with diag)")
# plt.xlabel("Number of patterns stored")
# plt.ylabel("Percentage of patterns that are stable")
# plt.legend((line1, line2), ('label1', 'label2'))
# plt.legend()
# plt.show()

# # noise 300 without diag
# all_patterns = np.array([np.random.choice([-1,1], size_net) for _ in range(num_patterns)])
# random_idx = [i for i in range(size_net)]
# np.random.shuffle(random_idx)
# noisy_patterns = np.array(all_patterns, copy=True)
# for i in range(num_patterns):
#     for j, idx in enumerate(random_idx):
#         noisy_patterns[i][idx] *= -1
#         if (j+1) % 10 == 0:
#             break

# success = np.zeros(300, dtype=np.float64)
# success_noisy = np.zeros(300)
# weights = np.zeros((100,100))
# weights += np.outer(all_patterns[0], all_patterns[0])

# def recall(w, x):
#     w = del_diag(w)
#     return np.sign(w.dot(x))

# def del_diag(w):
#     for i in range(w.shape[0]):
#         w[i][i] = 0
#     return w

# for i in range(1, num_patterns):
#     weights += np.outer(all_patterns[i], all_patterns[i])
#     success_rate = 0
#     success_rate_noisy = 0
#     for j in range(0, i-1):
#         tmp_p_noisy = noisy_patterns[j]
#         tmp_p = all_patterns[j]
#         pred_noisy = recall(weights, tmp_p_noisy)
#         pred = recall(weights, tmp_p)
#         if(np.array_equal(tmp_p, pred_noisy)):
#             success_rate_noisy += 1
#         if(np.array_equal(tmp_p, pred)):
#             success_rate += 1
#     success_rate = success_rate/i
#     success_rate_noisy = success_rate_noisy/i
#     success[i] = success_rate
#     success_noisy[i] = success_rate_noisy

# line1 = plt.plot(success, color='r', label = 'no noise')
# line2 = plt.plot(success_noisy, color='b', label = 'noise')
# plt.title("Noisy vs no noise (without diag)")
# plt.xlabel("Number of patterns stored")
# plt.ylabel("Percentage of patterns that are stable")
# plt.legend((line1, line2), ('label1', 'label2'))
# plt.legend()
# plt.show()

# noise 300 without diag + bias
all_patterns = np.array([np.random.choice([-1,1], size_net, p=[0.25,0.75]) for _ in range(num_patterns)])
random_idx = [i for i in range(size_net)]
np.random.shuffle(random_idx)
noisy_patterns = np.array(all_patterns, copy=True)
for i in range(num_patterns):
    for j, idx in enumerate(random_idx):
        noisy_patterns[i][idx] *= -1
        if (j+1) % 10 == 0:
            break

success = np.zeros(300, dtype=np.float64)
success_noisy = np.zeros(300)
weights = np.zeros((100,100))
weights += np.outer(all_patterns[0], all_patterns[0])

def recall(w, x):
    w = del_diag(w)
    return np.sign(w.dot(x))

def del_diag(w):
    for i in range(w.shape[0]):
        w[i][i] = 0
    return w

for i in range(1, num_patterns):
    weights += np.outer(all_patterns[i], all_patterns[i])
    success_rate = 0
    success_rate_noisy = 0
    for j in range(0, i-1):
        tmp_p_noisy = noisy_patterns[j]
        tmp_p = all_patterns[j]
        pred_noisy = recall(weights, tmp_p_noisy)
        pred = recall(weights, tmp_p)
        if(np.array_equal(tmp_p, pred_noisy)):
            success_rate_noisy += 1
        if(np.array_equal(tmp_p, pred)):
            success_rate += 1
    success_rate = success_rate/i
    success_rate_noisy = success_rate_noisy/i
    success[i] = success_rate
    success_noisy[i] = success_rate_noisy

line1 = plt.plot(success, color='r', label = 'no noise')
line2 = plt.plot(success_noisy, color='b', label = 'noise')
plt.title("Noisy vs no noise (without diag + bias)")
plt.xlabel("Number of patterns stored")
plt.ylabel("Percentage of patterns that are stable")
plt.legend((line1, line2), ('label1', 'label2'))
plt.legend()
plt.show()