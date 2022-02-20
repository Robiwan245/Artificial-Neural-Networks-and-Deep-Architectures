import numpy as np
import itertools
from matplotlib import pyplot as plt
from tqdm import tqdm

def calculate_weight_matrix(X):
    N = X.shape[0]
    w = np.zeros((N, N))
    w = np.sum([np.dot(x.reshape(-1,1), x.reshape(1,-1)) for x in X], axis = 0)
    return w

def update_rule(x, w, asyn):
    if asyn:
        x_new = x.copy()
        for i in range(len(x)):
            x_new[i] = np.sign(np.sum(w[i, :] * x))
    else:
        x_new = np.dot(w, x)
        x_new[x_new>=0] = 1
        x_new[x_new < 0] = -1
    return x_new

def find_attractors(data):
    attractors_list = []
    for att in tqdm(data):
        attractors = update_rule(att, weight_matrix, asyn)
        attractors_list.append(attractors)
    attractors = np.unique(np.array(attractors_list), axis = 0)
    return attractors

def energy(w, x):
    E = 0
    for i in range(len(x)):
        for j in range(len(x)):
            E += - np.dot(w[i, j], np.dot(x[i], x[j]))
    return E

distorted = True
super_distorted = False
asyn = False

x1 = [-1, -1, 1, -1, 1, -1, -1, 1]
x2 = [-1, -1, -1, -1, -1, 1, -1, -1]
x3 = [-1, 1, 1, -1, -1, 1, -1, 1]

x1d = [ 1, -1, 1, -1, 1, -1, -1, 1]
x2d = [ 1, 1, -1, -1, -1, 1, -1, -1]
x3d = [ 1, 1, 1, -1, 1, 1, -1, 1,]

x1d2 = [ 1, 1, -1, 1, -1, 1, -1, 1]
x2d2 = [ 1, 1, 1, 1, 1, -1, -1, -1,]
x3d2 = [ 1, -1, -1, 1, 1, -1, -1, 1,]

data = np.array(list(itertools.product([-1, 1], repeat=8)))

if distorted:
    X = np.vstack((x1d, x2d, x3d))
if super_distorted:
    X = np.vstack((x1d2, x2d2, x3d2))
x_true = np.vstack((x1, x2, x3))

weight_matrix = calculate_weight_matrix(x_true)

new_x = np.zeros((len(X), len(X[0])))
prev_update = np.zeros((len(X), len(X[0])))
done = False

for epoch in range(1000):
    for i in range(len(X)):
        update = update_rule(X[i], weight_matrix, asyn)
        new_x[i] = update
    # if np.array_equal(prev_update, new_x):
    #     print("Done after " + str(epoch) + " iterations")
    #     break 
    # prev_update = new_x.copy()
succes = False

for i in range(len(new_x)):
    if np.array_equal(x_true[i], new_x[i]):
        print("Cool!")
        print("True: " + str(x_true[i]))
        print("Recalled: " + str(X[i]) + " -> " + str(new_x[i]))
    else:
        print(":(")
        print("True: " + str(x_true[i]))
        print("Recalled: " + str(X[i]) + " -> " + str(new_x[i]))

#weight_matrix = calculate_weight_matrix(data)
attractors = find_attractors(data)
print("___________________")
print("There are " + str(len(attractors)) + " attractors.")
print("The attractors are:")
print(attractors)

E = energy(weight_matrix, attractors[0])
print(E)

pict = np.genfromtxt("pict.dat", delimiter = ',')

pict = pict.reshape(11, 1024)

p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11 = \
    pict[0], pict[1], pict[2], pict[3], pict[4], pict[5], pict[6], \
        pict[7], pict[8], pict[9], pict[10]

# for i in range(pict.shape[0]):
#     x = pict[i].reshape(32,32)
#     plt.imshow(x)
#     plt.title("Pattern")
#     plt.show()

epochs = 10
x_prev = None

x = pict[0].reshape(32,32)
plt.imshow(x)
plt.title("Pattern")
plt.show()

for i in range(10):
    weight_matrix = calculate_weight_matrix(np.array([p1, p2, p3, p4]))

#weight_matrix = np.array([[np.random.normal(0, 1) for _ in range(1024)] for _ in range(1024)])

#weight_matrix = np.array([[0.5 for _ in range(1024)] for _ in range(1024)])

for i in range(epochs):
    # e_distorted = energy(weight_matrix, p10)
    # print("Energy:" + str(e_distorted))
    x_new = update_rule(p10, weight_matrix, asyn)
    # e_distorted = energy(weight_matrix, x_new)
    # print("Energy:" + str(e_distorted))
    # if np.array_equal(x_prev, x_new):
    #     print("Done after " + str(i) + " iterations.")
    #     break
    # x_prev = x_new.copy()

x = x_new.reshape(32,32)
plt.imshow(x)
plt.title("Pattern")
plt.show()

noice_amount_max = 50
able_to_recall = []

#3.4
attractor_to_test = p4
for _ in range(epochs):
    noice_amount = 0
    while noice_amount <= noice_amount_max:
        attractor_to_test_new = attractor_to_test.copy()
        for i in range(noice_amount):
            if attractor_to_test[i] == 1:
                attractor_to_test_new[i] = -1
            if attractor_to_test[i] == -1:
                attractor_to_test_new[i] = 1
        x_new = update_rule(attractor_to_test_new, weight_matrix, asyn)
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

x_new = update_rule(p4, weight_matrix, asyn)
x = x_new.reshape(32,32)
plt.imshow(x)
plt.title("Pattern")
plt.show()

