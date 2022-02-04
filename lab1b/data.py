import numpy as np
import matplotlib as plt

def generate_not_linearly_separable_data():

    n = 100
    mA = [ 2.0, 0.3]
    mB = [ -0.1, -0.1] 
    sigmaA = 0.2
    sigmaB = 0.3

    classA = np.zeros(n)
    classB = np.zeros(n)
    classA2 = np.zeros(n)
    classB2 = np.zeros(n)
    classA_labels = np.ones(n)
    classB_labels = -np.ones(n)

    for i in range(round(0.5*n)):
        classA[i] = np.random.normal() * sigmaA - mA[0]
    for i in range(round(0.5*n)):
        classA[i + round(0.5*n)] = np.random.normal() * sigmaA + mA[0]
    for i in range(n):
        classA2[i] = np.random.normal() * sigmaA + mA[1]
        classB[i] = np.random.normal() * sigmaB + mB[0]
        classB2[i] = np.random.normal() * sigmaB + mB[1]

    Xs = np.hstack((classA, classB))
    Ys = np.hstack((classA2, classB2))

    # classA = np.vstack((classA, classA2))
    # classB = np.vstack((classB, classB2))

    #ones = np.ones((1, 2*n))

    # X = np.concatenate((classA, classB, ones))
    X = np.vstack((Xs, Ys))
    Y = np.concatenate((classA_labels, classB_labels))

    return X, Y, classA, classB, n

    #return classA, classB, classA_labels, classB_labels

def xor():
    X = np.array([[-1,1,-1,1], [-1, -1,1,1]])
    T = np.array([-1,1,1,-1])

    return X, T