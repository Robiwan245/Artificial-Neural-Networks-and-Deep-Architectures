import numpy as np
import matplotlib.pyplot as plt

def generate_linearly_separated_data():

    n = 100
    mA = [ 3, 2]
    mB = [-3, -2] 
    sigmaA = 1
    sigmaB = -1

    classA = np.zeros((2, n))
    classB = np.zeros((2, n))
    classA_labels = np.zeros(2*n)
    classB_labels = np.ones(2*n)

    classA[0,:] = np.random.normal(size=n) * sigmaA + mA[0]
    classA[1,:] = np.random.normal(size=n) * sigmaA + mA[1]
    classB[0,:] = np.random.normal(size=n) * sigmaB + mB[0]
    classB[1,:] = np.random.normal(size=n) * sigmaB + mB[1]

    testA = np.zeros((2,n))
    testB = np.zeros((2,n))
    testA_labels = np.zeros(n)
    testB_labels = np.ones(n)
    testA[0,:] = np.random.normal(size=n) * 1 + 5
    testB[0,:] = np.random.normal(size=n) * -1 -2
    testA[1,:] = np.random.normal(size=n) * 1 + 5
    testB[1,:] = np.random.normal(size=n) * -1 -2
    X_test = np.concatenate((testA, testB))
    Y_test = np.concatenate((testA_labels, testB_labels))

    X = np.concatenate((classA, classB))
    Y = np.concatenate((classA_labels, classB_labels))

    return X, Y, X_test, Y_test