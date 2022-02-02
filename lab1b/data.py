import numpy as np

def generate_not_linearly_separable_data():
    n = 100
    mA = [ 1.0, 0.3]
    sigmaA = 0.2
    mB = [ 0.0, -0.1]
    sigmaB = 0.3

    classA = np.zeros((2, n))
    classB = np.zeros((2, n))
    classA_labels = np.negative(np.ones(n))
    classB_labels = np.ones(n)

    classA[0,:] = np.random.normal(size=n) * sigmaA + mA[0]
    classA[1,:] = np.random.normal(size=n) * sigmaA + mA[1]
    classB[0,:] = np.random.normal(size=n) * sigmaB + mB[0]
    classB[1,:] = np.random.normal(size=n) * sigmaB + mB[1]

    return classA, classB, classA_labels, classB_labels