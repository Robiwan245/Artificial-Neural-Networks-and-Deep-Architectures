from cProfile import label
import numpy as np
import matplotlib.pyplot as plt

def generate_linearly_separated_data(perception, seq, batch):

    n = 100
    mA = [ 3, 2]
    mB = [-3, -2] 
    sigmaA = 1
    sigmaB = -1

    classA = np.zeros((2, n))
    classB = np.zeros((2, n))
    if perception:
        classA_labels = np.ones(n)
        classB_labels = -np.ones(n)
    elif seq or batch:
        classA_labels = np.ones(n)
        classB_labels = -np.ones(n)

    classA[0,:] = np.random.normal(size=n) * sigmaA + mA[0]
    classA[1,:] = np.random.normal(size=n) * sigmaA + mA[1]
    classB[0,:] = np.random.normal(size=n) * sigmaB + mB[0]
    classB[1,:] = np.random.normal(size=n) * sigmaB + mB[1]

    # classA[0,:] = np.random.normal(size=n) * sigmaA + 4.8
    # classA[1,:] = np.random.normal(size=n) * sigmaA + 4.6
    # classB[0,:] = np.random.normal(size=n) * sigmaB + (0.6)
    # classB[1,:] = np.random.normal(size=n) * sigmaB + (0.8)
    
    ones = np.ones((2, n))

    if perception or seq:
        ones_trans = np.transpose(ones)
        classA_trans = np.transpose(classA)
        classB_trans = np.transpose(classB)

        X = np.concatenate((classA_trans, classB_trans, ones_trans))
    elif batch:
        classA_labels = np.ones(int(n/2))
        classB_labels = -np.ones(int(n/2))

        X = np.concatenate((classA, classB, ones))
    Y = np.concatenate((classA_labels, classB_labels))

    return X, Y, classA, classB

def generate__non_linearly_separated_data(perception, seq, batch):

    n = 100
    mA = [ 2, 2]
    mB = [-2, -2] 
    sigmaA = 2
    sigmaB = -2

    classA = np.zeros((2, n))
    classB = np.zeros((2, n))
    if perception:
        classA_labels = np.zeros(n)
        classB_labels = np.ones(n)
    elif seq or batch:
        classA_labels = np.ones(int(n/2))
        classB_labels = -np.ones(int(n/2))

    classA[0,:] = np.random.normal(size=n) * sigmaA + mA[0]
    classA[1,:] = np.random.normal(size=n) * sigmaA + mA[1]
    classB[0,:] = np.random.normal(size=n) * sigmaB + mB[0]
    classB[1,:] = np.random.normal(size=n) * sigmaB + mB[1]
    
    ones = np.ones((2, n))

    if perception or seq:
        ones_trans = np.transpose(ones)
        classA_trans = np.transpose(classA)
        classB_trans = np.transpose(classB)

        X = np.concatenate((classA_trans, classB_trans, ones_trans))
    elif batch:
        X = np.concatenate((classA, classB, ones))
    Y = np.concatenate((classA_labels, classB_labels))

    return X, Y, classA, classB

def generate__non_linearly_separated_data2(perception, seq, batch, remove_25, remove_50_A, remove_50_B, a_20_80):

    n = 100
    mA = [ 1.0, 0.3]
    mB = [ 0.0, -0.1] 
    sigmaA = 0.2
    sigmaB = 0.3

    classA = np.zeros(n)
    classB = np.zeros(n)
    classA2 = np.zeros(n)
    classB2 = np.zeros(n)
    classA_labels = np.ones(int(n/2))
    classB_labels = -np.ones(int(n/2))

    for i in range(round(0.5*n)):
        classA[i] = np.random.normal() * sigmaA - mA[0]
    for i in range(round(0.5*n)):
        classA[i + round(0.5*n)] = np.random.normal() * sigmaA + mA[0]
    for i in range(n):
        classA2[i] = np.random.normal() * sigmaA + mA[1]
        classB[i] = np.random.normal() * sigmaB + mB[0]
        classB2[i] = np.random.normal() * sigmaB + mB[1]
    
    if remove_25:
        
        epsilon = np.random.choice(list(range(0, n)), size=round(n*0.25), replace=False)
        classA = np.delete(classA, epsilon)
        classA2 = np.delete(classA2, epsilon)
        epsilon2 = np.random.choice(list(range(0, n)), size=round(n*0.25), replace=False)
        classB = np.delete(classB, epsilon2)
        classB2 = np.delete(classB2, epsilon2)

        classA = np.vstack((classA, classA2))
        classB = np.vstack((classB, classB2))

        ones = np.ones((1, n - round(n*0.25)))

        X = np.concatenate((classA, classB, ones))
        Y = np.concatenate((classA_labels, classB_labels))
        Y = np.delete(Y, list(range(0,round(n*0.125))))
        Y = np.delete(Y, list(range(int(n/2), int(n/2) + round(n*0.125) + 1)))

        return X, Y, classA, classB
    
    if remove_50_A:

        Y_A = np.zeros(round(0.5*n))
        Y_B = np.zeros(n)

        classA = [0 for i in range(round(0.5*n))]
        classA2 = [0 for i in range(round(0.5*n))]
        classB = [0 for i in range(n)]
        classB2 = [0 for i in range(n)]

        for i in range(round(0.25*n)):
            classA[i] = np.random.normal() * sigmaA - mA[0]
        for i in range(round(0.25*n)):
            classA[i + round(0.25*n)] = np.random.normal() * sigmaA + mA[0]
        for i in range(round(0.5*n)):
            classA2[i] = np.random.normal() * sigmaA + mA[1]
        for i in range(n):
            classB[i] = np.random.normal() * sigmaB + mB[0]
            classB2[i] = np.random.normal() * sigmaB + mB[1]

        ones = np.ones((2, n))

        classAconc = np.zeros((2, round(0.5*n)))
        classBconc = np.zeros((2, n))

        classAconc[0] = classA
        classAconc[1] = classA2
        classBconc[0] = classB
        classBconc[1] = classB2

        X_A_new = np.vstack((classA, classA2))
        X_B_new = np.vstack((classB, classB2))

        X = np.hstack((X_A_new, X_B_new))

        for i in range(round(0.5*n)):
            Y_A[i] = 1
        for i in range(round(n)):
            Y_B[i] = -1

        Y = np.hstack((Y_A, Y_B))

        return X, Y, classAconc, classBconc

    if remove_50_B:

        Y_A = np.zeros(n)
        Y_B = np.zeros(round(0.5*n))

        classA = [0 for i in range(n)]
        classA2 = [0 for i in range(n)]
        classB = [0 for i in range(round(0.5*n))]
        classB2 = [0 for i in range(round(0.5*n))]

        for i in range(round(0.5*n)):
            classA[i] = np.random.normal() * sigmaA - mA[0]
        for i in range(round(0.5*n)):
            classA[i + round(0.5*n)] = np.random.normal() * sigmaA + mA[0]
        for i in range(n):
            classA2[i] = np.random.normal() * sigmaA + mA[1]
        for i in range(round(0.5*n)):
            classB[i] = np.random.normal() * sigmaB + mB[0]
            classB2[i] = np.random.normal() * sigmaB + mB[1]

        ones = np.ones((2, n))

        classAconc = np.zeros((2, n))
        classBconc = np.zeros((2, round(0.5*n)))

        classAconc[0] = classA
        classAconc[1] = classA2
        classBconc[0] = classB
        classBconc[1] = classB2

        X_A_new = np.vstack((classA, classA2))
        X_B_new = np.vstack((classB, classB2))

        X = np.hstack((X_A_new, X_B_new))

        for i in range(n):
            Y_A[i] = 1
        for i in range(round(0.5*n)):
            Y_B[i] = -1

        Y = np.hstack((Y_A, Y_B))

        return X, Y, classAconc, classBconc

    if a_20_80:

        classA_curr = np.zeros(n)

        index = 0

        while index < round(n*0.2):
            point = np.random.normal() * sigmaA - mA[0]
            if point < 0:
                classA_curr[index] = point
                index += 1

        while index < n:
            point = np.random.normal() * sigmaA + mA[0]
            if point > 0:
                classA_curr[index] = point
                index += 1

        classA = np.vstack((classA_curr, classA2))
        classB = np.vstack((classB, classB2))

        ones = np.ones((1, n))

        X = np.concatenate((classA, classB, ones))
        Y = np.concatenate((classA_labels, classB_labels))

        return X, Y, classA, classB

    else:
        classA = np.vstack((classA, classA2))
        classB = np.vstack((classB, classB2))

        ones = np.ones((1, n))

        X = np.concatenate((classA, classB, ones))
        Y = np.concatenate((classA_labels, classB_labels))

    return X, Y, classA, classB

