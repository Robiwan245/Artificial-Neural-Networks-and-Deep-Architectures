import itertools
import numpy as np
import matplotlib.pyplot as plt

class HOP:
    def __init__(self) -> None:
        self.theta = None

    def train(self, X):
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

    def recall(self, x, synchronous=False, max_iter = 1):
        iter = 0
        while iter<max_iter:
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
        
        return np.array([attractor for attractor in attractors])

X = np.array([[-1,-1,1,-1,1,-1,-1,1],
                [-1,-1,-1,-1,-1,1,-1,-1],
                [-1,1,1,-1,-1,1,-1,1]])
model = HOP()
model.train(X)

X_distorted = np.array([[1, -1, 1, -1, 1, -1, -1, 1],
                        [1, 1, -1, -1, -1, 1, -1, -1],
                        [1, 1, 1, -1, 1, 1, -1, 1]])

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
