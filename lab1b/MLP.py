import data
import matplotlib.pyplot as plt
import numpy as np

class MLP:
    def __init__(self) -> None:
        pass

    def test_plot(self):
        classA, classB, labelsA, labelsB = data.generate_not_linearly_separable_data()

        plt.figure(figsize=(10, 10))
        plt.xlabel("x-coordinates of the data")
        plt.ylabel("y-coordinates of the data")
        plt.scatter(classA[0,:], classA[1,:], c = "red")
        plt.scatter(classB[0,:], classB[1,:], c = "green")
        plt.plot(labelsA, labelsB, "k")

        plt.show()

MLP_model = MLP()

MLP_model.test_plot()