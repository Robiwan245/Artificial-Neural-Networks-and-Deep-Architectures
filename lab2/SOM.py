import numpy as np
import matplotlib.pyplot as plt


class SOM():

    def __init__(self) -> None:
        pass

    def compute_SOM(self, inputs, outputs, alpha=0.2, epochs=20, n_init_size=50, neighbourhood='default'):
        n_size = n_init_size
        n_step = n_size/epochs

        if neighbourhood == 'grid':
            theta = np.random.uniform(0, 1, size=(np.prod(outputs), inputs.shape[1]))
            output_idx = np.arange(np.prod(outputs)).reshape(outputs)
        else:
            theta = np.random.uniform(0,1, size=(outputs, inputs.shape[1]))
            output_idx = list(range(outputs))

        for _ in range(epochs):
            for row in inputs:
                
                # compute winner
                winner = np.argmin(np.linalg.norm(row-theta, axis=1))

                if neighbourhood == 'circular':
                    n_l_bound = winner-int(n_size)
                    n_u_bound = winner+int(n_size)+1
                    if n_l_bound<0:
                        idx = output_idx[n_l_bound]+output_idx[n_u_bound]
                    elif n_u_bound>outputs:
                        idx = output_idx[n_l_bound]+output_idx[(n_u_bound-outputs)]
                    else:
                        idx = output_idx[n_l_bound:n_u_bound]
                if neighbourhood == 'grid':
                    winner_x, winner_y = np.unravel_index(winner, outputs)
                    idx = []
                    for i in range(output_idx.shape[0]):
                        for j in range(output_idx.shape[1]):
                            if (abs(winner_x-i)+abs(winner_y-j))<= int(n_size):
                                idx.append(output_idx[i, j])
                else:
                    n_l_bound = max(0, winner-int(n_size))
                    n_u_bound = min(outputs, winner+int(n_size)+1)
                    idx = range(n_l_bound, n_u_bound)

                delta = alpha*(row-theta[idx, :])
                theta[idx, :] += delta

            n_size -= n_step
        
        predicted_nodes = [np.argmin(np.linalg.norm(theta-row, axis=1)) for row in inputs]
        sorted_indexes = np.argsort(predicted_nodes)
        return np.array(predicted_nodes), sorted_indexes, theta
    
    def plot_city_tour(self, cities, pred_nodes, sorted_indexes, theta):
        def plot_arrow(start, end, color):
            start_x, start_y = start
            end_x, end_y = end
            plt.arrow(start_x,start_y,end_x-start_x, end_y-start_y, color= color,head_width=0.01, length_includes_head=True)
        
        plt.figure(figsize=(12,12))
        plt.scatter(theta[:, 0], theta[:, 1], color='r', label="Predicted tour")
        plt.scatter(cities[:,0], cities[:,1], color='b', label="Real tour")
        for i in range(-1,len(sorted_indexes)-1):
            plot_arrow(theta[pred_nodes[i]], theta[pred_nodes[i+1]], 'r')
            plot_arrow(cities[i], cities[i+1], 'b')
        plt.legend()
        plt.show()
    
    # Add noise to point for 4.3
    def add_noise(self, point, noise=0.4):
        return point.astype(float)+np.random.uniform(-noise, noise, size=point.shape)



som = SOM()
# 4.1 - Topological Ordering of Animal Species
animals_in  = np.genfromtxt('data/animals.dat', delimiter=',').reshape(32,84)
file = open('data/animalnames.txt', 'r')
animal_names = [line.strip("\n\t'") for line in file]
file.close()
pred_nodes, sorted_idx, _ = som.compute_SOM(animals_in, outputs=100)
[print(animal_names[i], "| node: ", pred_nodes[i]) for i in sorted_idx]

# 4.2 - Cyclic Tour
file = open('data/cities.dat', 'r')
input_array = [line.replace(';', '') for line in file]
file.close()
cities = np.genfromtxt(input_array, comments='%', delimiter=',')
plt.scatter(cities[:,0], cities[:,1])
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.show()

pred_nodes, sorted_idx, theta = som.compute_SOM(cities, outputs=10, neighbourhood='cicular', n_init_size=2)
som.plot_city_tour(cities, pred_nodes, sorted_idx, theta)

# 4.3 - Data Clustering: Votes of MPs
votes = np.genfromtxt('data/votes.dat', delimiter=',').reshape(349, 31)
parties = np.genfromtxt('data/mpparty.dat', comments='%', dtype=np.uint8)
districts = np.genfromtxt('data/mpdistrict.dat', comments='%', dtype=np.uint8)
genders = np.genfromtxt('data/mpsex.dat', comments='%', dtype=np.uint8)
mp_attrs = np.column_stack((parties,districts,genders))
mp_attr_titles = ["Party","District","Gender"]
mp_attr_labels = [
    ["Inget parti","Moderaterna","Folkpartiet","Socialdemokraterna","Vänsterpartiet",
     "Miljöpartiet","Kristdemokraterna","Centerpartiet"],
    ["District "+str(d) for d in np.unique(districts)],
    ["Male","Female"]
]

gridsize = (10,10)
pred_nodes, sorted_idx, theta = som.compute_SOM(votes, outputs=gridsize, neighbourhood='grid', n_init_size=4)

for attr in range(len(mp_attr_titles)):
    fig = plt.figure(figsize=(5,5))

    for value in np.unique(mp_attrs[:, attr]):
        x, y = np.unravel_index(pred_nodes[mp_attrs[:, attr] == value], gridsize)
        plt.scatter(som.add_noise(x), som.add_noise(y), label=mp_attr_labels[attr][value-1])

    plt.title(mp_attr_titles[attr], fontsize=30, y=1.02)
    plt.xticks(np.arange(-0.5, gridsize[0]-0.5, 1), labels=[])
    plt.yticks(np.arange(-0.5, gridsize[1]-0.5, 1), labels=[])
    plt.grid(True)
    plt.xlim([-0.5, gridsize[0]-0.5])
    plt.ylim([-0.5, gridsize[1]-0.5])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()