import numpy as np
import random
import time
from psutil import cpu_freq


class SimulatedAnnealing:
    def __init__(self, cvrp, cluster):
        self.cities = cluster.copy()
        self.dist_matrix = cvrp.dist_matrix
        self.saBest = self.generate_start_permutation_3NN()
        self.saBest_cost = self.cost(self.saBest)

    def sa_search(self, search_time=120, output=False):
        candidate = self.saBest
        candidate_cost = self.cost(candidate)
        end_time = time.time() + search_time
        time_left = end_time - time.time()
        i = 0
        while time_left > 0:
            temp = 90 * time_left / search_time
            random_neighbor = self.getRandomNeighborhood(candidate)
            neighbor_cost = self.cost(random_neighbor)

            if temp < 0.01:
                chance = 0
            else:
                chance = np.exp((candidate_cost - neighbor_cost) / temp)
            if neighbor_cost < self.saBest_cost:
                self.saBest = random_neighbor.copy()
                self.saBest_cost = neighbor_cost
                if output:
                    print("Improvement Found!")
                    print("Best:")
                    print(self.saBest)
                    total_time = search_time-time_left
                    print("Elapsed Time: ", total_time)
                    print("Total clock ticks: ", total_time * cpu_freq()[0] * 2 ** 20)
                    print("Total iteration: ", i, "\n")

            if neighbor_cost < candidate_cost or random.random() < chance:
                candidate = random_neighbor.copy()
                candidate_cost = neighbor_cost
            i += 1
            time_left = end_time - time.time()

    def getRandomNeighborhood(self, candidate):
        neighbor = candidate.copy()
        idx = range(len(candidate))
        i1, i2 = random.sample(idx, 2)
        neighbor[i1], neighbor[i2] = neighbor[i2], neighbor[i1]
        return neighbor

    def generate_start_permutation_3NN(self):
        permutation = []
        unvisited_cities = self.cities.copy()
        current_city = 1
        while unvisited_cities:
            min1 = min2 = min3 = np.inf
            min1_city, min2_city, min3_city = 0, 0, 0
            for city in unvisited_cities:
                dist = self.dist_matrix[current_city-1][city-1]
                if city != current_city and dist < min3 and city in unvisited_cities and city != 0:
                    min3 = dist
                    min3_city = city
                    if dist < min2:
                        min3, min2 = min2, min3
                        min3_city, min2_city = min2_city, min3_city
                        if min2 < min1:
                            min1, min2 = min2, min1
                            min1_city, min2_city = min2_city, min1_city
            choices = [min1_city, min2_city, min3_city]
            choices = list(filter(lambda x: x != 0, choices))
            city = random.choice(choices)
            permutation.append(city)
            unvisited_cities.remove(city)
        return np.array(permutation)

    def cost(self, config):
        current_city = 1
        cost = 0
        config_copy = config.copy()
        config_copy = config_copy.tolist()
        while config_copy:
            cost += self.dist_matrix[current_city-1][config_copy[0]-1]
            current_city = config_copy.pop(0)
        return cost + self.dist_matrix[current_city-1][0]

    def __str__(self):
        string = "Simulated Annealing:\nThe Best Route Found: \n"
        return string + str(self.saBest)
