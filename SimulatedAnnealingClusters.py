import numpy as np
import random
import time
from scipy.spatial import distance
from psutil import cpu_freq
import copy


# SA-search for cluster improvment
class SimulatedAnnealing:
    def __init__(self, cvrp):
        self.cvrp = cvrp
        self.saBest = cvrp.city_clusters.copy()
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

    # random neighbor defined by randomly moving a city to different cluster
    def getRandomNeighborhood(self, candidate):
        cand_copy = copy.deepcopy(candidate)
        idx = range(len(cand_copy))
        i1, i2 = random.sample(idx, 2)
        city = random.choice(cand_copy[i1])
        cand_copy[i2].append(city)
        cand_copy[i1].remove(city)
        cand_copy = [x for x in cand_copy if x]
        if self.cvrp.legal(cand_copy):
            return cand_copy
        return candidate

    # cost defined by sum of distances of cities in cluster to center point of cluster
    def cost(self, clusters):
        value = 0
        for cluster in clusters:
            x, y = 0, 0
            for city in cluster:
                x += self.cvrp.city_cords[city-1][0]
                y += self.cvrp.city_cords[city-1][1]
            if cluster:
                sack_center = np.asarray([x / len(cluster), y / len(cluster)])
                for city in cluster:
                    value += distance.euclidean(sack_center, self.cvrp.city_cords[city-1])
        return value*len(clusters)
