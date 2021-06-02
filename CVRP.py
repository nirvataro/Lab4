from Knapsack import Knapsack, Item
import LDS_search
import numpy as np
from scipy.spatial import distance
from copy import deepcopy
from SimulatedAnnealingCVRP import SimulatedAnnealing as SA_cvrp
from SimulatedAnnealingClusters import SimulatedAnnealing as SA_clusters
import time
import random
import networkx as nx
import matplotlib.pyplot as plt


class CVRP:
    def __init__(self, file=None, capacity=None, dist_matrix=None, goods=None, cords=None):
        if file:
            self.capacity, self.dist_matrix, self.goods, self.city_cords = self.extract(file)
        else:
            self.capacity = capacity
            self.dist_matrix = dist_matrix
            self.goods = goods
            self.city_cords = cords
        self.trucks = []
        self.unvisited_cities = list(range(1, len(self.goods)))
        self.cost = 0

    def extract(self, input_file):
        with open(input_file) as f:
            lines = f.readlines()
        for l in lines:
            l.strip()
        cords, goods = [], []
        for idx, l in enumerate(lines):
            # extract city number
            if l.startswith('DIMENSION :'):
                l = l.replace('DIMENSION : ', '')
                dim = int(l)
            # extracts trucks capacity
            if l.startswith("CAPACITY : "):
                l = l.replace('CAPACITY : ', '')
                capacity = int(l)
            # extract city coordinates
            if l.startswith('NODE_COORD_SECTION'):
                for i in range(idx + 1, dim + idx + 1):
                    p = lines[i].split()
                    cords.append((int(p[1]), int(p[2])))
            if l.startswith('DEMAND_SECTION'):
                for i in range(idx + 1, dim + idx + 1):
                    g = lines[i].split()
                    goods.append(int(g[1]))
        # preprocessing - calculate distance matrix
        dist_matrix = np.array([[0 for _ in range(dim)] for _ in range(dim)], float)
        for index_i, cord_i in enumerate(cords):
            for index_j, cord_j in enumerate(cords):
                dist_matrix[index_i][index_j] = distance.euclidean(cord_i, cord_j)
        return capacity, dist_matrix, goods, cords

    def by_config(self, config):
        for clus in config:
            self.add_truck()
            self.add_route_to_truck(self.trucks[-1], clus)

    def add_truck(self):
        self.trucks.append(Truck(self.capacity))

    def add_route_to_truck(self, truck, route):
        truck.road = route
        current_city = 1
        truck.cost = 0
        route_copy = route.copy()
        while route_copy:
            truck.cost += self.dist_matrix[current_city-1][route_copy[0]-1]
            truck.room -= self.goods[current_city-1]
            current_city = route_copy.pop(0)
        truck.cost += self.dist_matrix[current_city-1][0]
        self.cost += truck.cost

    def legal(self, clusters):
        for cluster in clusters:
            room = self.capacity
            for c in cluster:
                room -= self.goods[c-1]
            if room < 0:
                return False
        return True

    def draw(self):
        node_color = "#" + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)])
        plt.plot(self.city_cords[0][0], self.city_cords[0][1], node_color,linewidth=5)
        plt.text(self.city_cords[0][0], self.city_cords[0][1], "1")
        for truck in self.trucks:
            node_color = "#" + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)])
            last_city = (self.city_cords[0][0], self.city_cords[0][1])
            for city in truck.road:
                plt.plot(self.city_cords[city-1][0], self.city_cords[city-1][1], node_color,linewidth=5)
                plt.text(self.city_cords[city-1][0], self.city_cords[city-1][1], str(city))
                plt.plot([self.city_cords[city-1][0], last_city[0]], [self.city_cords[city-1][1], last_city[1]], node_color)
                last_city = (self.city_cords[city-1][0], self.city_cords[city-1][1])
            plt.plot([self.city_cords[0][0], last_city[0]], [self.city_cords[0][1], last_city[1]],
                     node_color)

        plt.show()

    def generate_clusters(self):
        clusters = []
        unused_cities = self.unvisited_cities.copy()
        while unused_cities:
            cluster = []
            cluster_room = self.capacity
            while cluster_room and unused_cities:
                city = random.choice(unused_cities)
                if self.goods[city-1] > cluster_room:
                    break
                unused_cities.remove(city)
                cluster.append(city)
                cluster_room -= self.goods[city-1]
            clusters.append(cluster)
        return clusters


class Truck:
    def __init__(self, capacity):
        self.room = capacity
        self.road = []
        self.cost = 0


class TwoStepSolution(CVRP):
    def __init__(self, file=None):
        super().__init__(file)
        self.city_clusters = []
        self.sack_center = None
        self.ks = Knapsack(adjustable_values=self.adjust_values)

    def search(self, time=120):
        print("Clustering")
        self.clustering(time/2)
        print("Finding Paths")
        self.TSP(time/2)

    def clustering(self, time):
        max_trucks = 1 + sum(self.goods)/self.capacity
        time_for_sack = 0.5 * time / max_trucks
        while self.unvisited_cities:
            items = []
            for index, city_number in enumerate(self.unvisited_cities):
                items.append(Item(index, 0.0001, [self.goods[city_number]]))
            self.ks.init_not_from_file(1, [self.capacity], items)
            self.ks.clear_sacks()
            lds = LDS_search.LDS(self.ks)
            lds.search(0, time_for_sack)
            cluster = [self.unvisited_cities[item] for item in self.ks.items_used]
            self.city_clusters.append([self.unvisited_cities[item]+1 for item in self.ks.items_used])
            self.unvisited_cities = [item for item in self.unvisited_cities if item not in cluster]
        self.improve_clustering(0.5 * time)

    def improve_clustering(self, timer):
        sa_clusters = SA_clusters(self)
        sa_clusters.sa_search(timer, True)
        self.city_clusters = sa_clusters.saBest.copy()

    def TSP(self, time):
        large_clusters = sum([1 for cluster in self.city_clusters if len(cluster) > 1])
        for cluster in self.city_clusters:
            self.trucks.append(Truck(self.capacity))
            road = np.array(self.find_TSP(cluster, time/large_clusters) if len(cluster) > 1 else cluster)
            self.add_route_to_truck(self.trucks[-1], road.tolist())

    def find_TSP(self, cities, timer):
        sa = SA_cvrp(self, cities)
        sa.sa_search(timer, True)
        return sa.saBest

    def adjust_values(self):
        if not self.ks.items_used:
            return
        x, y = 0, 0
        for item in self.ks.items_used:
            x += self.city_cords[item][0]
            y += self.city_cords[item][1]
        self.sack_center = np.asarray([x/len(self.ks.items_used), y/len(self.ks.items_used)])
        for item in self.ks.items:
            if item.number not in self.ks.items_used:
                distance = np.linalg.norm(self.sack_center-self.city_cords[item.number])
                item.value = 1 / (distance + 0.0001)
