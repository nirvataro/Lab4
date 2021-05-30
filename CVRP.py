from Knapsack import Knapsack, Item
import LDS_search
import numpy as np
from scipy.spatial import distance


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

    def add_truck(self):
        self.trucks.append(Truck(self.capacity))


class Truck:
    def __init__(self, capacity):
        self.room = capacity
        self.road = []
        self.center_cord = None


class TwoStepSolution(CVRP):
    def __init__(self, file=None):
        super().__init__(file)
        self.city_clusters = []
        self.sack_center = None
        self.ks = Knapsack(adjustable_values=self.adjust_values)

    def initial_clustering(self):
        while self.unvisited_cities:
            items =[]
            for index, city_number in enumerate(self.unvisited_cities):
                items.append(Item(index, 0.0001, [self.goods[city_number]]))
            self.ks.init_not_from_file(1, [self.capacity], items)
            self.ks.clear_sacks()
            lds = LDS_search.LDS(self.ks)
            lds.search(0, 5)
            cluster = [self.unvisited_cities[item] for item in self.ks.items_used]
            self.city_clusters.append([self.unvisited_cities[item]+1 for item in self.ks.items_used])
            print("cluster: ", cluster)
            self.unvisited_cities = [item for item in self.unvisited_cities if item not in cluster]
        print(self.city_clusters)


    def TSP_stage(self):
        return

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
                if distance == 0:
                    print(self.sack_center)
                item.value = 1 / distance
