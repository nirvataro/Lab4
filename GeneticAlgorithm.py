import random
import time
import numpy as np
from psutil import cpu_freq
from scipy.spatial import distance
import copy


############## constants ###############
GA_POPSIZE = 1000        # ga population size
GA_ELITRATE = .2		    # elitism rate
GA_MUTATIONRATE = .25      # mutation rate
########################################


class Gen:
    def __init__(self, city_clusters, distance_matrix, city_cords, age=0):
        self.city_clusters = [clus for clus in city_clusters if clus != []]
        self.cluster_val, self.distance_val = self.calc_val(distance_matrix, city_cords)
        self.dominating = []
        self.domination_count = 0
        self.front = None
        self.crowd_distance = 0
        self.age = age

    def calc_val(self, distance_matrix, city_cords):
        distance_sum = 0
        cluster_val = 0
        for cluster in self.city_clusters:
            x, y = 0, 0
            last_city = 1
            for city in cluster:
                x += city_cords[city-1][0]
                y += city_cords[city - 1][1]
                distance_sum += distance_matrix[last_city-1][city-1]
                last_city = city
            distance_sum += distance_matrix[last_city][1]
            center = (x/len(cluster), y/len(cluster))
            for city in cluster:
                cluster_val += distance.euclidean(center, city_cords[city-1])
        return cluster_val*len(self.city_clusters), distance_sum

    def __str__(self):
        string = "CVRP Gen:\n" + "Config: " + str(self.city_clusters) + "\nClusters Value: "
        string += str(self.cluster_val) + "\nTotal Distance: " + str(self.distance_val)
        return string


class GeneticAlgorithm:
    def __init__(self, cvrp, popsize=GA_POPSIZE, mutation_rate=GA_MUTATIONRATE):
        # uncolored graph of the graph we are trying to color
        self.cvrp = cvrp
        # size of population
        self.pop_size = popsize
        # the chance of mutation in gen
        self.mutation_rate = mutation_rate
        # arrays of current and next generation
        self.gen_arr = self.init_population()
        self.fronts = []
        # best fitness
        self.best_fitness = 0
        # iteration count
        self.iterations = 0

    # print method
    def __str__(self):
        output = str(self.gen_arr[0])
        output += "\nAvg fitness of gen: {}".format(self.avg_fit())
        output += "\nFitness STD: {}".format(self.std_fit())

        return output

    # initializes population
    def init_population(self):
        gen_arr = [Gen(self.cvrp.generate_clusters(), self.cvrp.dist_matrix, self.cvrp.city_cords, random.randint(0, 4)) for _ in range(self.pop_size)]
        return gen_arr

    # sorts population by key fitness value
    def sort_by_fitness(self, timer):
        self.calculate_fronts()
        self.crowding_distance_sort()
        print(self)
        iter_time = time.time() - timer
        print("Total time of generation: {}".format(iter_time))
        print("Total clock ticks (CPU)) of generation: {}\n".format(iter_time * cpu_freq()[0] * (2 ** 20)))

    def calculate_fronts(self):
        for gen in self.gen_arr:
            gen.front = None
            gen.dominating = []
            gen.domination_count = 0
        for i, gen1 in enumerate(self.gen_arr):
            for gen2 in self.gen_arr[i+1:]:
                if gen1.cluster_val <= gen2.cluster_val and gen1.distance_val <= gen2.distance_val:
                    if gen1.cluster_val < gen2.cluster_val or gen1.distance_val < gen2.distance_val:
                        gen1.dominating.append(gen2)
                        gen2.domination_count += 1
                elif gen2.cluster_val <= gen1.cluster_val and gen2.distance_val <= gen1.distance_val:
                    if gen2.cluster_val < gen1.cluster_val or gen2.distance_val < gen1.distance_val:
                        gen2.dominating.append(gen1)
                        gen1.domination_count += 1
        front_count = 1
        unsorted_gens = len(self.gen_arr)
        while unsorted_gens:
            front = []
            for gen in self.gen_arr:
                if gen.front is None and gen.domination_count == 0:
                    front.append(gen)
                    gen.front = front_count
                    unsorted_gens -= 1
            for gen in self.gen_arr:
                if gen.front == front_count:
                    for dom in gen.dominating:
                        dom.domination_count -= 1
            front_count += 1
            self.fronts.append(front)

    def crowding_distance_sort(self):
        for front in self.fronts:
            front.sort(key=lambda x: x.cluster_val)
            front[0].crowd_distance = np.inf
            front[-1].crowd_distance = np.inf
            for i, gen in enumerate(front[1:-1], start=1):
                if front[0].cluster_val-front[-1].cluster_val:
                    gen.crowd_distance += (front[i+1].cluster_val-front[i-1].cluster_val)/\
                                      (front[0].cluster_val-front[-1].cluster_val)
            front.sort(key=lambda x: x.distance_val)
            front[0].crowd_distance = np.inf
            front[-1].crowd_distance = np.inf
            for i, gen in enumerate(front[1:-1], start=1):
                if front[0].distance_val-front[-1].distance_val:
                    gen.crowd_distance += (front[i+1].distance_val-front[i-1].distance_val)/\
                                      (front[0].distance_val-front[-1].distance_val)

    def elitism(self):
        self.gen_arr.clear()
        while self.pop_size - len(self.gen_arr):
            front = self.fronts.pop(0)
            pop_size_left = self.pop_size - len(self.gen_arr)
            end = len(front) if pop_size_left >= len(front) else pop_size_left
            for gen in front[:end]:
                self.gen_arr.append(gen)
        self.fronts = []

    # crossover method - if one of the parents is legal pass their color to child
    # otherwise, randomly choose a color between them
    def crossover(self, gen1, gen2):
        new_gen_clusters = []
        new_gen_room = []
        unchosen_cities = list(range(2, len(self.cvrp.goods)))
        clusters_number = max(len(gen1.city_clusters), len(gen2.city_clusters))
        for i in range(clusters_number):
            choose_cluster = random.randint(0, 1)
            chosen_gen = gen1 if choose_cluster == 0 else gen2
            if len(chosen_gen.city_clusters) <= i:
                break
            cluster = []
            cluster_weight = 0
            for city in chosen_gen.city_clusters[i]:
                if city in unchosen_cities:
                    cluster.append(city)
                    cluster_weight += self.cvrp.goods[city-1]
                    unchosen_cities.remove(city)
            new_gen_room.append(self.cvrp.capacity - cluster_weight)
            new_gen_clusters.append(cluster)
        while unchosen_cities:
            city = unchosen_cities.pop()
            cluster_ratings = [0 for _ in range(len(new_gen_clusters))]
            for i, cluster in enumerate(new_gen_clusters):
                if self.cvrp.goods[city-1] < new_gen_room[i]:
                    gen1_cluster, gen2_cluster = [], []
                    for cluster1, cluster2 in zip(gen1.city_clusters, gen2.city_clusters):
                        if city in cluster1:
                            gen1_cluster = cluster1
                        if city in cluster2:
                            gen2_cluster = cluster2
                    for city_in_cluster in new_gen_clusters[i]:
                        if city_in_cluster in gen1_cluster:
                            cluster_ratings[i] += 1
                        if city_in_cluster in gen2_cluster:
                            cluster_ratings[i] += 1
                else:
                    cluster_ratings[i] = -1
            if not all(x == -1 for x in cluster_ratings):
                cluster_num = np.argmax(cluster_ratings)
                cluster = new_gen_clusters[cluster_num]
                index = random.randint(0, len(cluster))
                cluster.insert(index, city)
                new_gen_room[cluster_num] -= self.cvrp.goods[city-1]
            else:
                new_gen_clusters.append([city])
                new_gen_room.append(self.cvrp.capacity - self.cvrp.goods[city-1])
        return Gen(new_gen_clusters, self.cvrp.dist_matrix, self.cvrp.city_cords)

    # checks if a node has a neighbor of same color
    def is_violating(self, node):
        for neighbor in node.neighbors:
            if node.color == neighbor.color:
                return True
        return False

    def get_parent(self, can_mate):
        g1, g2 = random.sample(can_mate, 2)
        if g1.front > g2.front:
            return g1
        if g2.front > g1.front:
            return g2
        if g1.crowd_distance > g2.crowd_distance:
            return g1
        return g2

    # Tournament selection
    def selection(self, can_mate):
        parent1 = self.get_parent(can_mate)
        parent2 = self.get_parent(can_mate)
        return parent1, parent2

    # mutates a gen by replacing it with a random neighbor
    def mutate(self, gen):
        clusters_copy = copy.deepcopy(gen.city_clusters)
        route = random.choice(clusters_copy)
        random.shuffle(route)
        return Gen(clusters_copy, self.cvrp.dist_matrix, self.cvrp.city_cords)

    # creates a new generation from the previous one
    def mate(self):
        # moves the best from this generation to the new one
        self.elitism()

        # updates ages of gens
        for gen in self.gen_arr:
            gen.age += 1

        # finds gens that are mature enough to mate
        can_mate = self.can_mate()

        # mating parents
        while len(self.gen_arr) < 2*self.pop_size:
            p1, p2 = self.selection(can_mate)
            self.gen_arr.append(self.crossover(p1, p2))

            # randomly mutates a gen
            if random.random() <= self.mutation_rate:
                self.gen_arr[-1] = self.mutate(self.gen_arr[-1])

    # chooses the gens that can mate according to age
    def can_mate(self):
        can_mate = []
        for gen in self.gen_arr:
            if gen.age >= 3:
                can_mate.append(gen)
        return can_mate

    def avg_fit(self):
        fit_arr = [g.distance_val for g in self.gen_arr]
        return np.mean(fit_arr)

    def std_fit(self):
        fit_arr = [g.distance_val for g in self.gen_arr]
        return np.std(fit_arr)

    # main loop of search
    def genetic(self, search_time=120):
        end_time = time.time() + search_time
        time_left = end_time - time.time()
        total_timer = time.time()
        while time_left > 0:
            gen_timer = time.time()

            self.sort_by_fitness(gen_timer)
            self.mate()

            time_left = end_time - time.time()
            self.iterations += 1

        total_time = time.time() - total_timer
        print("Total time : {}\nTotal clock ticks : {}\nTotal iterations:{}".format(total_time, total_time * cpu_freq()[
            0] * 2 ** 20, self.iterations + 1))
