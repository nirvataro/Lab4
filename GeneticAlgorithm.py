import random
import time
import numpy as np
from psutil import cpu_freq
from scipy.spatial import distance


############## constants ###############
GA_POPSIZE = 1000        # ga population size
GA_ELITRATE = .2		    # elitism rate
GA_MUTATIONRATE = .25      # mutation rate
########################################


class Gen:
    def __init__(self, city_clusters, distance_matrix, city_cords):
        self.city_clusters = city_clusters
        self.cluster_val, self.distance_val = self.calc_val(distance_matrix, city_cords)
        self.dominating = []
        self.domination_count = 0
        self.front = None

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
            center = (x/len(cluster), y/len(cluster))
            for city in cluster:
                cluster_val += distance.euclidean(center, city_cords[city-1])
        return cluster_val, distance_sum


class GeneticAlgorithm:
    def __init__(self, cvrp, popsize=GA_POPSIZE, elite_rate=GA_ELITRATE, mutation_rate=GA_MUTATIONRATE):
        # uncolored graph of the graph we are trying to color
        self.cvrp = cvrp
        # size of population
        self.pop_size = popsize
        # number of gens that continue to next generation
        self.elite_size = round(elite_rate*popsize)
        # the chance of mutation in gen
        self.mutation_rate = mutation_rate
        # arrays of current and next generation
        self.gen_arr = self.init_population()
        self.fronts = []
        # ages of gens
        self.ages = [random.randint(0, 4) for _ in range(self.pop_size)]
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
        gen_arr = [Gen(self.cvrp.generate_clusters(), self.cvrp.dist_matrix, self.cvrp.city_cords) for _ in range(self.pop_size)]
        return gen_arr

    # sorts population by key fitness value
    def sort_by_fitness(self, timer):
        self.calculate_fronts()
        self.crowding_distance_sort()
        print(self)
        iter_time = time.time() - timer
        print("Total time of generation: {}".format(iter_time))
        print("Total clock ticks (CPU)) of generation: {}\n".format(iter_time * cpu_freq()[0] * (2 ** 20)))

    def calculate_front(self):
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



    # takes GA_ELITRATE percent to next generation
    def elitism(self):
        for i in range(self.elite_size):
            self.buffer[i] = self.gen_arr[i].__deepcopy__()

    # crossover method - if one of the parents is legal pass their color to child
    # otherwise, randomly choose a color between them
    def crossover(self, gen1, gen2):
        newgen = HLS(self.empty_graph)
        for node_number in range(1, self.empty_graph.V+1):
            if not self.is_violating(gen1.graph.nodes[node_number]):
                newgen.color_node(newgen.graph.nodes[node_number], gen1.graph.nodes[node_number].color)
                continue
            if not self.is_violating(gen2.graph.nodes[node_number]):
                newgen.color_node(newgen.graph.nodes[node_number], gen2.graph.nodes[node_number].color)
                continue
            color = random.choice([gen1.graph.nodes[node_number].color, gen2.graph.nodes[node_number].color])
            newgen.color_node(newgen.graph.nodes[node_number], color)
        newgen.arrange_nodes()
        newgen.fitness = newgen.objective_function()
        return newgen

    # checks if a node has a neighbor of same color
    def is_violating(self, node):
        for neighbor in node.neighbors:
            if node.color == neighbor.color:
                return True
        return False

    # RWS selection
    def selection(self, can_mate):
        # sel will store selected parents to mate
        total_fitness = sum([gen.fitness for gen in can_mate])

        # randomize a number between 0-sum of all fitness, find where that gen is and use as parent
        ran_selection = random.uniform(0, total_fitness)
        current, j = 0, 0
        while current < ran_selection:
            current += can_mate[j].fitness
            j += 1
        parent1 = can_mate[j-1]
        # randomize a number between 0-sum of all fitness, find where that gen is and use as parent
        ran_selection = random.uniform(0, total_fitness)
        current, j = 0, 0
        while current < ran_selection:
            current += can_mate[j].fitness
            j += 1
        parent2 = can_mate[j-1]
        return parent1, parent2

    # mutates a gen by replacing it with a random neighbor
    def mutate(self, gen):
        return gen.random_neighbor()

    # creates a new generation from the previous one
    def mate(self):
        # moves the best from this generation to the new one
        self.elitism()

        # updates ages of gens
        for i, age in enumerate(self.ages):
            self.ages[i] += 1

        # finds gens that are mature enough to mate
        can_mate = self.can_mate()

        # mating parents
        for i in range(self.elite_size, self.pop_size):
            p1, p2 = self.selection(can_mate)
            self.buffer[i] = self.crossover(p1, p2)
            # zeroing new gens ages
            self.ages[i] = 0

            # randomly mutates a gen
            if random.random() <= self.mutation_rate:
                self.buffer[i] = self.mutate(self.buffer[i])
        # swaps gen_arr and buffer
        self.buffer, self.gen_arr = self.gen_arr, self.buffer

    # chooses the gens that can mate according to age
    def can_mate(self):
        can_mate = []
        for i, gen in enumerate(self.gen_arr):
            if self.ages[i] >= 3:
                can_mate.append(gen)
        return can_mate

    def avg_fit(self):
        fit_arr = [g.fitness for g in self.gen_arr]
        return np.mean(fit_arr)

    def std_fit(self):
        fit_arr = [g.fitness for g in self.gen_arr]
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
