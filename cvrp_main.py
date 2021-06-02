from sys import argv
from CVRP import TwoStepSolution
from CVRP import CVRP
from GeneticAlgorithm import GeneticAlgorithm as GA


def main_cvrp(file):
    cvrp1 = CVRP(file)
    cvrp2 = CVRP(file)
    ga = GA(cvrp1)
    ga.genetic(180)
    best_clusters = min(ga.gen_arr, key=lambda x: x.cluster_val)
    best_TSP = min(ga.gen_arr, key=lambda x: x.distance_val)
    cvrp1.by_config(best_clusters.city_clusters)
    print(cvrp1.cost)
    cvrp1.draw()
    cvrp2.by_config(best_TSP.city_clusters)
    print(cvrp2.cost)
    cvrp2.draw()


if __name__ == '__main__':
    input_file = argv[1]
    main_cvrp(input_file)
