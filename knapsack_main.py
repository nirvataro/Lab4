from sys import argv
from CVRP import TwoStepSolution


def main_cvrp(file):
    tss = TwoStepSolution(file)
    tss.initial_clustering()


if __name__ == '__main__':
    input_file = argv[1]
    main_cvrp(input_file)
