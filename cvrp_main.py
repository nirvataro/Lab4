from sys import argv
from CVRP import TwoStepSolution


def main_cvrp(file):
    tss = TwoStepSolution(file)
    tss.search(120)
    print(tss.cost)
    tss.draw()


if __name__ == '__main__':
    input_file = argv[1]
    main_cvrp(input_file)
