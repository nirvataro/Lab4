from sys import argv
from CVRP import TwoStepSolution as TSS


def main_cvrp(file):
    two_step_solution = TSS(file, output=True)
    two_step_solution.search()
    print(two_step_solution)
    two_step_solution.draw()


if __name__ == '__main__':
    input_file = argv[1]
    main_cvrp(input_file)
