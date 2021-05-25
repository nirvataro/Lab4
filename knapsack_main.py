from sys import argv
from Knapsack import Knapsack


def main_knapsack(file):
    ks = Knapsack(file)


if __name__ == '__main__':
    input_file = argv[1]
    main_knapsack(input_file)
