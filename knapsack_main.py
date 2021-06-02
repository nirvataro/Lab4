from sys import argv
from Knapsack import Knapsack
from LDS_search import LDS


def main_knapsack(file):
    ks = Knapsack(file)
    lds = LDS(ks)
    lds.search(0, 120)
    print(ks)


if __name__ == '__main__':
    input_file = argv[1]
    main_knapsack(input_file)