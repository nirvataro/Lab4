import numpy as np


class Knapsack:
    def __init__(self, input_file=None):
        self.m = None   # number of items
        self.n = None   # number of sacks
        self.sacks = []
        self.items = []
        self.value = None
        self.opt = None
        self.from_file = False
        if input_file:
            self.from_file = True
            self.extract(input_file)

    def extract(self, file):
        text = open(file, 'r+')
        text = text.read()
        text = text.split()
        self.n, self.m = int(text[0]), int(text[1])
        start_index = 2
        item_values = [int(value) for value in text[start_index:start_index+self.m]]
        start_index = start_index+self.m
        sack_capacities = [int(cap) for cap in text[start_index:start_index+self.n]]
        start_index = start_index+self.n
        item_weights = []
        for i in range(self.n):
            weights = []
            for j in range(self.m):
                weights.append(int(text[start_index+i*self.m + j]))
            item_weights.append(weights)
        item_weights = np.array(item_weights).T.tolist()
        self.opt = int(text[start_index + self.m*self.n])
        self.sacks = [Sack(sack_capacities[i], i) for i in range(self.n)]
        self.items = [Item(item_values[i], item_weights[i], i) for i in range(self.m)]


class Sack:
    def __init__(self, capacity, number):
        self.number = number
        self.capacity = capacity
        self.items = []
        self.value = None

    def add_item(self, item):
        self.items.append(item.copy())
        self.value += item.value
        self.capacity -= item.weights(self.number)


class Item:
    def __init__(self, value, weights, number):
        self.number = number
        self.value = value
        self.weights = weights.copy()
