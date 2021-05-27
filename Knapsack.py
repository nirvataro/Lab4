import numpy as np


class Knapsack:
    def __init__(self, input_file=None):
        self.m = None   # number of items
        self.n = None   # number of sacks
        self.sacks = []
        self.items = []
        self.items_used = []
        self.value = 0
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
        self.items = [Item(i, item_values[i], item_weights[i]) for i in range(self.m)]

    def neglecting_integrality_constraints(self, start_item):
        normalized_values = [[None for _ in range(len(self.items))] for _ in range(len(self.sacks))]
        for i, item in enumerate(self.items):
            for j, sack in enumerate(self.sacks):
                if item.weights[j]:
                    normalized_values[j][i] = item.value/item.weights[j]
                else:
                    normalized_values[j][i] = np.inf
        best_overall = 0
        for sack_number, item_value in enumerate(normalized_values):
            sack_capacity = self.sacks[sack_number].capacity
            current_value = 0
            while sack_capacity and not all(x == -1 for x in item_value):
                item = np.argmax(item_value)
                weight = self.items[item].weights[sack_number]
                if sack_capacity > weight:
                    sack_capacity -= weight
                    current_value += self.items[item].value
                    item_value[item] = -1
                else:
                    fraction = sack_capacity / weight
                    current_value += (self.items[item].value * fraction)
                    sack_capacity -= (weight * fraction)
                    break
            if current_value > best_overall:
                best_overall = current_value
        print(best_overall)
        return best_overall

    def arrange_by_config(self, config):
        self.clear_sacks()
        for i in range(len(config)):
            if config[i] == 1:
                self.add_item_to_sacks(i)
                self.value += self.items[i].value

    def add_item_to_sacks(self, item_index):
        self.items_used.append(item_index)
        for sack in self.sacks:
            sack.add_item(self.items[item_index])
        self.value += self.items[item_index].value

    def remove_item_from_sacks(self, item_index):
        self.items_used.remove(item_index)
        for sack in self.sacks:
            sack.remove_item(self.items[item_index])
        self.value -= self.items[item_index].value

    def clear_sacks(self):
        for item in self.items_used:
            for sack in self.sacks:
                sack.remove_item(item)
        self.items_used = []
        self.value = 0

    def is_legal(self):
        for sack in self.sacks:
            if sack.room < 0:
                return False
        return True


class Sack:
    def __init__(self, capacity, number):
        self.number = number
        self.capacity = capacity
        self.room = capacity
        self.items = []
        self.value = 0

    def add_item(self, item):
        self.items.append(item.copy())
        self.value += item.value
        self.room -= item.weights[self.number]

    def remove_item(self, item_number):
        for item in self.items:
            if item.number == item_number:
                self.value -= item.value
                self.room += item.weights[self.number]
        self.items = list(filter(lambda x: x.number != item.number, self.items))


class Item:
    def __init__(self, number, value, weights):
        self.number = number
        self.value = value
        self.weights = weights.copy()

    def copy(self):
        return Item(self.number, self.value, self.weights.copy())
