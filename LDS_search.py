import numpy as np


class LDS:
    def __init__(self, knapsack_problem):
        self.knapsack = knapsack_problem
        self.best_found = 0
        self.best_used = None
        self.best_items_used = None
        self.current_config = [1 for _ in range(self.knapsack.m)]
        self.errors = 0

    def search(self):
        for wave in range(self.knapsack.m):
            print(wave)
            self.unlimited_sack_search(wave)
            self.knapsack.clear_sacks()
        for item in self.best_items_used:
            self.knapsack.add_item_to_sacks(item)

    def branch_search(self, wave, dont_use=None):
        if dont_use is None:
            dont_use = []
        # find estimate on given configuration and unknown end
        estimate, partial_item = self.knapsack.neglecting_integrality_constraints(dont_use)
        if estimate < self.best_found and wave == 0:
            return
        if self.knapsack.value > self.best_found:
            self.best_found = self.knapsack.value
            self.best_items_used = self.knapsack.items_used.copy()
            print(self.knapsack)
        if partial_item is None:
            return

        if wave > 0:
            items_taken = self.knapsack.items_used.copy()
            items_taken.append(partial_item)
            for i in items_taken:
                dont_use_copy = dont_use.copy()
                items_taken_copy = items_taken.copy()
                dont_use_copy.append(i)
                items_taken_copy.remove(i)
                if i in self.knapsack.items_used:
                    self.knapsack.remove_item_from_sacks(i)
                    self.branch_search(wave-1, dont_use_copy)

    def unlimited_sack_search(self, wave, dont_use=None, last_error=-1):
        if dont_use is None:
            dont_use = []
        unused_items = [i for i in range(self.knapsack.m) if i not in self.knapsack.items_used and i not in dont_use]
        estimate = self.knapsack.value + sum([self.knapsack.items[item].value for item in unused_items])
        if estimate < self.best_found:
            return
        while self.knapsack.is_legal():
            if not unused_items:
                return
            self.knapsack.add_item_to_sacks(unused_items[0])
            last_item = unused_items[0]
            unused_items = unused_items[1:]
            if self.knapsack.value > self.best_found and self.knapsack.is_legal():
                self.best_found = self.knapsack.value
                self.best_items_used = self.knapsack.items_used.copy()
                print(self.knapsack)
        self.knapsack.remove_item_from_sacks(last_item)
        if wave > 0:
            used_items = self.knapsack.items_used.copy()
            config = self.knapsack.items_used.copy()
            while used_items:
                dont_use_copy = dont_use.copy()
                item = used_items.pop()
                if item == last_error:
                    return
                self.knapsack.remove_item_from_sacks(item)
                dont_use_copy.append(item)
                self.unlimited_sack_search(wave-1, dont_use_copy, item)
                self.knapsack.add_item_to_sacks(item)
                self.knapsack.clear_sacks()
                for i in config:
                    self.knapsack.add_item_to_sacks(i)
