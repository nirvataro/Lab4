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
            self.branch_search(wave)
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
