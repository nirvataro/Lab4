import numpy as np


class LDS:
    def __init__(self, knapsack_problem):
        self.knapsack = knapsack_problem
        self.best_found = 0
        self.best_used = None
        self.best_unused = None
        self.current_config = [1 for _ in range(self.knapsack.m)]
        self.errors = 0

    def search(self):
        for num_errors in range(self.knapsack.m):
            print(num_errors)
            self.branch_search(0, num_errors)

    def branch_search(self, start_index, num_of_errors, unused_items=None):
        if unused_items is None:
            unused_items = []
        # find estimate on given configuration and unknown end
        estimate, partial_item = self.knapsack.neglecting_integrality_constraints()
        if partial_item is None:
            return
        if estimate < self.best_found:
            return

        # legal solution found
        if start_index == self.knapsack.m:
            # check if solution is better than current
            if self.knapsack.value > self.best_found:
                self.best_found = self.knapsack.value
                self.best_unused = unused_items
                print("best found: ", self.best_found)
                print("opt: ", self.knapsack.opt)
            return

        # found configuration -> find branch in DFS
        if num_of_errors == 0:
            # if item was taken
            if start_index not in unused_items:
                # update room remaining in sacks and value
                self.knapsack.add_item_to_sacks(start_index)
                # if sack is not legal
                if not self.knapsack.is_legal():
                    self.knapsack.remove_item_from_sacks(start_index)
                    return
            return self.branch_search(start_index+1, num_of_errors, unused_items)

        # loop to find all configuration with given number of errors
        start_range = unused_items[-1] if unused_items else self.knapsack.m
        for i in range(start_range-1, num_of_errors-2, -1):
            temp = unused_items.copy()
            temp.append(i)
            self.branch_search(start_index, num_of_errors-1, temp)
            temp.remove(i)
