import numpy as np


class LDS:
    def __init__(self, knapsack_problem):
        self.knapsack = knapsack_problem
        self.best_found = 0
        self.best_config = None
        self.current_config = [1 for i in range(self.knapsack.m)]
        self.current_value = 0
        self.current_room = [sack.capacity for sack in self.knapsack.sacks]
        self.estimate = self.knapsack.branch_and_bound_value

    def init_everything(self):
        self.current_config = [1 for i in range(self.knapsack.m)]
        self.current_value = 0
        self.current_room = [sack.capacity for sack in self.knapsack.sacks]
        self.estimate = self.knapsack.branch_and_bound_value

    def search(self):
        for num_errors in range(self.knapsack.m):
            self.branch_search(0, self.knapsack.m-1, num_errors)
            self.init_everything()

    def branch_search(self, start_index, end_index, num_of_errors):
        print("checking configuration: ", self.current_config)
        if start_index > 0:
            if self.current_config[start_index - 1]:
                self.current_room = np.subtract(self.current_room, self.knapsack.items[start_index-1].weights)
                self.current_value += self.knapsack.items[start_index-1].value
            else:
                self.estimate -= self.knapsack.items[start_index-1].value

        if not np.all(np.asarray(self.current_room) > 0):
            self.current_room = np.add(self.current_room, self.knapsack.items[start_index-1].weights)
            self.current_value -= self.knapsack.items[start_index - 1].value
            return False

        if start_index == len(self.current_config):
            if self.estimate > self.best_found:
                self.best_found = self.estimate
                self.best_config = self.current_config.copy()
                self.knapsack.arrange_by_config(self.best_config)
                print(self.best_config)
                print(self.knapsack.)
            return

        if num_of_errors == 0:
            return self.branch_search(start_index+1, end_index, num_of_errors)

        for i in range(end_index - num_of_errors + 2):
            self.current_config[end_index-i] = 0
            self.branch_search(start_index, end_index-i-1, num_of_errors-1)
            self.current_config[end_index-i] = 1
            self.estimate += self.knapsack.items[end_index-i].value