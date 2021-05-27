import numpy as np


class LDS:
    def __init__(self, knapsack_problem):
        self.knapsack = knapsack_problem
        self.best_found = 0
        self.best_config = None
        self.current_config = [1 for i in range(self.knapsack.m)]

    def search(self):
        for num_errors in range(self.knapsack.m):
            print(num_errors)
            self.branch_search(0, self.knapsack.m-1, num_errors, 0,
                               [sack.capacity for sack in self.knapsack.sacks],
                               sum([item.value for item in self.knapsack.items]),
                               [1 for i in range(self.knapsack.m)])

    def branch_search(self, start_index, end_index, num_of_errors, value, room, estimate, config):
        if estimate < self.best_found:
            return
        # update if last item was taken or not
        if start_index > 0:
            # if item was taken
            if config[start_index - 1]:
                # update room remaining in sacks
                room = np.subtract(room, self.knapsack.items[start_index-1].weights)
                # if sack is not legal
                if not np.all(np.asarray(room) > 0):
                    return
                # update current value of problem
                value += self.knapsack.items[start_index-1].value
            # if item was not taken
            else:
                # reduce your estimation by the value of the item
                estimate -= self.knapsack.items[start_index-1].value

        # legal solution found
        if start_index == len(config):
            # check if solution is better than current
            if value > self.best_found:
                self.best_found = value
                self.best_config = config
                self.knapsack.arrange_by_config(self.best_config)
                print("best found: ", self.best_found)
                print("opt: ", self.knapsack.opt)
            return

        # found configuration -> find branch in DFS
        if num_of_errors == 0:
            return self.branch_search(start_index+1, end_index, num_of_errors, value, room.copy(), estimate, config)

        # loop to find all configuration with given number of errors
        for i in range(end_index - num_of_errors + 2):
            new_config = config.copy()
            new_config[end_index-i] = 0
            self.branch_search(start_index, end_index-i-1, num_of_errors-1, value, room.copy(), estimate, new_config)
