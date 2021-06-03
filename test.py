import os
import xlsxwriter as excel
from CVRP import TwoStepSolution as TSS
from CVRP import CVRP
from GeneticAlgorithm import GeneticAlgorithm as GA


runs = 10
directory = "cvrp_data/"
test_files = []
for file in os.listdir(directory):
    f = os.path.join(directory, file)
    if os.path.isfile(f):
        test_files.append(f)

two_step_res = [[None for _ in range(runs)] for _ in range(len(test_files))]
NSGA_res = [[None for _ in range(runs)] for _ in range(len(test_files))]

for test_num, test in enumerate(test_files):
    print(test_num)
    for i in range(runs):
        two_step_solution = TSS(test, output=False)
        two_step_solution.search(180)
        two_step_res[test_num][i] = two_step_solution.cost

        cvrp = CVRP(test)
        ga = GA(cvrp)
        ga.genetic(180)
        best_TSP = min(ga.gen_arr, key=lambda x: x.distance_val)
        cvrp.by_config(best_TSP.city_clusters)
        NSGA_res[test_num][i] = cvrp.cost

with excel.Workbook(os.path.join("data", "CVRP_RES.xlsx")) as book:
    worksheet = book.add_worksheet("NSGA")
    for col, data in enumerate(NSGA_res):
        worksheet.write_column(1, col, data)
    for col, test_name in enumerate(test_files):
        worksheet.write(0, col, test_name)

    worksheet = book.add_worksheet("2 step sol")
    for col, data in enumerate(two_step_res):
        worksheet.write_column(1, col, data)
    for col, test_name in enumerate(test_files):
        worksheet.write(0, col, test_name)

