from Knapsack import Knapsack
from LDS_search import LDS
import os
import xlsxwriter as excel


directory = r"data\check"
test_files = []
for file in os.listdir(directory):
    f = os.path.join(directory, file)
    if os.path.isfile(f):
        test_files.append(f)
BBS = []
ULS = []
OPT = []
for test_num, test in enumerate(test_files):
    print(test_num)
    for alg in range(2):
        ks = Knapsack(test)
        lds = LDS(ks)
        lds.search(alg)
        if alg == 0:
            BBS.append(ks.value)
            OPT.append(ks.opt)
        if alg == 1:
            ULS.append(ks.value)

with excel.Workbook(os.path.join("data", "results.xlsx")) as book:
    worksheet = book.add_worksheet("תוצאות")
    worksheet.write(0, 0, "Test")
    worksheet.write(0, 1, "BBS")
    worksheet.write(0, 2, "ULS")
    worksheet.write(0, 3, "OPT")
    worksheet.write_column(1, 0, test_files)
    worksheet.write_column(1, 1, BBS)
    worksheet.write_column(1, 2, ULS)
    worksheet.write_column(1, 3, OPT)
