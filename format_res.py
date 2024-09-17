import os
import re
import numpy as np
from matplotlib import pyplot as plt
from functools import cmp_to_key

rootdir = "./dest/"
res_dir = "./res/"
rand_regex = re.compile('mp_band_.+_.+_.+_.+\\.csv')
file_regex = re.compile('mp_.+_.+_.+\\.csv')

strong_data = {}
weak_data = {}
for root, dirs, files in os.walk(rootdir):
  for file in files:
    entry_count = -1
    mpi_size = -1
    name = ""
    if rand_regex.match(file):
        tmp = file[0:-4]
        res = tmp.split("_")
        entry_count = res[-3]
        mpi_size = res[-4]
        name = "band"
    elif (file_regex.match(file)):
        tmp = file[0:-4]
        res = tmp.split("_")
        entry_count = res[-1]
        mpi_size = res[-2]
        name = res[-3]
    if entry_count != -1:
        with open(rootdir + file) as handle:
            eq_res = handle.readline().split(",")
            row_res = handle.readline().split(",")
            eq_arr = np.array(eq_res).astype(np.float64)
            row_arr = np.array(row_res).astype(np.float64)
            ratio = round(float(entry_count) / float(mpi_size))
            if (ratio not in weak_data):
                weak_data[ratio] = []
            if ((name, entry_count) not in strong_data):
                strong_data[(name, entry_count)] = []
            strong_data[(name, entry_count)].append((mpi_size, eq_arr, row_arr))
            weak_data[ratio].append((mpi_size, entry_count, eq_arr, row_arr))

for entry in strong_data.items():
    if (len(entry[1]) == 1):
        continue

    sorted_list = entry[1]
    sorted_list = sorted(sorted_list, key=cmp_to_key(lambda x, y: int(x[0]) - int(y[0])))

    base_eq = np.mean(sorted_list[0][1]) / int(sorted_list[0][0])
    base_row = np.mean(sorted_list[0][2]) / int(sorted_list[0][0])

    index = []
    means_eq = []
    variance_eq = []
    means_row = []
    variance_row = []
    for info in sorted_list:
        index.append(int(info[0]))
        speedup_eq = base_eq / info[1]
        speedup_row = base_row / info[2]
        means_eq.append(np.mean(speedup_eq))
        variance_eq.append(np.var(speedup_eq))
        means_row.append(np.mean(speedup_row))
        variance_row.append(np.var(speedup_row))
    index = np.array(index)
    means_eq = np.array(means_eq)
    variance_eq = np.array(variance_eq)
    means_row = np.array(means_row)
    variance_row = np.array(variance_row)
    fig, ax = plt.subplots()

    ax.fill_between(index, means_eq - variance_eq, means_eq + variance_eq, alpha=.5, linewidth=0)
    ax.plot(index, means_eq, linewidth=2)
    ax.set(xlim=(0, 20), xticks=np.arange(1, 20, 2),
        ylim=(0, 20), yticks=np.arange(20))
    ax.plot(np.arange(20), np.arange(20))
    plt.savefig("res/" + entry[0][0] + "_" + entry[0][1] + "_eq_strong")
    
    fig, ax = plt.subplots()

    ax.fill_between(index, means_row - variance_row, means_row + variance_row, alpha=.5, linewidth=0)
    ax.plot(index, means_row, linewidth=2)
    ax.set(xlim=(0, 20), xticks=np.arange(1, 20, 2),
        ylim=(0, 20), yticks=np.arange(20))
    ax.plot(np.arange(20), np.arange(20))

    plt.savefig("res/" + entry[0][0] + "_" + entry[0][1] + "_row_strong")


for entry in weak_data.items():
    if (len(entry[1]) == 1):
        continue
    start = next(filter(lambda x: x[0] == '1', entry[1]), None)
    if (start == None):
        continue
    base_eq = np.mean(start[2])
    base_row = np.mean(start[3])

    sorted_list = entry[1]
    sorted_list = sorted(sorted_list, key=cmp_to_key(lambda x, y: int(x[0]) - int(y[0])))

    index = []
    means_eq = []
    variance_eq = []
    means_row = []
    variance_row = []
    for info in sorted_list:
        index.append(int(info[0]))
        speedup_eq = base_eq / info[2]
        speedup_row = base_row / info[3]
        means_eq.append(np.mean(speedup_eq))
        variance_eq.append(np.var(speedup_eq))
        means_row.append(np.mean(speedup_row))
        variance_row.append(np.var(speedup_row))
    index = np.array(index)
    means_eq = np.array(means_eq)
    variance_eq = np.array(variance_eq)
    means_row = np.array(means_row)
    variance_row = np.array(variance_row)
    fig, ax = plt.subplots()

    ax.fill_between(index, means_eq - variance_eq, means_eq + variance_eq, alpha=.5, linewidth=0)
    ax.plot(index, means_eq, linewidth=2)
    ax.set(xlim=(0, 20), xticks=np.arange(1, 20, 2),
        ylim=(0, 1.1), yticks=np.arange(0, 1, 0.2))
    ax.plot(np.arange(20), np.zeros(20) + 1)
    plt.savefig("res/" + str(entry[0]) + "_ratio_eq_weak")
    
    fig, ax = plt.subplots()

    ax.fill_between(index, means_row - variance_row, means_row + variance_row, alpha=.5, linewidth=0)
    ax.plot(index, means_row, linewidth=2)
    ax.set(xlim=(0, 20), xticks=np.arange(1, 20, 2),
        ylim=(0, 1.1), yticks=np.arange(0, 1, 0.2))
    ax.plot(np.arange(20), np.zeros(20) + 1)
    plt.savefig("res/" + str(entry[0]) + "_ratio_row_weak")

