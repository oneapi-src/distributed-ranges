import csv
import subprocess


def run_mhp(vec_size:int, reps:int, filter: str, sycl_used:bool, n: int=None, pin_domain:str=None, pin_order:str=None, pin_cell:str=None):
    bench_filter = "BENCHMARK_FILTER="+filter
    try:
        mpi_pin_domain = "I_MPI_PIN_DOMAIN="+pin_domain
    except:
        mpi_pin_domain = ""
    try:
        mpi_pin_order = "I_MPI_PIN_ORDER="+pin_order
    except:
        mpi_pin_order = ""
    try:
        mpi_pin_cell = "I_MPI_PIN_CELL="+pin_cell
    except:
        mpi_pin_cell = ""
    bench_path = "./build/benchmarks/gbench/mhp/mhp-bench"
    if sycl_used:
        command = [bench_filter, mpi_pin_domain, mpi_pin_order, mpi_pin_cell, "mpirun", "-n", str(n), bench_path, "--sycl", "--vector-size", str(vec_size), "--reps", str(reps)]
    else:
        command = [bench_filter, mpi_pin_domain, mpi_pin_order, mpi_pin_cell, "mpirun", "-n", str(n), bench_path, "--vector-size", str(vec_size), "--reps", str(reps)]
    process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, err = process.communicate()


def run_shp(vec_size:int, reps:int, d: int, only_fsycl:bool, filter: str=None, kmp_aff:str=None):
    try:
        bench_filter = "BENCHMARK_FILTER="+filter
    except:
        bench_filter = ""
    try:
        kmp_affinity = "KMP_AFFINITY="+kmp_aff
    except:
        kmp_affinity=""
    if only_fsycl:
        bench_path = "./build/benchmarks/gbench/shp/shp-bench-only-fsycl" 
    else:
        bench_path = "./build/benchmarks/gbench/shp/shp-bench"
    command = [bench_filter, kmp_affinity, bench_path, "-d", str(d), "--vector-size", str(vec_size), "--reps", str(reps)]

    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, err = process.communicate()

# def save_to_csv(self, output):
#     rows = output.splitlines()
#     rows = [row.split(',') for row in rows]

#     with open(self.csv_path, 'a', newline='') as f:
#         writer = csv.writer(f)
#         for row in rows:
#             writer.writerow(row)

def save_to_txt(self, binary_output, txt_path:str="./benchmark.txt"):
    with open(txt_path, 'a') as f:
        f.write(binary_output.decode())

def run_all_benchmarks_fsycl_O3():
    vec_size = 1000000000
    reps = 100
    bench_filter = "Stream_"
    kmp_affinity = "compact"
    only_fsycl = True

    # shp
    d = 1
    run_shp(vec_size, reps, d, only_fsycl)
    run_shp(vec_size, reps, d, only_fsycl, bench_filter, kmp_affinity)

    d = 2
    run_shp(vec_size, reps, d, only_fsycl)
    run_shp(vec_size, reps, d, only_fsycl, bench_filter, kmp_affinity)

    # mhp/cpu
    sycl_used = False
    run_mhp(vec_size, reps, bench_filter, sycl_used)
    n = 24
    mpi_pin_domain = "core"
    mpi_pin_order = "compact"
    mpi_pin_cell = "unit"

    run_mhp(vec_size, reps, bench_filter, sycl_used, n, mpi_pin_domain, mpi_pin_order, mpi_pin_cell)
    n = 48
    run_mhp(vec_size, reps, bench_filter, sycl_used, n, mpi_pin_domain, mpi_pin_order, mpi_pin_cell)

    # mhp/sycl
    mpi_pin_domain = "socket"
    sycl_used = True
    n = 1
    run_mhp(vec_size, reps, bench_filter, sycl_used, n, mpi_pin_domain)
    run_mhp(vec_size, reps, bench_filter, sycl_used, n)
    n = 2
    run_mhp(vec_size, reps, bench_filter, sycl_used, n, mpi_pin_domain, mpi_pin_order, mpi_pin_cell)

def run_all_benchmarks_other_options():
    pass
