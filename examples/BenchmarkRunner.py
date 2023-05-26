import subprocess
import csv
import os

class BenchmarkRunner:
    def __init__(self, node=None, machine=None, ortce=True) -> None:
        self.set_env_path = '/shp/set_env.csh'
        self.ortce = ortce
        if ortce:
            try:
                subprocess.call(['/bin/bash', '-c', self.set_env_path])
            except Exception as e:
                raise e
            
            self.node = node
            self.machine = machine
            
        self.dev_size = 0
        self.vec_size = 0
        self.dev_filter = ""
        self.csv_path = "shp/dp.csv"
        self.txt_path = "shp/dp.txt"

    def run_dp_shp(self, path:str, dev_nums:list, vec_sizes:list, dev_type:str="gpu") -> None:

        for dev_num in sorted(dev_nums):
            self.__set_device_filter(dev_num, dev_type)
            for vec_size in sorted(vec_sizes):
                command = [path] + [str(2*dev_num)] + [str(vec_size)]
                if self.ortce:
                    command = ['srun', '-p', str(self.node), "-w", str(self.machine)] + command
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                output, err = process.communicate()
                # save it to txt/csv
                self.save_to_csv(output.decode())


    def __set_device_filter(self, dev_num, dev_type):
        for i in range(dev_num):
            dev_filter="*:" + dev_type + ":" + str(i)
            if i < dev_num-1:
                dev_filter+=", "
            self.dev_filter += dev_filter
        os.environ['SYCL_DEVICE_FILTER'] = self.dev_filter


    def run_dp_sycl_ucm():
        pass

    def run_dp_sycl_buffers():
        pass

    def run_dot_product_all():
        pass

    def compare():
        pass

    def save_to_csv(self, output):
        rows = output.splitlines()
        rows = [row.split(',') for row in rows]

        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            for row in rows:
                writer.writerow(row)

    def save_to_txt(self, binary_output):
        with open(self.txt_path, 'a') as f:
            f.write(binary_output.decode())


runner = BenchmarkRunner(node="QZ1B-SPR-4oam-PVC", machine="sdp125072", ortce=False)
runner.run_dp_shp("../build/examples/shp/dot_product_benchmark", [1], [128], "cpu")
        

