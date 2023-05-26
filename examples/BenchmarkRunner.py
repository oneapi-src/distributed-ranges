import csv
import os
import subprocess
from typing import List


class BenchmarkRunner:
    def __init__(
        self, node: str = None, machine: str = None, ortce: bool = True
    ) -> None:
        self._set_env_path = 'examples/shp/set_env.csh'
        self._ortce = ortce
        if ortce:
            try:
                subprocess.call(['csh', self._set_env_path])
            except Exception as e:
                raise e
            self._sinfo_output = self.run_sinfo()

            self.node = node
            self.machine = machine

        self._dev_filter = ""
        self._csv_path = "examples/shp/dp.csv"
        self._txt_path = "examples/shp/dp.txt"

    @property
    def node(self) -> str:
        return self._node

    @node.setter
    def node(self, node: str) -> None:
        if self._ortce and node in self._sinfo_output:
            self._node = node

    @property
    def machine(self) -> str:
        return self._machine

    @machine.setter
    def machine(self, machine: str) -> None:
        if self._ortce and machine in self._sinfo_output:
            self._machine = machine

    @property
    def csv_path(self) -> str:
        return self._csv_path

    @property
    def txt_path(self) -> str:
        return self._txt_path

    def run_dp_shp(
        self,
        path: str,
        dev_nums: List[int],
        vec_sizes: List[int],
        dev_type: str = "gpu",
    ) -> None:
        for dev_num in sorted(dev_nums):
            self.__set_device_filter(dev_num, dev_type)
            for vec_size in sorted(vec_sizes):
                command = [path] + [str(2 * dev_num)] + [str(vec_size)]
                if self._ortce:
                    command = [
                        'srun',
                        '-p',
                        str(self.node),
                        "-w",
                        str(self.machine),
                    ] + command
                process = subprocess.Popen(
                    command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                output, err = process.communicate()
                # save it to txt/csv
                self.save_to_csv(output.decode())

    def __set_device_filter(self, dev_num: int, dev_type: str):
        for i in range(dev_num):
            dev_filter = "*:" + dev_type + ":" + str(i)
            if i < dev_num - 1:
                dev_filter += ", "
            self._dev_filter += dev_filter
        os.environ['SYCL_DEVICE_FILTER'] = self._dev_filter

    def run_dot_product_all(
        self, dev_num: List[int], vec_size: List[int], dev_type: str
    ):
        # shp
        self.run_dp_shp(
            "./build/examples/shp/dot_product_benchmark",
            dev_num,
            vec_size,
            dev_type,
        )
        # sycl usm
        self.run_dp_shp(
            "./build/examples/shp/dot_product_sycl",
            dev_num,
            vec_size,
            dev_type,
        )
        # sycl buffers
        # self.run_dp_shp("./build/examples/shp/dot_product_sycl_buff", dev_num, vec_size, dev_type)

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

    @staticmethod
    def run_sinfo() -> str:
        command = "sinfo"
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        output, err = process.communicate()
        return output.decode()


runner = BenchmarkRunner(
    node="QZ1B-SPR-4oam-PVC", machine="sdp125072", ortce=True
)
# runner.run_dp_shp("./build/examples/shp/dot_product_benchmark", [1], [128], "gpu")
# runner.run_dp_shp("./build/examples/shp/dot_product_sycl", [1], [128], "cpu")
