import matplotlib.pyplot as plt
import pandas as pd


class CsvPlotter:
    def __init__(self) -> None:
        self.output_csv = "sycl_bench_output.csv"
        self.output_plots = "plots/"


def plot_time_of_devices(
    csv_file="examples/shp/sycl_bench_output.csv",
    output_plots="examples/shp/plots",
):
    # headers = ["dev_size", "vec_size", "sycl_median"]
    df = pd.read_csv(csv_file)
    # print("Contents in csv file:", df["vec_size"])

    vec = {}
    # sort for different vec_size
    for i, row in df.iterrows():
        try:
            vec[row["vec_size"]][row["dev_size"]] = row["sycl_median"]
        except:
            vec[row["vec_size"]] = {row["dev_size"]: row["sycl_median"]}

    for i, vec_size in enumerate(vec):
        plt.figure(i)
        data = vec[vec_size]
        x = data.keys()
        y = data.values()
        plt.plot(x, y, 'ro')
        plt.title(
            f"Median out of a device number for vector size - {vec_size}"
        )
        plt.xlabel("device number")
        plt.ylabel("sycl median")
        plt.savefig(output_plots + f"/vec{vec_size}.png")
    # plt.plot(df['dev_size'], df["sycl_median"], 'ro')
    # plt.savefig(output_plots +"/test.png")


def run_benchmarks():
    pass


plot_time_of_devices()
