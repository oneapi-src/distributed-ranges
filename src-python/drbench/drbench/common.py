# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

from collections import namedtuple
from enum import Enum

Model = Enum("Model", ["SHP", "MHP"])
Runtime = Enum("Runtime", ["SYCL", "DIRECT"])
Device = Enum("Device", ["GPU", "CPU"])


class Target(namedtuple("Target", "model runtime device")):
    def __repr__(self):
        return f"{self.model.name}_{self.runtime.name}_{self.device.name}"


targets = {
    "mhp_direct_cpu": Target(Model.MHP, Runtime.DIRECT, Device.CPU),
    "mhp_sycl_cpu": Target(Model.MHP, Runtime.SYCL, Device.CPU),
    "mhp_sycl_gpu": Target(Model.MHP, Runtime.SYCL, Device.GPU),
    "shp_sycl_cpu": Target(Model.SHP, Runtime.SYCL, Device.CPU),
    "shp_sycl_gpu": Target(Model.SHP, Runtime.SYCL, Device.GPU),
}
