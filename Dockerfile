# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

FROM intel/oneapi:latest

RUN DEBIAN_FRONTEND=noninteractive apt-get update --allow-insecure-repositories && apt-get install -y \
    build-essential \
    debhelper=13* \
    cmake \
    g++-13 \
    git \
    mpi-default-dev \
    libfmt-dev \
    librange-v3-dev \
    # intel-mkl-full \
    # libmkl-dev \
    # intel-oneapi-mkl-devel \
    # libomp-dev \
    devscripts \
    dh-make \
    lintian \
    libgtest-dev \
    libcxxopts-dev \
    && apt-get clean

ENV CXX=g++-13

WORKDIR /repo

CMD ["/bin/bash"]
