FROM intel/oneapi:latest

RUN apt-get update && apt-get install -y \
    build-essential \
    debhelper=13* \
    cmake \
    g++-12 \
    git \
    mpi-default-dev \
    libfmt-dev \
    librange-v3-dev \
    # libomp-dev \
    devscripts \
    dh-make \
    libgtest-dev \
    libcxxopts-dev \ 
    && apt-get clean

ENV DEBIAN_FRONTEND=noninteractive
ENV CXX=g++-12

COPY . /distributed-ranges

WORKDIR /distributed-ranges

CMD ["/bin/bash"]
