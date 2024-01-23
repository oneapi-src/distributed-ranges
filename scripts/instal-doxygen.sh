# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

set -xe
curl -s https://www.doxygen.nl/files/doxygen-1.9.6.linux.bin.tar.gz -o /tmp/dox.tgz
sudo tar zxf /tmp/dox.tgz -C /usr/local
sudo ln -s /usr/local/doxygen*/bin/* /usr/bin
sudo apt install -y \
     graphviz
