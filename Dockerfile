FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

RUN apt-get update && apt install -y wget && apt install -y zstd

# install g++
RUN apt-get install -y gcc-7 g++-7 build-essential
ENV CC=/usr/bin/gcc-7
ENV CXX=/usr/bin/g++-7

WORKDIR /root/Util

# install cmake
RUN wget https://cmake.org/files/v3.15/cmake-3.15.0-Linux-x86_64.tar.gz
RUN tar xvfz cmake-3.15.0-Linux-x86_64.tar.gz
ENV CMAKE_HOME=/root/Util/cmake-3.15.0-Linux-x86_64
ENV PATH=$CMAKE_HOME/bin:$PATH

# install boost
RUN wget https://archives.boost.io/release/1.82.0/source/boost_1_82_0.tar.gz
RUN tar xvfz boost_1_82_0.tar.gz
WORKDIR /root/Util/boost_1_82_0
RUN ./bootstrap.sh --prefix=/root/local
RUN ./b2 install
ENV LD_LIBRARY_PATH=/root/local/lib:$LD_LIBRARY_PATH

# build cuMatch
WORKDIR /root/cuMatch
COPY ./ ./
RUN rm -rf build
RUN mkdir build
WORKDIR /root/cuMatch/build
RUN cmake ../
RUN make -j `nproc`

WORKDIR /root/cuMatch