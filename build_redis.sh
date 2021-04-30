#!/usr/bin/env bash
set -e # Abort on error

mkdir -p build 

cd build
wget https://github.com/redis/redis/archive/refs/tags/6.2.2.tar.gz
tar -xzvf 6.2.2.tar.gz
cd redis-6.2.2
make
cp src/redis-server ../../pistarlab/thirdparty_lib
