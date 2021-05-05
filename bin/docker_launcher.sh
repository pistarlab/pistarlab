#!/usr/bin/env bash
echo "-------------------------------------------------------"
echo "Launching piSTAR Lab in Docker Container"
echo "-------------------------------------------------------"

mkdir -p ${HOME}/pistarlab_docker
# --gpus all
docker run --rm -i -t -u `id -u` \
    --shm-size=2g \
    -v ${HOME}/pistarlab_docker:/home/ray/pistarlab pistarlab/pistarlab-dev:latest $@
