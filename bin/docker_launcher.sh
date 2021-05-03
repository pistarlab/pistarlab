#!/usr/bin/env bash
echo "-------------------------------------------------------"
echo "Launching piSTAR Lab in Docker Container"
echo "  - Please update the .env file to change which ports   "
echo "      are forward from docker."
echo "-------------------------------------------------------"
# Ports are set in .env.dev
set -o allexport
source bin/.env
set +o allexport
mkdir -p ${HOME}/pistarlab_docker
# --gpus all

docker run --rm -i -t -u `id -u` \
    --shm-size=2g \
    -v ${HOME}/pistarlab_docker:/home/ray/pistarlab \
    -p ${IDE_PORT}:${IDE_PORT}  \
    -p ${LAUNCHER_PORT}:${LAUNCHER_PORT}  \
    -p ${BACKEND_PORT}:${BACKEND_PORT}  \
    -p ${WEB_UI_PORT}:${WEB_UI_PORT}  \
    -p ${REDIS_PORT}:${REDIS_PORT}  \
    -p ${STREAM_PORT}:${STREAM_PORT}  \
    -p ${RAY_DASHBOARD_PORT}:${RAY_DASHBOARD_PORT} \
    -e PYTHONUSERBASE=/home/ray/pistarlab/plugins/site-packages/ \
    pistarlab/pistarlab-dev:latest python pistarlab/launcher.py $@
