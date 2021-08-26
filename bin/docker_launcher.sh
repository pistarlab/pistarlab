#!/usr/bin/env bash
echo "-------------------------------------------------------"
echo "Launching piSTAR Lab in Docker Container"
echo "-------------------------------------------------------"
export LAUNCHER_PORT=7776
export BACKEND_PORT=7777
export WEB_UI_PORT=8080
export REDIS_PORT=7771
export IDE_PORT=7781
export STREAM_PORT=7778
export RAY_DASHBOARD_PORT=8265

mkdir -p ${HOME}/pistarlab_docker
# --gpus all
docker run --rm -i -t -u `id -u` \
    --shm-size=2g \
    -p ${IDE_PORT}:${IDE_PORT}  \
    -p ${LAUNCHER_PORT}:${LAUNCHER_PORT}  \
    -p ${BACKEND_PORT}:${BACKEND_PORT}  \
    -p ${WEB_UI_PORT}:${WEB_UI_PORT}  \
    -p ${REDIS_PORT}:${REDIS_PORT}  \
    -p ${STREAM_PORT}:${STREAM_PORT}  \
    -p ${RAY_DASHBOARD_PORT}:${RAY_DASHBOARD_PORT} \
    -v ${HOME}/pistarlab_docker:/home/ray/pistarlab pistarlab/pistarlab:latest --launcher_host="0.0.0.0" --$@
