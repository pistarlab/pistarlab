#!/usr/bin/env bash
echo "-------------------------------------------------------"
echo "Launching Postgres Server"
echo "-------------------------------------------------------"

set -o allexport

set +o allexport

docker run -d -i --rm \
    --name pistarlabdb \
    -p 5432:5432 \
    -e POSTGRES_PASSWORD=pistarlab \
    -e PGDATA=/var/lib/postgresql/data/pgdata \
    -v ${HOME}/pistarlab/postgresdb:/var/lib/postgresql/data \
    postgres
