#!/usr/bin/env bash
echo "-------------------------------------------------------"
echo "Removing all data"
echo "-------------------------------------------------------"
pg_dump -U postgres -h localhost postgres > ${HOME}/pistarlab/postgres_data_backup.pgsql
dropdb -p 5432 -h localhost -U postgres -e postgres
createdb -p 5432 -h localhost -U postgres -e postgres
mv  ${HOME}/pistarlab/data  ${HOME}/pistarlab/data.`date +"%m%d%y_%H%M%S"`
