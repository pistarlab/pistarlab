Usage
=====


**To launch piSTAR Lab, run:**

.. code-block:: bash

   pistarlab_launcher


Open your browser to http://localhost:7777


**To launch the piSTAR Lab docker container, run:**

.. code-block:: bash

    mkdir pistarlab_docker ; docker run --rm -i -t -u `id -u` \
        --shm-size=2g \
        -p 7776:7776  \
        -p 7777:7777  \
        -p 7778:7778  \
        -p 8265:8265 \
        -v ${HOME}/pistarlab_docker:/home/ray/pistarlab pistarlab/pistarlab:latest  --launcher_host="0.0.0.0"

Open your browser to http://localhost:7777