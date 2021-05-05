Advanced
================

Settings
--------


Data and Configuration Path
~~~~~~~~~~~~~~~~~~~~~~~~~~~
By default pistarlab stores data and configuration in the **$HOME/pistarlab/** directory. (eg: /home/$USER/pistarlab) This can be changed by using the **PISTARLAB_ROOT** environment variable

If you are using *Docker*, PISTARLAB_ROOT is under **$HOME/pistarlab_docker/**.  This has been changed for Docker to avoid permission issues that Docker may instroduce.

**Configuration** is stored in the **config.yaml** file located at PISTARLAB_ROOT

**Data** is stored under PISTARLAB_ROOT/data

Debugging
---------
Our current solution uses the remote_pdb python module to enable remote debuging over telnet.

**To use debugging, do the following:**

#. Insert the following code snip where you want the debugger to started and wait for your input

.. code-block:: python

   from ..utils.remote_pdb import set_trace
   set_trace() # you'll see the port number in the logs

#. When the above code is executed, it will print the port number to connect to in the logs.
#. use telnet to connect and use pdb as usual: for example ```telnet 127.0.0.1 PORTNUM_HERE```

Cluster Mode
------------------

.. warning::

    Experimental

Cluster mode is handled by Ray. For more information visit ray.io.

WIP, this documentation is incomplete and more testing needed.

Requirements
* PostgreSQL to communicate over network (Temporary)
* Python Version should be same on all nodes

with Docker
-----------

* Switch to python env to 3.7.7 *
This is helpful because the docker version of ray uses 3.7.7. Multiple versions of python will create problems when pickling.
  conda install python=3.7.7

**Docker Related Issue with using Ray**

* file permission issues when using docker. files copied using rsync get permissions
* /tmp/ray_tmp_mount 

without Docker
--------------

* Install MINICONDA
    #. wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    #. bash ~/Miniconda3-latest-Linux-x86_64.sh
    #. login logout
    #. conda install python=3.7.7

** Configuration **

Config file path: ~/pistarlab/cluster.yaml

TODO, see  for details

Startup
~~~~~~~

#. Starting cluster

   .. code-block:: bash

    ray up -vvvv ~/pistarlab/cluster.yaml

#. Launch piSTAR Lab on head node

   .. code-block:: bash

    python pistarlab/launcher.py --ray_address="IP_ADDRESS:6379" 


Install plugins on nodes
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   ray exec $HOME/pistarlab/cluster.yaml  "pip install --user -e /home/pistarlabuser/app/pistarlab/plugins/pistarlab-envs-gym-main"


Sync Data with Nodes
~~~~~~~~~~~~~~~~~~~~~~~~

Before each task run

.. code-block:: bash

   ray rsync-up -vvv ~/pistarlab/cluster.yaml


After each task (or as needed)

.. code-block:: bash

   ray rsync-down -vvv ~/pistarlab/cluster.yaml /home/pistarlabuser/pistarlab/data/ $HOME/pistarlab/data/


