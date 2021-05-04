.. _installation:

Installation
============

with Anaconda
--------------

#. Install Anaconda or Miniconda

   Visit https://www.anaconda.com/products/individual for instructions

#. Install PIP

   .. code-block:: bash

    conda install pip

#. Clone Repo and install

   .. code-block:: bash

    git clone https://github.com/pistarlab/pistarlab
    cd pistarlab
    pip install -e .
    
#. Build Redis

   .. code-block:: bash

    bash ./install_redis.sh_
    
#. Install node for UI and IDE

   .. code-block:: bash

    bash ./install_node.sh
    bash ./build_ui.sh
    bash ./build_ide.sh #optional
    

#. Install additional dependencies
    - XVFB to render without display (No MS Windows Support)
    - ffmpeg for video processing

   .. code-block:: bash

    sudo apt-get install -y xvfb ffmpeg
    

with Docker
-----------

#. Install Docker:
    Visit: https://docs.docker.com/engine/install/

#. Clone Repo

   .. code-block:: bash

    git clone https://github.com/pistarlab/pistarlab
    cd pistarlab

#. Build Docker Image

   .. code-block:: bash

    ./build_docker.sh



on Windows [Experimental]
-------------------------

NOTE: It is recommended to use the Docker Setup Instead.

**Limitation:** no headless mode for many environments so rendering will open a window

#. Install Miniconda
#. Install GitBash
#. Follow Ubuntu Instructions

Troubleshooting
~~~~~~~~~~~~~~~~

**Building Theia IDE on Windows.**
* https://github.com/eclipse-theia/theia/blob/master/doc/Developing.md#building-on-windows

**Install Scoop**

* https://github.com/lukesampson/scoop#installation

   .. code-block:: bash

    Invoke-Expression (New-Object System.Net.WebClient).DownloadString('https://get.scoop.sh')

    # or shorter
    iwr -useb get.scoop.sh | iex
    # IF SCOOP doesn't get added to path
    $env:Path += ";C:\Users\${USER}\scoop\shims"

