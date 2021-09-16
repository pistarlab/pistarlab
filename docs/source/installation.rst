.. _installation:

Installation
============

NOTE: Only tested on Ubuntu. We recommend using Docker if installing on Windows.

with PyPi
--------------

#. Install Anaconda or Miniconda

   Visit https://www.anaconda.com/products/individual for instructions


#. Create Conda Virtual Environment

   .. code-block:: bash

      conda create -n pistarlab python=3.7



# Option 1

   .. code-block:: bash

   pip install https://github.com/pistarlab/pistarlab/archive/refs/heads/main.zip#egg=pistarlab[all]


# Option 1

   .. code-block:: bash
   
   git clone  --single-branch --depth=1 http://github.com/pistarlab/pistarlab/
   pip install -e .[all]


#. Install additional dependencies (Ubuntu only)
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

      docker pull docker.pkg.github.com/pistarlab/pistarlab/image:latest