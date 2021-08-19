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

#. Install PIP

   .. code-block:: bash

    pip install -U "pistarlab[all]"


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