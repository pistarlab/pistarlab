.. _installation:

Installation
============

NOTE: Tested on Ubuntu and Windows 10. If you run into issues, we recommend using Docker.

with Pip
--------------

#. Install Anaconda or Miniconda

   Visit https://www.anaconda.com/products/individual for instructions


#. Install additional dependencies (Ubuntu only)
    - XVFB to render without display (No MS Windows Support)
    - ffmpeg for video processing

   .. code-block:: bash

      sudo apt-get install -y xvfb ffmpeg


#. Create Conda Virtual Environment

   .. code-block:: bash

      conda create -n pistarlab python=3.7
      conda activate pistarlab
      conda install pip

#. Option 1 (Package only)

   .. code-block:: bash

      pip install https://github.com/pistarlab/pistarlab/archive/refs/heads/main.zip#egg=pistarlab[all]

#. Option 2 (Package + Source)

   .. code-block:: bash
   
      git clone  --single-branch --depth=1 http://github.com/pistarlab/pistarlab/
      cd pistarlab
      pip install -e .[all]


with Docker
-----------

#. Install Docker:
    Visit: https://docs.docker.com/engine/install/

#. Clone Repo

   .. code-block:: bash

      docker pull docker.pkg.github.com/pistarlab/pistarlab/image:latest