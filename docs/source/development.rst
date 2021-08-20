Developer Notes
===============

Setting up your environment
---------------------------

Tested on Ubuntu

#. Install Anaconda or Miniconda

   Visit https://www.anaconda.com/products/individual for instructions


#. Create Conda Virtual Environment

   .. code-block:: bash

      conda create -n pistarlab python=3.7

#. Install PIP

   .. code-block:: bash

    conda install pip

#. Clone Repo and install

   .. code-block:: bash

    git clone https://github.com/pistarlab/pistarlab
    cd pistarlab
    pip install -e .

#. Install additional dependencies
    - XVFB to render without display (No MS Windows Support)
    - ffmpeg for video processing

   .. code-block:: bash

    sudo apt-get install -y xvfb ffmpeg

    
#. Install NPM/Nodejs for UI and Theia IDE

   .. code-block:: bash

    bash ./install_node.sh

#. Build Web IDE (optional)

   .. code-block:: bash

    bash ./build_ide.sh
    

on Windows [Experimental]
-------------------------

**Limitation:** no headless mode for many environments so rendering will open a window

#. Install Miniconda
#. Install GitBash
#. Follow Ubuntu Instructions

Troubleshooting
~~~~~~~~~~~~~~~~

**Building Theia IDE on Windows.**
* https://github.com/eclipse-theia/theia/blob/master/doc/Developing.md#building-on-windows

**Install Scoop**

See: https://github.com/lukesampson/scoop#installation

   .. code-block:: bash

    Invoke-Expression (New-Object System.Net.WebClient).DownloadString('https://get.scoop.sh')

    # or shorter
    iwr -useb get.scoop.sh | iex
    # IF SCOOP doesn't get added to path
    $env:Path += ";C:\Users\${USER}\scoop\shims"


Making changes to the UI
------------------------

The UI is build using Vuejs cli and requires npm to run.  Once setup, changes to the ui source code will be reflected immidiately in the browser.

Option 1:
# Start pistarlab with enable_dev_ui argument. Eg: ```pistarlab_launcher --enable_dev_ui```

Option 2:
#. Run the UI using ```npm run serve```
#. By default, changes will be reflected at http://localhost:8080


Building for Readonly Viewing
-----------------------------

   .. code-block:: bash

    pip install -e . -no-deps
    pip install -r requirements-webreadonly.txt

Building for PiPy
-----------------

#. Build Redis Server Binary and copy to pistarlab/thirdparty_lib

   .. code-block:: bash

    bash ./install_redis.sh_

#. Build and deploy UI in pistarlab/uidist/ package directory

   .. code-block:: bash

    bash ./build_ui.sh

#. Run Tests with tox

   .. code-block:: bash

    pip install tox
    tox

#. Building wheel and source distribution and view files

   .. code-block:: bash

    rm -rf build dist *.egg-info && 
    python setup.py bdist_wheel && python -m build --sdist --wheel && unzip -l dist/*.whl

#. Uploading to PiPy

   .. code-block:: bash

    pip install twine
    twine upload dist/*

Building the Documentation
--------------------------

#. Install dependencies


   .. code-block:: bash

    pip install -r docs/requirements.txt

#. Rebuild API Docs

   From the project root, run:

   .. code-block:: bash

    cd docs
    sphinx-apidoc -o source ../pistarlab

#. Update the HTML

   .. code-block:: bash

    cd docs
    make html

Building and Publishing a new Docker Image
------------------------------------------

Instructions on how to create a docker image from an Ubuntu environment

#. Make changes to docker file

#. Update requirements.txt
    .. code-block:: bash

    conda create -n pistarlab377 python=3.7.7
    conda activate pistarlab377
    pip install -e .
    pip freeze > requirements.txt

#. Run Docker Build

./build_docker
