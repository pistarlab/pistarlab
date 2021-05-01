# piSTAR Lab

piSTAR Lab is a modular AI development and experimentation platform.

WARNING: Under active development - not ready for general public use.

## Overview

piSTAR Lab is a modular deep reinforcement learning platform built to make AI development educational as well as fun.


### Tenets

### Features:
* Intuitive UI
* Python API, anthing you can do in the UI, you can do in Python as well
* Single and Multi Environments
* Track Experiments
* Built in Webbased IDE (via Theia (https://theia-ide.org/))
* Easy custom module creation

### Planned Features:
* Learning Resource: Videos, Documentation, and Tutorials  
* Missions which help users gain intuition about different aspects of reinforcement learning
* Composable Agents - agents built from reusable components
* Public Repositories for Agents Snapshots, Component Snapshots, and Modules
* Filter by Observation/Action Space
* Persistent Environments
* UI for easy Hyper parameter exploration
* Competitions



## Installation 

These instructions are for single node only. For cluster mode, see TODO

### Using PIP (Ubuntu only)

Requirements

#### install pistarlab PiPy package
```bash
git clone PATH_TO_REPO/pistarlab
cd pistarlab
pip install .
```

#### install node for UI and IDE

```bash
bash ./install_node.sh
bash ./build.sh
```

#### ffmpeg for video generation

```bash
apt-get install -y xvfb ffmpeg
```

### Using Docker



## Usage


### Connecting to the UI


localhost:8080

## Development

### Building Docker Image



## Configuration

### Root path

By default pistarlab stores data and configuration in the USER_HOME/pistarlab directory. This can be changed by using the PISTARLAB_ROOT environment variable

## Contributing

The project is still in it's infancy so there are plenty of areas that need work. See the Roadmap (TODO) for details.

### Development Setup

These instructions assume you are using [Anaconda](https://www.anaconda.com/products/individual) for your Python environment and you named your environment *pistarlab*.

1. Create environment

    ```bash
    conda create -n pistarlab  python=3.8
    conda activate pistarlab
    ```

1. Install external depdendancies

    on Ubuntu:

    ```bash
    sudo apt-get install -y xvfb ffmpeg
    ```

1. clone git repo

    ```bash
    git clode https://github.com/pistarlab/pistarlab
    ```

1. Install with pip in edit (development) mode.

    ```bash
    pip install -e .
    ```

1. Install extra development dependencies

    ```bash
    pip install -r requirements-dev.txt
    ```

1. Install Node (only required for UI development)

    on Ubuntu:

    ```bash
    sudo apt-get install -y nodejs npm
    ```

1. (Optional) Using with Jupyter Lab
    - You can install Jupyter Lab using ```pip install jedi==0.17.2 jupyterlab ipykernel```
    - TO make your environment avialable to jupyter, you can run ```python -m ipykernel install --user --name  pistarlab``` to add the kernel
    - start jupyter with (note:)
    ```jupyter lab```

### Making changes to the UI

The UI is build using Vuejs cli and requires npm to run.  Once setup, changes to the ui source code will be reflected immidiately in the browser.

1. Run the UI using ```npm run serve```
1. By default, changes will be reflected at http://localhost:8080

### Building for PiPy

1. Run Tests with tox

```bash
pip install tox
tox
```

1. Building wheel and source distribution and view files

```bash
rm -rf build dist *.egg-info && 
python setup.py bdist_wheel && python -m build --sdist --wheel && unzip -l dist/*.whl
```

1. Uploading to PiPy

```bash
pip install twine
twine upload dist/*
```

## Building the Documentation

### Rebuild API Docs

From the project root, run:

```bash
cd docs
sphinx-apidoc -o . ..
```

### Update the HTML

```bash
make html
```

## Building the UI

The build_ui.py script use npm to build the Vuejs project and then moves the files into the pistarlab package under uidist.

To run the build script, navigate to the project directory and run: 

```bash
build_ui.py
```

## Building Docker Dev Image

Install Docker: https://docs.docker.com/engine/install/ubuntu/

docker build docker/dev -t pistarlab/pistarlab-dev:latest


## Tips

### GPU

#### Missing .so error when running tensorflow

ensure LD_LIBRARY_PATH is correct

#### Testing GPU

Check if torch is detecting the GPU

```bash
python -c "import torch; print(torch.cuda.is_available());"
```

#### GPU not avilable

If your GPU was working previously, but suddenly is not accessable the the system. The following script may help the script at ```scripts/fix_gpu.sh``` may help.


# Windows Setup Notes: 

Install Miniconda
Install GitBash
Open MiniConda and create pistarlab env


theia ide: https://github.com/eclipse-theia/theia/blob/master/doc/Developing.md#building-on-windows
https://github.com/lukesampson/scoop#installation
```
Invoke-Expression (New-Object System.Net.WebClient).DownloadString('https://get.scoop.sh')

# or shorter
iwr -useb get.scoop.sh | iex
# IF SCOOP doesn't get added to path
 $env:Path += ";C:\Users\${USER}\scoop\shims"
```

# Cluster Notes

- Requires: postgresql
## 

## Switch to python env to 3.7.7
This is helpful because the docker version of ray uses 3.7.7. Multiple versions of python will create problems when pickling.
```conda install python=3.7.7```

remote install of packages

GOCHYAS
- file permission issues when using docker. files copied using rsync get permissions
- /tmp/ray_tmp_mount 



## NON DOCKER SETUP
- Install MINICONDA
    - wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    - bash ~/Miniconda3-latest-Linux-x86_64.sh
    - login logout
    - conda install python=3.7.7

- start cluster
    - ray up -vvvv ~/pistarlab/cluster.yaml 
- start pistarlab
    -python pistarlab/launcher.py --ray_address="192.168.1.31:6379" 
- Install plugins on nodes:
    - ray exec /home/brandyn/pistarlab/cluster.yaml  "pip install --user -e /home/pistarlabuser/app/pistarlab/plugins/pistarlab-envs-gym-main"
- Before Each Task:
    - ray rsync-up -vvv ~/pistarlab/cluster.yaml
- After Each Task or on demand:
    - ray rsync-down -vvv ~/pistarlab/cluster.yaml /home/pistarlabuser/pistarlab/data/ /home/brandyn/pistarlab/data/

# VENVS
-- just use miniconda


# CI

# Debugging

NOTE: We plan to support more user friendly debugging options in the future.

Our current solution uses the remote_pdb python module to enable remote debuging over telnet.

Steps:
1. Insert the following code snip where you want the debugger to started and wait for your input
```
from ..utils.remote_pdb import set_trace
set_trace() # you'll see the port number in the logs
```
1. When the above code is executed, it will print the port number to connect to in the logs.
1. use telnet to connect and use pdb as usual: for example ```telnet 127.0.0.1 PORTNUM_HERE```


