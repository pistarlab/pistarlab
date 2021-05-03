# piSTAR Lab

WARNING: Under active development - not ready for general public use.

# Overview

piSTAR Lab is a modular deep reinforcement learning platform built to make AI development educational and fun.

## Features:
* Intuitive UI
* Python API, anthing you can do in the UI, you can do in Python as well
* Run agents in single and multi player environments
* Experiment tracking
* Built in web-based IDE (via Theia (https://theia-ide.org/))
* Plugin System for adding new agents, environments or tasks types
* Uses Ray Project (https://ray.io/) under the hood for distributed processing


## Planned Features:
* Composable Agents - agents built from reusable components
* Public repositories for agents and component snapshots
* Filter by observation/action space
* Persistent environments
* UI for easy hyper parameter exploration
* Learning resource: videos, documentation, and tutorials  
* Missions which help users gain intuition about different aspects of reinforcement learning
* Hosted competitions


# Installation 

These instructions are for single node only. For cluster mode, see TODO

## Using Anaconda

***Only tested on Ubuntu***

1. Install Anaconda or Miniconda
Visit https://www.anaconda.com/products/individual for instructions

1. Install PIP
    ```bash
    conda install pip
    ```

1. Clone Repo and install
    ```bash
    git clone https://github.com/pistarlab/pistarlab
    cd pistarlab
    pip install -e .
    ```
1. build Redis
    ```bash
    bash ./install_redis.sh_
    ```
1. install node for UI and IDE
    ```bash
    bash ./install_node.sh
    bash ./build_ui.sh
    bash ./build_ide.sh #optional
    ```

1. install additional dependencies
    - xvfb to render without display (No MS Windows Support)
    - ffmpeg for video processing

    ```bash
    sudo apt-get install -y xvfb ffmpeg
    ```

## Launching UI 


```bash
python pistarlab/launcher.py
```

- UI: http://localhost:8080

- Launcher Control Panel: http://localhost:7776


## Install using Docker 
1. Install Ddocker:
    https://docs.docker.com/engine/install/

1. Clone Repo
    ```bash
    git clone https://github.com/pistarlab/pistarlab
    cd pistarlab
    ```
1. Build Docker Image
    ```
    ./build_docker.sh
    ```

### Launch UI    

```bash
    .bin/docker_launcher.sh 
```

# Settings

## Root path

By default pistarlab stores data and configuration in the **$HOME/pistarlab/** directory. (eg: /home/$USER/pistarlab) This can be changed by using the **PISTARLAB_ROOT** environment variable

## config.yaml

Default Path: $PISTARLAB_ROOT/config.yaml

# Windows Setup [**Experimental**]

NOTE: It is recommended to use the Docker Setup

## Limitations:
- no headless mode for many environments so rendering will open a window

1. Install Miniconda
1. Install GitBash
1. Follow Ubuntu Instructions

## Troubleshooting

Building Theia IDE on Windows.
* https://github.com/eclipse-theia/theia/blob/master/doc/Developing.md#building-on-windows

Install Scoop
* https://github.com/lukesampson/scoop#installation
    ```
    Invoke-Expression (New-Object System.Net.WebClient).DownloadString('https://get.scoop.sh')

    # or shorter
    iwr -useb get.scoop.sh | iex
    # IF SCOOP doesn't get added to path
    $env:Path += ";C:\Users\${USER}\scoop\shims"
    ```


# Plugin Development

TODO


## Manifest
Manfiest files are used to speed up the installation of Plugins. They are especially useful for Environments where probing is required.

**Example of creating a manifest**
```bash
xvfb-run python pistarlab/plugin_tools.py --action=save_manifest --plugin_path PATH_TO_PLUGIN/pistarlab-envs-gym-main
```

# Cluster Mode Setup Notes [**Experimental**]


Cluster mode is handled by Ray

WIP, this documentation is incomplete and more testing needed.

* Requires: PostgreSQL to communicate over network (Temporary)

## Switch to python env to 3.7.7
This is helpful because the docker version of ray uses 3.7.7. Multiple versions of python will create problems when pickling.
```conda install python=3.7.7```

remote install of packages

### GOCHYAS
- file permission issues when using docker. files copied using rsync get permissions
- /tmp/ray_tmp_mount 

## NON DOCKER SETUP
- Install MINICONDA
    - wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    - bash ~/Miniconda3-latest-Linux-x86_64.sh
    - login logout
    - conda install python=3.7.7

Starting cluster
```bash
ray up -vvvv ~/pistarlab/cluster.yaml 
```

start pistarlab
```bash
python pistarlab/launcher.py --ray_address="IP_ADDRESS:6379" 
```

Install plugins on nodes:
```bash
ray exec $HOME/pistarlab/cluster.yaml  "pip install --user -e /home/pistarlabuser/app/pistarlab/plugins/pistarlab-envs-gym-main"
```

Before each task run:
```bash
ray rsync-up -vvv ~/pistarlab/cluster.yaml
```

After Each Task or as needed
```bash
ray rsync-down -vvv ~/pistarlab/cluster.yaml /home/pistarlabuser/pistarlab/data/ $HOME/pistarlab/data/
```



## Debugging

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

# Troubleshooting
## Missing .so error when running tensorflow

ensure LD_LIBRARY_PATH is correct

## Testing GPU

Check if torch is detecting the GPU

```bash
python -c "import torch; print(torch.cuda.is_available());"
```

## My GPU is suddenly not avilable (Ubuntu)

If your GPU was working previously, but suddenly is not accessable the the system. The following script may help the script at ```scripts/fix_gpu.sh``` may help.


# Developer Instructions

## Making changes to the UI

The UI is build using Vuejs cli and requires npm to run.  Once setup, changes to the ui source code will be reflected immidiately in the browser.

1. Run the UI using ```npm run serve```
1. By default, changes will be reflected at http://localhost:8080

## Building for PiPy

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

1. Rebuild API Docs

    From the project root, run:

    ```bash
    cd docs
    sphinx-apidoc -o . ..
    ```

1. Update the HTML

    ```bash
    make html
    ```

## Building Docker Dev Image


Install Docker: https://docs.docker.com/engine/install/ubuntu/

Run
```bash
./build_docker
```
