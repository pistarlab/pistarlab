# piSTAR Lab

WARNING: Under active development - not ready for general public use.

# Overview

piSTAR Lab is a modular deep reinforcement learning platform built to make AI development accessible fun.

## Features:
* Web UI
* Plugin System for adding new agents, environments or tasks types
* Python API, anthing you can do in the UI, you can do in Python as well
* Run agents in single and multi player environments
* Experiment tracking
* Built in web-based IDE (via Theia (https://theia-ide.org/))
* Uses Ray Project (https://ray.io/) under the hood for distributed processing


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
    - XVFB to render without display (No MS Windows Support)
    - ffmpeg for video processing

    ```bash
    sudo apt-get install -y xvfb ffmpeg
    ```

### Usage

Launch piSTAR Lab Services
```bash
python pistarlab/launcher.py
```

- UI: http://localhost:8080

- Launcher Control Panel: http://localhost:7776


## Installation using Docker

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

### Usage with Docker

Launch piSTAR Lab Services
```bash
    ./bin/docker_launcher.sh 
```
