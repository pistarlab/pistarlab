

# <img src="docs/source/images/pistar_edit_w.png" alt="agent home" width="40"/> piSTAR Lab  

WARNING: Under active development - not ready for public use.

# Overview

piSTAR Lab is a modular deep reinforcement learning platform built to make AI development accessible and fun.

**Release: 0.0.1-dev** (early release)

## Features:
* Web UI
* Plugin System for adding new agents, environments or tasks types
* Python API, anthing you can do in the UI, you can do in Python as well
* Run agents in single and multi player environments
* Experiment tracking
* Built in web-based IDE (via Theia (https://theia-ide.org/))
* Uses Ray Project (https://ray.io/) under the hood for distributed processing

Licensed under an [Apache-2.0](https://github.com/pistarlab/pistarlab/blob/main/LICENSE) license.


## UI Screenshots

<br/> <img src="docs/source/images/pistarlab_agent_home.png" alt="agent home" width="600"/>  <br/>

<br/> <img src="docs/source/images/pistarlab_envs.png" alt="agent home" width="600"/>  <br/>

<br/> <img src="docs/source/images/pistarlab_newtask.png" alt="agent home" width="600"/>  <br/>

<br/> <img src="docs/source/images/pistarlab_session_stats.png" alt="agent home" width="600"/>  <br/>

<br/> <img src="docs/source/images/pistarlab_agent_stats.png" alt="agent home" width="600"/>  <br/>




# Quick Start 

These instructions are for single node only. For cluster mode, see TODO

## Installation with Anaconda

NOTE: Only tested on **Ubuntu**, but should also work on **OS X**. **MS Windows** users see [Installation using Docker](#Installation-using-Docker)


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

To launching piSTAR Lab, run:
```bash
pistarlab
```

- UI: http://localhost:8080

- Launcher Control Panel: http://localhost:7776


## Installation using Docker

*Recommended for MS Windows users*

1. Install Docker:

    Visit https://docs.docker.com/engine/install/

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

Launching piSTAR Lab

```bash
./bin/docker_launcher.sh 
```

# Contributing

We are still in an early phase of this release and have many loose ends to wrap up. If you are interested in contributing to piSTAR Lab, please reach out.