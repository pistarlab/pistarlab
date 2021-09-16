

# <img src="docs/source/images/pistar_logo_black_solid.png"  alt="agent home" width="40"/> piSTAR Lab  

<!-- ![PyPI](https://img.shields.io/pypi/v/pistarlab)  -->
![PyPI - License](https://img.shields.io/pypi/l/pistarlab) 
[![Documentation Status](https://readthedocs.org/projects/pistarlab/badge/?version=latest)](https://pistarlab.readthedocs.io/en/latest/?badge=latest)

WARNING: This is an <u>**early release**</u>.

# Overview

piSTAR Lab is a modular deep reinforcement learning platform built to make AI experimentation accessible and fun.

**Documentation** https://pistarlab.readthedocs.io

## Features
* Web UI
* Extension System for adding new agents, environments or tasks types
* Python API, anthing you can do in the UI, you can do in Python as well
* Run agents in single and multi player environments
* Experiment tracking
* Uses Ray Project (https://ray.io/) under the hood for distributed processing
* Includes piSTAR [Landia](http://github.com/pistarlab/landia) a hackable Multi Agent Envrionment
* More to come

## Known Issues/Limitations
- Cluster mode is under development and not recommended at this time
- Running remotely requires SSH tunneling. All services must be running on localhost
- Mac not tested
- extension installation requires restarting piSTAR Lab to complete

## UI Screenshots

<br/> <img src="docs/source/images/pistarlab_demo1.gif" alt="agent home" width="600"/>  <br/>

<br/> <img src="docs/source/images/envs.png" alt="agent home" width="600"/>  <br/>

<br/> <img src="docs/source/images/assign_multi_agent_envs.png" alt="agent home" width="600"/>  <br/>

<br/> <img src="docs/source/images/agent_training1.png" alt="agent home" width="600"/>  <br/>


# Quick Start 
Detailed documentation is published at https://pistarlab.readthedocs.io

**Notes**
* Only tested on **Ubuntu**, but should also work on **OS X**. **MS Windows** users see [Installation using Docker](#Installation-using-Docker)
* Suggest using Anaconda or Miniconda for Python installation (visit https://www.anaconda.com/products/individual for instructions)
* Requires pip and python 3.7 or 3.8

## Installation

### Requirements

* Python 3.7+ with conda (recommended) or venv
    * Suggested install methods
        * Miniconda (https://docs.conda.io/en/latest/miniconda.html) 
        * Anaconda (https://www.anaconda.com/products/individual)
* FFMPEG (https://www.ffmpeg.org/download.html)
    * Required for episode recordings
    * Windows install instructions: https://www.wikihow.com/Install-FFmpeg-on-Windows


### Option 1
```bash
pip install https://github.com/pistarlab/pistarlab/archive/refs/heads/main.zip#egg=pistarlab[all]
```

### Option 2
if you intend to make modifications to pistarlab
```bash
git clone  --single-branch --depth=1 http://github.com/pistarlab/pistarlab/
pip install -e .[all]
```

NOTE: If install command fails, try with quotes  ```pip install -e ."[all]"```


## Usage

To launch piSTAR Lab UI, run:
```bash
pistarlab_launcher
```

Open browser to: http://localhost:7777


# Contributing

We are still in an early phase of this release but if you are interested in contributing to piSTAR Lab, please reach out.