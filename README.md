

# <img src="docs/source/images/pistar_edit_w.png" alt="agent home" width="40"/> piSTAR Lab  

![PyPI](https://img.shields.io/pypi/v/pistarlab) 
![PyPI - License](https://img.shields.io/pypi/l/pistarlab) 
[![Documentation Status](https://readthedocs.org/projects/pistarlab/badge/?version=latest)](https://pistarlab.readthedocs.io/en/latest/?badge=latest)

WARNING: This is an <u>**early release**</u> and is not ready for public use.

# Overview

piSTAR Lab is a modular deep reinforcement learning platform built to make AI development accessible and fun.

**Documentation** https://pistarlab.readthedocs.io

## Features
* Web UI
* Extension System for adding new agents, environments or tasks types
* Python API, anthing you can do in the UI, you can do in Python as well
* Run agents in single and multi player environments
* Experiment tracking
* Uses Ray Project (https://ray.io/) under the hood for distributed processing
* More to come


## UI Screenshots

<br/> <img src="docs/source/images/pistarlab_agent_home.png" alt="agent home" width="600"/>  <br/>

<br/> <img src="docs/source/images/pistarlab_envs.png" alt="agent home" width="600"/>  <br/>

<br/> <img src="docs/source/images/pistarlab_newtask.png" alt="agent home" width="600"/>  <br/>

<br/> <img src="docs/source/images/pistarlab_session_stats.png" alt="agent home" width="600"/>  <br/>

<br/> <img src="docs/source/images/pistarlab_agent_stats.png" alt="agent home" width="600"/>  <br/>



# Quick Start 
Detailed documentation is published at https://pistarlab.readthedocs.io

**Notes**
* Only tested on **Ubuntu**, but should also work on **OS X**. **MS Windows** users see [Installation using Docker](#Installation-using-Docker)
* Suggest using Anaconda or Miniconda for Python installation (visit https://www.anaconda.com/products/individual for instructions)
* Requires pip and python 3.7 or 3.8

## Installation 

### from pypi

```bash
pip install -U "pistarlab[all]"
```

### from repo 

```bash
git clone  --single-branch --depth=1 http://github.com/pistarlab/pistarlab/
pip install -e ."[all]"
```


## Usage


To launch piSTAR Lab UI, run:
```bash
pistarlab_launcher
```

Open browser to: http://localhost:7777


# Contributing

We are still in an early phase of this release but if you are interested in contributing to piSTAR Lab, please reach out.