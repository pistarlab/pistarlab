Extensions
=======

Extensions are the primary mechanism for adding new Agents, Environments, Tasks and Components.

Below are extensions available via piSTAR Lab and the repo (https://github.com/pistarlab/pistarlab-repo/).  Future versions will allow user created extensions to be used and shared as well.

* Gym Atari Environments: 
    * Extension: https://github.com/pistarlab/pistarlab/tree/main/pistarlab/extensions/pistarlab-envs-gym-atari
    * Project Page: https://gym.openai.com/envs/#atari
* Gym Text Environments: 
    * Extension: https://github.com/pistarlab/pistarlab/tree/main/pistarlab/extensions/pistarlab-envs-gym-text
    * Project Page: https://gym.openai.com/envs/#text
* Gym 2D Box and Classic Control Environments: 
    * Extension: https://github.com/pistarlab/pistarlab/tree/main/pistarlab/extensions/pistarlab-envs-gym-main
    * Project Pages: https://gym.openai.com/envs/#box2d, https://gym.openai.com/envs/#classic_control
* piSTAR Landia Environment
    * Extension: https://github.com/pistarlab/pistarlab-repo/tree/main/src/pistarlab-landia
    * Project Page: https://github.com/pistarlab/landia/
* Ray RLlib Agents
    * Extension: https://github.com/pistarlab/pistarlab/tree/main/pistarlab/extensions/pistarlab-rllib
    * Project Page: https://docs.ray.io/en/latest/rllib.html
    * Note: extension only includes a subset of agent algorithms
* MiniGrid Envirnoments:
    * Extension: https://github.com/pistarlab/pistarlab-repo/tree/main/src/pistarlab-envs-gym-minigrid
    * Project Page: https://github.com/maximecb/gym-minigrid
* Petting Zoo Environments (WIP)
    * Extension: https://github.com/pistarlab/pistarlab-repo/tree/main/src/pistarlab-petting-zoo
    * Project Page: https://www.pettingzoo.ml/
    * Note: extension only includes a small subset of environments
* Stable Baselines Agents (WIP)
    * Extension: https://github.com/pistarlab/pistarlab-repo/tree/main/src/pistarlab-stable-baselines
    * Project Page:
    * Note: extension only includes a small subset of agents available and currently does not support parameterization
* Unity ML-Agent Environments (Planned)
    * Extension: https://github.com/pistarlab/pistarlab-repo/tree/main/src/pistarlab-unity-envs
    * Project Page: https://github.com/Unity-Technologies/ml-agents
    * Note: currently unavailable in extension repo
* Minecraft RL Environments (Planned)
    * Extension: https://github.com/pistarlab/pistarlab-repo/tree/main/src/pistarlab-envs-minecraft
    * Project Page: https://github.com/minerllabs/minerl
    * Note: currently unavailable in extension repo
* MultiGrid MARL Environments (Planned)
    * Extension: https://github.com/pistarlab/pistarlab-repo/tree/main/src/pistarlab-multigrid
    * Project Page: https://github.com/ArnaudFickinger/gym-multigrid
    * Note: currently unavailable in extension repo




Development Notes
-----------------

Creating a manifest
~~~~~~~~~~~~~~~~~~~

Manfiest files are used to speed up the installation of Extensions. They are especially useful for Environments where probing is required.

**Example of creating a manifest**

.. code-block:: bash
   
    xvfb-run python pistarlab/extensions_tools.py --action=save_manifest --extension_path PATH_TO_EXTENSION/pistarlab-envs-gym-main

