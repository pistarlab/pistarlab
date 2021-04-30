import logging
from pistarlab import ctx

PLUGIN_ID = "pistarlab-envs-gym-minigrid"
PLUGIN_VERSION = "0.0.1-dev"

from pistarlab.utils.gym_importer import get_env_specs_from_gym_registry

def manifest():
    import gym_minigrid
    spec_list = get_env_specs_from_gym_registry(
        entry_point_prefix=f"gym_minigrid.envs",
        max_count = 300,
        default_wrappers=[{'entry_point':"gym_minigrid.wrappers:ImgObsWrapper",'kwargs':{}}])
    return {'env_specs': spec_list}

def install():
    ctx.install_plugin_from_manifest(PLUGIN_ID,PLUGIN_VERSION)
    return True

def load():
    import gym_minigrid
    return True

def uninstall():
    logging.info("Uninstalling {}".format(PLUGIN_ID))
    ctx.disable_plugin_by_id(PLUGIN_ID)
    return True