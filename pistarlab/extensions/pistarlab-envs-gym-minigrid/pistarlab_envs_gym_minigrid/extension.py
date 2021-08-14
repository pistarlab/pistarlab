import logging
from pistarlab import ctx

EXTENSION_ID = "pistarlab-envs-gym-minigrid"
EXTENSION_VERSION = "0.0.1-dev"

from pistarlab.utils.gym_importer import get_env_specs_from_gym_registry

def manifest():
    import gym_minigrid
    spec_list = get_env_specs_from_gym_registry(
        entry_point_prefix=f"gym_minigrid.envs",
        max_count = 300,
        default_wrappers=[{'entry_point':"gym_minigrid.wrappers:ImgObsWrapper",'kwargs':{}}])
    return {'env_specs': spec_list}

def install():
    ctx.install_extension_from_manifest(EXTENSION_ID,EXTENSION_VERSION)
    return True

def load():
    import gym_minigrid
    return True

def uninstall():
    logging.info("Uninstalling {}".format(EXTENSION_ID))
    ctx.disable_extension_by_id(EXTENSION_ID)
    return True