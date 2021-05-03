import logging
from pistarlab import ctx

PLUGIN_ID = "pistarlab-envs-gym-atari"
PLUGIN_VERSION = "0.0.1-dev"

from pistarlab.utils.gym_importer import get_env_specs_from_gym_registry

def manifest():
    spec_list = get_env_specs_from_gym_registry(
        entry_point_prefix=f"gym.envs.atari",
        max_count = 600,
        additional_categories=['atari'])
    return {'env_specs': spec_list}

def install():
    ctx.install_plugin_from_manifest(PLUGIN_ID,PLUGIN_VERSION)
    return True

def load():
    return True

def uninstall():
    logging.info("Uninstalling {}".format(PLUGIN_ID))
    ctx.disable_plugin_by_id(PLUGIN_ID)
    return True