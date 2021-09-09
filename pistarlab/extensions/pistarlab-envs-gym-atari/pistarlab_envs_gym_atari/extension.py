import logging
from pistarlab import ctx

from pistarlab.extension_tools import load_extension_meta
EXT_META = load_extension_meta(__name__)
EXTENSION_ID = EXT_META["id"]
EXTENSION_VERSION =  EXT_META["version"]

from pistarlab.utils.gym_importer import get_environments_from_gym_registry

def manifest():
    envs = get_environments_from_gym_registry(
        entry_point_prefix=f"gym.envs.atari",
        max_count = 600,
        additional_categories=['atari'],
        collection = "Gym Atari")    
    return {'environments': envs}

def install():
    ctx.install_extension_from_manifest(EXTENSION_ID,EXTENSION_VERSION)
    return True

def load():
    return True

def uninstall():
    logging.info("Uninstalling {}".format(EXTENSION_ID))
    ctx.disable_extension_by_id(EXTENSION_ID)
    return True