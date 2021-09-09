import logging
from pistarlab import ctx

EXTENSION_ID = "pistarlab-envs-gym-text"
EXTENSION_VERSION = "0.0.1-dev"

from pistarlab.utils.gym_importer import get_environments_from_gym_registry

def manifest():
    envs_all = []
    for category in ['toy_text','unittest']:
        envs = get_environments_from_gym_registry(
            entry_point_prefix=f"gym.envs.{category}",
            additional_categories=[category],
            collection="Gym Test and Text")
        envs_all.extend(envs)
    return {'environments': envs_all}

def install():
    ctx.install_extension_from_manifest(EXTENSION_ID,EXTENSION_VERSION)
    return True

def load():
    return True

def uninstall():
    logging.info("Uninstalling {}".format(EXTENSION_ID))
    ctx.disable_extension_by_id(EXTENSION_ID)
    return True