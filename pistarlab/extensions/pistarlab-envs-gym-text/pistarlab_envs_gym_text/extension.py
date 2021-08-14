import logging
from pistarlab import ctx

EXTENSION_ID = "pistarlab-envs-gym-text"
EXTENSION_VERSION = "0.0.1-dev"

from pistarlab.utils.gym_importer import get_env_specs_from_gym_registry

def manifest():
    env_spec_list = []
    for collection in ['toy_text','unittest']:
        spec_list = get_env_specs_from_gym_registry(
            entry_point_prefix=f"gym.envs.{collection}",
            additional_categories=[collection]
        )
        env_spec_list.extend(spec_list)

    return {'env_specs': env_spec_list}

def install():
    ctx.install_extension_from_manifest(EXTENSION_ID,EXTENSION_VERSION)
    return True

def load():
    return True

def uninstall():
    logging.info("Uninstalling {}".format(EXTENSION_ID))
    ctx.disable_extension_by_id(EXTENSION_ID)
    return True