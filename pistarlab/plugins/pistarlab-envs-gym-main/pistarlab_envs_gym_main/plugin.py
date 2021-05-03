import logging
from pistarlab import ctx

PLUGIN_ID = "pistarlab-envs-gym-main"
PLUGIN_VERSION = "0.0.1-dev"

from pistarlab.utils.gym_importer import get_env_specs_from_gym_registry

def manifest():
    env_spec_list = []
    for collection in ['classic_control','box2d']:
        spec_list = get_env_specs_from_gym_registry(
            entry_point_prefix=f"gym.envs.{collection}",
            additional_categories=[collection]
        )
        env_spec_list.extend(spec_list)

    return {'env_specs': env_spec_list}

def install():
    ctx.install_plugin_from_manifest(PLUGIN_ID,PLUGIN_VERSION)
    return True

def load():
    return True

def uninstall():
    logging.info("Uninstalling {}".format(PLUGIN_ID))
    ctx.disable_plugin_by_id(PLUGIN_ID)
    return True