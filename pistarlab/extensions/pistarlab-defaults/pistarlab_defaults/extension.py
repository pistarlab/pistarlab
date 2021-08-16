from pistarlab.utils.agent_helpers import get_agent_spec_dict, get_agent_spec_interface_dict

import logging
from pistarlab import ctx

EXTENSION_ID = "pistarlab-defaults"
EXTENSION_VERSION = "0.0.1-dev"


def manifest():
    env_spec_list = []
    # for collection in ['algorithmic','classic_control','box2d','toy_text','unittest']:
    #     spec_list = get_env_specs_from_gym_registry(
    #         entry_point_prefix=f"gym.envs.{collection}",
    #         additional_categories=[collection]
    #     )
    #     env_spec_list.extend(spec_list)

    return {'env_specs': env_spec_list}


def install():
    agent_spec = get_agent_spec_dict(
        spec_id="defaults_a2c",
            entry_point='pistarlab_defaults.a2c:A2C',
            runner_entry_point='pistarlab_defaults.a2c:A2CTaskRunner',
            config={

            },
            components=[],
            interfaces={'run': get_agent_spec_interface_dict()}, # Not required, this is default if none
            params={},
            disabled=False,
            displayed_name="piSTARLab A2C",
            version="0.0.1-dev",
            description='')
    
    # ctx.install_extension_from_manifest(EXTENSION_ID,EXTENSION_VERSION)
    ctx.register_agent_spec(**agent_spec, extension_id = EXTENSION_ID,
                            extension_version = EXTENSION_VERSION)
    return True


def load():
    return True


def uninstall():
    logging.info("Uninstalling {}".format(EXTENSION_ID))
    ctx.disable_extension_by_id(EXTENSION_ID)
    return True
