import logging

PLUGIN_ID = "pistarlab-multigrid"

def install():
    
    # Original: https://github.com/ArnaudFickinger/gym-multigrid
    logging.info("Installing")
    import gym_multigrid
    from pistarlab.utils.env_helpers import add_gym_envs_from_registry    
    add_gym_envs_from_registry(
        plugin_id=PLUGIN_ID,
        entry_point_prefix="gym_multigrid.envs",
        overwrite=False,
        max_count = 300,
        default_wrappers=[
            {'entry_point':"gym_minigrid.wrappers:ImgObsWrapper",'kwargs':{}}
        ])

    return True

def load():
    import gym_multigrid
    logging.info("Importing gym_minigrid")
    return True

def uninstall():
    logging.info("Uninstalling minigrid")
    from pistarlab.dbmodels import EnvSpec
    from pistarlab import ctx
    ctx.disable_plugin_by_id(PLUGIN_ID)

    return True