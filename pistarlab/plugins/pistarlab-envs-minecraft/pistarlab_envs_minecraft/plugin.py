import logging

def install():
    import minerl
    # https://github.com/minerllabs/minerl

    default_image = "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwww.rosememoriallibrary.org%2Fwp-content%2Fuploads%2F2015%2F03%2Fminecraft-logo.jpg&f=1&nofb=1"
    default_image ="https://images.sftcdn.net/images/t_app-logo-xl,f_auto/p/b276c62c-96bf-11e6-ad7b-00163ec9f5fa/2176013111/minecraft-logo.png"
    # add to index of environments
    
    from pistarlab.utils.env_helpers import add_gym_envs_from_registry    
    add_gym_envs_from_registry(entry_point_prefix="minerl.env",overwrite=False,probe_for_metadata=False)

    # import minerl
    # minerl.data.download(directory="/your/local/path/")
    return True

def load():
    logging.info("Loading")
    import minerl
    return True

def uninstall():
    logging.info("Uninstalling - yo")
    return True