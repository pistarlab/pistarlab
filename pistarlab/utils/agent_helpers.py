import logging


def get_agent_spec_interface_dict(
        interface_id='run',  # Same as function name in runner
        interface_type='rl',
        observation_space=None,
        action_space=None,
        auto_config_spaces=True):
    return {
        'interface_id': interface_id,
        'interface_type': interface_type,
        'auto_config_spaces': auto_config_spaces,
        'observation_space': observation_space,
        'action_space': action_space
    }


def get_agent_spec_dict(
        spec_id,
        runner_entry_point,
        entry_point=None,
        config={},
        components={},
        interfaces=None,
        collection=None,
        params={},
        algo_type_ids = [],
        disabled=False,
        displayed_name=None,
        version="0.0.1.dev0",
        description=None,
        usage = None):
    """
    Used to catpure spec info for registration info in serialized form
    NOTE: must be maintained with core.register_agent_spec method
    """

    data = {}
    data['spec_id'] = spec_id
    data['displayed_name'] = displayed_name or spec_id
    data['description'] = description
    data['entry_point'] = entry_point
    data['runner_entry_point'] = runner_entry_point
    data['version'] = version
    data['collection'] = collection
    data['disabled'] = disabled
    data['config'] = config
    data['config']['interfaces'] = interfaces or {'run': get_agent_spec_interface_dict()}
    data['config']['components'] = components
    data['params'] = params
    data['algo_type_ids'] = algo_type_ids
    data['usage'] = usage

    return data
