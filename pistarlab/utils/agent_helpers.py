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
        params={},
        algo_type_id = None,
        disabled=False,
        displayed_name=None,
        version="0.0.1.dev0",
        description=None):

    spec_data = {}
    spec_data['spec_id'] = spec_id
    spec_data['displayed_name'] = displayed_name or spec_id
    spec_data['description'] = description
    spec_data['entry_point'] = entry_point
    spec_data['runner_entry_point'] = runner_entry_point
    spec_data['version'] = version
    spec_data['disabled'] = disabled
    spec_data['config'] = config
    spec_data['config']['interfaces'] = interfaces or {'run': get_agent_spec_interface_dict()}
    spec_data['config']['components'] = components
    spec_data['params'] = params
    spec_data['algo_type_id'] = algo_type_id

    return spec_data
