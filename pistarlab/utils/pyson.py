import importlib
import numpy as np


def var(name):
    return {'__type': "var", "name": name}


def pyson(data, var_lookup={}, depth=0):
    """
    Converts JSON compatable dictionaries into code using a simple syntax.

    returns an object reference with all internal components intialized
    TODO: could add loops
    """
    if data is None or isinstance(data, (int, float, str, bool, np.float32, np.float64, np.integer)):
        return data
    if data is None or isinstance(data, (list)):
        return [pyson(v, var_lookup, depth + 1) for v in data]
    if data is None or isinstance(data, (tuple)):
        return tuple([pyson(v, var_lookup, depth + 1) for v in data])
    elif type(data) is dict and "__type" in data and data["__type"] == 'tuple':
        return tuple([pyson(v, var_lookup, depth + 1) for v in data['values']])
    elif type(data) is dict and "__type" in data and data["__type"] == 'var':
        var_name = data['name']
        var_name_parts = var_name.split(".")
        # Should be recursive for deeply nested attributes
        total_parts = len(var_name_parts)
        if total_parts > 1:
            var_parent = var_lookup[var_name_parts[0]]
            return getattr(var_parent, var_name_parts[1])
        else:
            return var_lookup[var_name]
    elif type(data) is dict and "__type" in data and data["__type"] == 'python':
        return eval(data['code'])
    elif type(data) is dict and "__type" in data and data["__type"] == 'class':
        var_name = data.get('var_name', data['name'].lower())
        module_name = data['module']
        name = data['name']
        module = importlib.import_module(module_name)
        result = getattr(module, name)
        var_lookup[var_name] = result
        return result
    elif type(data) is dict and (("__type" in data and (data["__type"] == 'function_call' or data["__type"] == 'class_instance'))
                                 or ('module' in data and ('class_name' in data)) or 'func_name' in data):
        module_name = data.get('module')
        func_name = data.get('func_name')
        class_name = data.get('class_name')
        args = [pyson(v, var_lookup, depth + 1) for v in data.get('args', [])]
        kwargs = {k: pyson(v, var_lookup, depth + 1) for k, v in data.get('kwargs', {}).items()}
        if module_name:
            callable_ = importlib.import_module(module_name)

        if class_name:
            callable_ = getattr(callable_, class_name)

        if func_name:
            #TODO Could make more flexible
            func_name_parts = func_name.split(".")
            if len(func_name_parts) > 1:
                callable_ = var_lookup[func_name_parts[0]]
                func_name = func_name_parts[1]
            callable_ = getattr(callable_, func_name)
        var_name = data.get('var_name', "_".join([v for v in [class_name, func_name] if v]))
        result = callable_(*args, **kwargs)
        var_lookup[var_name] = result
        return result
    elif isinstance(data, (dict)):

        results = {}
        for k, v in data.items():
            output = pyson(v, var_lookup, depth + 1)
            var_lookup[k] = output
            results[k] = output
        return results

    else:
        raise Exception("Unparsable at {} at {} with type = {}".format(data, depth, type(data)))
