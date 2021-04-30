import numbers
import statistics
from collections import defaultdict
import copy
import importlib
from math import isnan, isinf

stats_funcs = {
    'mean': statistics.mean,
    'min': min,
    'max': max,
    'sum': sum
}


def transpose_dict_list(lst):
    d = defaultdict(list)
    for dct in lst:
        for k, v in dct.items():
            d[k].append(v)
    return d


def flatten_graphql(data):
    return [n['node'] for n in data['edges']]


def flatten_lists(lists):
    return [j for i in lists for j in i]


def get_stats(collected_data):
    s_data = {}
    for k, v in collected_data.items():
        sd = {}
        for s_name, s_func in stats_funcs.items():
            sd[s_name] = s_func(v)
        s_data[k] = sd
    return s_data


def get_stats_flat(collected_data, suffix='group'):
    s_data = {}
    for k, v in collected_data.items():
        for s_name, s_func in stats_funcs.items():
            s_data["{}__{}_{}".format(k, suffix, s_name)] = None if v is None else s_func(v)
    return s_data


def merged_dict(d1, d2, depth=0):
    if depth == 0:
        if d1 is None:
            d1 = {}
        d1 = copy.deepcopy(d1)
    if d2 is None:
        return d1
    for k, v in d2.items():
        if k in d1:
            if isinstance(v, dict) and isinstance(d1[k], dict):
                d1[k] = merged_dict(d1[k], v, depth + 1)
            else:
                d1[k] = v
        else:
            d1[k] = v
    return d1


def value_cleaner(v):
    if isinstance(v, numbers.Number):
        if isnan(v):
            return "NaN"
        elif isinf(v):
            return "Inf"
    return v


def clean_collection(d, clean_f=value_cleaner, inplace=False):
    target_d = d if inplace else {}
    items = None
    if isinstance(d, dict):
        items = d.items()
    elif isinstance(d, list):
        items = enumerate(d)
    else:
        raise Exception(f"Must be list or dict. Found {type(d)}")
    for k, v in items:
        if isinstance(v, dict) or isinstance(v, list):
            target_d[k] = clean_collection(d[k], inplace=inplace)
        else:
            target_d[k] = clean_f(v)
    return target_d


def sort_entity_by_id(data):
    try:
        uid = data.get('info', {}).get('uid')
        return int(uid.split('-')[1])
    except Exception as e:
        return 999999  # str(data)


def get_class_from_entry_point(entry_point):
    module_name, class_name = entry_point.split(":")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def cls_as_entry_point(cls):
    return "{}:{}".format(cls.__module__, cls.__name__)
