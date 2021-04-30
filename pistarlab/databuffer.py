from typing import Dict, Any
from typing import Tuple


class DataBuffer:
    """
    Databuffer with rollover
    """

    def __init__(self, cols, size):
        self.size = size
        self.entries = [None for i in range(size)]

        self.counter = 0
        self.cols = cols
        self.col_to_idx = {c: idx for idx, c in enumerate(cols)}

    def add(self, values: Tuple):
        self.entries[self.counter % self.size] = values
        self.counter += 1

    def add_dict(self, dict_data):
        self.add(tuple([dict_data[col] for col in self.cols]))

    def get(self, key, limit=None, limit_from_tail=False):
        col_idx = self.col_to_idx[key]

        if self.counter > self.size:
            idx = self.counter % self.size
            remainder = self.size - idx
            ordered_data = self.entries[-remainder:] + self.entries[:idx]
        else:
            ordered_data = self.entries[:self.counter]

        if limit:
            limit = min(limit, self.size)
            if limit_from_tail:
                result_list = ordered_data[-limit:]
            else:
                result_list = ordered_data[:limit]
        else:
            result_list = ordered_data

        return [v[col_idx] for v in result_list]

    def mean(self, key, limit=None, limit_from_tail=False):
        data = list(filter(None, self.get(key, limit, limit_from_tail)))
        data_len = len(data)
        result = 0
        if data_len > 0:
            result = sum(data) / data_len
        return result

    def sum(self, key, limit=None, limit_from_tail=False):
        data = list(filter(None, self.get(key, limit, limit_from_tail)))
        data_len = len(data)
        result = 0
        if data_len > 0:
            result = sum(data)
        return result

    def get_dict(self, cols=None, limit=None, limit_from_tail=False):
        if cols is None:
            cols = self.cols
        return {col: self.get(col, limit, limit_from_tail) for col in cols}

    def is_full(self):
        return self.size <= self.counter

    def clear(self):
        self.entries = [None for i in range(self.size)]
        self.counter = 0
