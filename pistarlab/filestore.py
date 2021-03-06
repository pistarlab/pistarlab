import logging
import csv
import os
import pickle
import shutil
import time
from typing import Any, Dict
import copy
import numpy as np
import PIL.Image
import simplejson as json
from os.path import isfile, join
import threading
from multiprocessing import Queue
# from queue import Queue
from collections import defaultdict

from concurrent.futures import ThreadPoolExecutor
import concurrent
SUPPORTED_FILE_TYPES = ['png', 'jpg', 'pkl', 'json', 'csv']


class JSONEncoderDefault(json.JSONEncoder):

    def default(self, obj):  # pylint: disable=E0202
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            if obj.size() > 1000:
                return "REDACTED: NUMPY OBJ OF SIZE {} TOO LARGE".format(obj.size())
            else:
                return obj.tolist()
        else:
            try:
                return super(JSONEncoderDefault, self).default(obj)
            except Exception as e:
                return "ENCODE_FAILED:{}_AS_STR:{}".format(type(obj), obj)


class JSONDecoderDefault(json.JSONDecoder):

    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(
            self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):  # pylint: disable=E0202
        return obj


def format_key(key):
    if type(key) is str:
        return (key,)
    else:
        return tuple([str(k) for k in key])


def get_filetype(filepath_prefix):
    for ft in SUPPORTED_FILE_TYPES:
        if os.path.exists(filepath_prefix + "." + ft):
            return ft
    return None


def path_to_key_name_stype(root_path, full_file_path, sep="/"):
    if not root_path.endswith(sep):
        root_path = root_path + sep
    if len(root_path) > 1:
        full_file_path = full_file_path.replace(root_path, "", 1)
    if full_file_path.startswith(sep):
        full_file_path = full_file_path[1:]
    fpath, stype = os.path.splitext(full_file_path)
    path_parts = fpath.split(sep)
    name = path_parts[-1]
    path_parts = path_parts[:-1]
    key = format_key(path_parts)
    return key, name, stype[1:]


class FileStore:
    """
    Threaded filestorage class
    """

    def __init__(self,
                 root_path,
                 overwrite=False,
                 json_encoder=JSONEncoderDefault,
                 json_decoder=JSONDecoderDefault,
                 read_only=False,
                 use_write_thread=True,
                 use_pool_executor=False,
                 check_file_last_updated=True):  # this is a write optmization

        self.json_encoder = json_encoder
        self.json_decoder = json_decoder

        self.root_path = root_path

        if overwrite:
            logging.info("Overwrite enabled...")
            if os.path.exists(self.root_path):
                logging.info(
                    ".... Removing directory {}".format(self.root_path))
                shutil.rmtree(self.root_path)
            else:
                logging.info("No folder exists, not overwriting")

        if not os.path.exists(self.root_path):
            os.makedirs(self.root_path)
        self.read_only = read_only
        self.running = True
        if self.read_only:
            self.use_write_thread = False

        self.check_file_last_updated = check_file_last_updated
        self.data_store: Dict[str, Dict[str, Any]] = {}
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.use_write_thread = use_write_thread
        self.use_pool_executor = use_pool_executor
        self.futures = []

        if self.use_write_thread:

            self.write_queue = Queue()
            self.writer_thread = threading.Thread(target=self._process_write_requests, args=())
            self.writer_thread.daemon = True
            self.writer_thread.start()
        elif self.use_pool_executor:
            self.executor = ThreadPoolExecutor(max_workers=1)

        else:
            print("Not using write thread")

    def _process_write_requests(self):
        running = True

        t = threading.currentThread()
        counter = 0
        error_counter = 0
        while running:

            try:
                key, name, value, stype = self.write_queue.get(timeout=0.5)
                try:
                    self._write(key, value, name, stype)
                    counter += 1
                except Exception as e:
                    logging.error("Exception while writing for root_path:{}, key:{}, name: {} --- {}".format(self.root_path, key, name, e))
                    error_counter += 1

                if counter % 10000 == 0:
                    current_key = key or ""
                    logging.info("Writing item number {}, error count:{}, current_key:{}".format(counter, error_counter, current_key))

            except Exception as e:
                pass

            running = getattr(t, "running", True)
            if not running and not self.write_queue.empty():
                running = True

    def check_path(self, path):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    def update_dict(self, key, value, name='data', stype="json", clear_cache=False):
        key = format_key(key)
        value_dict = self.get(key, name, stype=stype)
        if value_dict is None:
            value_dict = {}
            value_dict.update(value)
            self.save(key, name=name, value=value_dict, stype=stype, clear_cache=clear_cache)
        else:
            value_dict.update(value)
            self._flush(key, name, clear_cache)

    def remove_items_from_dict(self, key, items, name='data', stype="json", clear_cache=False):
        key = format_key(key)

        value_dict = self.get(key, name, stype=stype)
        if value_dict is None:
            return
        for item in items:
            value_dict.pop(item, None)
        self._flush(key, name, clear_cache)

    def append_to_list(self, key, value, name='data', stype="json", clear_cache=False, flush=True):
        key = format_key(key)
        value_list = self.get(key, name, stype=stype)
        if value_list is None:
            value_list = []
            value_list.append(value)
            self.save(key, name=name, value=value_list, stype=stype, clear_cache=clear_cache)
        else:
            value_list.append(value)
            if flush:
                self._flush(key, name, clear_cache)

    def get_multipart_dict(self,
                           key,
                           name="data",
                           start_idx=0,
                           end_idx=None):

        manifest_name = "{}__{}".format(name, "manifest")

        manifest = self.get(key=key, name=manifest_name)

        if manifest is None:
            return None

        start_idx = start_idx
        end_idx = end_idx if end_idx else manifest['parts_index'] + 1

        value_list = {}
        for i in range(start_idx, end_idx):
            part_name = "{}__part{}".format(name, i)

            value_list_part = self.get(key=key, name=part_name, stype='json')
            if value_list_part is not None:
                for k, part_list, in value_list_part.items():
                    item_list = value_list.get(k, [])
                    item_list.extend(part_list)
                    value_list[k] = item_list
            else:
                print("no data found1")
        return value_list

    def get_multipart_list(self,
                           key,
                           name="data",
                           start_idx=0,
                           end_idx=None):

        manifest_name = "{}__{}".format(name, "manifest")

        manifest = self.get(key=key, name=manifest_name)

        if manifest is None:
            return None

        start_idx = start_idx
        end_idx = end_idx if end_idx else manifest['parts_index'] + 1

        value_list = []
        for i in range(start_idx, end_idx):
            part_name = "{}__part{}".format(name, i)

            value_list_part = self.get(key=key, name=part_name, stype='json')
            if value_list_part is not None:
                value_list.extend(value_list_part)
            else:
                print("no data found1")
        return value_list

    def append_to_multipart_dict(self,
                                 key,
                                 value: Any,
                                 name="data", stype="json", chunksize=500, col_names=None):

        if type(value) is np.ndarray:
            value = np.asscalar(value)
        manifest_name = "{}__{}".format(name, "manifest")
        if col_names is None:
            col_names = sorted(list(value.keys()))

        manifest = self.get(key, manifest_name, stype="json")
        if manifest is None:
            manifest = {}
            manifest['chunksize'] = chunksize
            manifest['cols'] = col_names
            manifest['parts_index'] = 0
            self.save(key=key, name=manifest_name, value=manifest)

        part_name = "{}__part{}".format(name, manifest['parts_index'])

        value_list = self.get(key=key, name=part_name)
        if value_list is None:
            value_list = defaultdict(list)
            self.save(key, name=part_name, value=value_list, stype=stype)

        length = None
        for col in col_names:
            value_list[col].append(value[col])
            if length is None:
                length = len(value_list[col])

        is_full = length >= manifest['chunksize']
        self.save(key, name=part_name, value=value_list, stype=stype, clear_cache=is_full)

        if is_full:
            manifest['parts_index'] += 1
            new_part_name = "{}__part{}".format(name, manifest['parts_index'])
            self.save(key=key, name=new_part_name, value={}, stype=stype)
            self.save(key=key, name=manifest_name, value=manifest)

    def extend_multipart_dict(self,
                              key,
                              value: Any,
                              name="data", stype="json", chunksize=500, col_names=None):

        if type(value) is np.ndarray:
            value = np.asscalar(value)
        manifest_name = "{}__{}".format(name, "manifest")
        if col_names is None:
            col_names = sorted(list(value.keys()))

        manifest = self.get(key, manifest_name, stype="json")
        if manifest is None:
            manifest = {}
            manifest['chunksize'] = chunksize
            manifest['cols'] = col_names
            manifest['parts_index'] = 0
            self.save(key=key, name=manifest_name, value=manifest)

        part_name = "{}__part{}".format(name, manifest['parts_index'])

        value_list = self.get(key=key, name=part_name)
        if value_list is None:
            value_list = defaultdict(list)
            self.save(key, name=part_name, value=value_list, stype=stype)

        length = None
        for col in col_names:
            list_data = value_list.get(col, [])
            list_data.extend(value[col])
            value_list[col] = list_data
            if length is None:
                length = len(value_list[col])

        is_full = length >= manifest['chunksize']
        self.save(key, name=part_name, value=value_list, stype=stype, clear_cache=is_full)

        if is_full:
            manifest['parts_index'] += 1
            new_part_name = "{}__part{}".format(name, manifest['parts_index'])
            self.save(key=key, name=new_part_name, value={}, stype=stype)
            self.save(key=key, name=manifest_name, value=manifest)

    def append_to_multipart_list(self,
                                 key,
                                 value: Any,
                                 name="data", stype="json", chunksize=5000):

        if type(value) is np.ndarray:
            value = np.asscalar(value)
        manifest_name = "{}__{}".format(name, "manifest")

        manifest = self.get(key, manifest_name, stype="json")
        if manifest is None:
            manifest = {}
            manifest['chunksize'] = chunksize
            manifest['parts_index'] = 0
            self.save(key=key, name=manifest_name, value=manifest)

        part_name = "{}__part{}".format(name, manifest['parts_index'])

        value_list = self.get(key=key, name=part_name)
        if value_list is None:
            value_list = []
            self.save(key, name=part_name, value=value_list, stype=stype)

        value_list.append(value)
        length = len(value_list)

        is_full = length >= manifest['chunksize']
        self.save(key, name=part_name, value=value_list, stype=stype, clear_cache=is_full)

        if is_full:
            manifest['parts_index'] += 1
            new_part_name = "{}__part{}".format(name, manifest['parts_index'])
            self.save(key=key, name=new_part_name, value={}, stype=stype)
            self.save(key=key, name=manifest_name, value=manifest)

    def save(self, key, value, name='data', stype="json", clear_cache=False, last_updated=None, flush=True):
        key = format_key(key)
        items = self.data_store.get(key, None)
        if items is None:
            items = {}
            self.data_store[key] = items
        items[name] = {
            'key': key,
            'value': value,
            'name': name,
            'last_updated': last_updated,  # file updated
            'stype': stype}
        if flush:
            self._flush(key, name, clear_cache)

    def list(self, key, use_cache=False):
        #TODO: add list cache
        key = format_key(key)
        names = []
        subkeys = []
        path = self.get_path_from_key(key)
        for fname in os.listdir(path):
            if isfile(os.path.join(path, fname)):
                name = os.path.splitext(fname)[0]
                stype = os.path.splitext(fname)[1]
                if name.endswith("_tmp"):
                    continue
                names.append(name)
            else:
                subkeys.append(fname)
        return names, subkeys

    def list_keys_stream(self, key, use_cache=False):
        #TODO: add list cache
        key = format_key(key)
        path = self.get_path_from_key(key)
        for fname in os.listdir(path):
            if not isfile(os.path.join(path, fname)):
                yield fname

    def list_objects_with_name_stream(self, key, name):
        key = format_key(key)
        for col in self.list_keys_stream(key):
            fullkey = key + (col,)
            obj = self.get(fullkey, name)
            if obj is not None:
                yield (col, obj)

    def list_objects_with_name(self, key, name):
        key = format_key(key)
        names, subkeys = self.list(key)
        print("Done listing")
        objects = []
        for col in subkeys:
            fullkey = key + (col,)
            obj = self.get(fullkey, name)
            if obj is not None:
                objects.append((col, obj))
        return objects

    def list_objects(self, key):
        key = format_key(key)

        names, subkeys = self.list(key)
        objects = []
        for name in names:
            obj = self.get(key, name)
            if obj is not None:
                objects.append((name, obj))
        return objects

    def get(self, key, name="data", stype=None, refresh=False):
        key = format_key(key)
        items = self.data_store.get(key, None)
        if self.check_file_last_updated:
            file_last_updated = self._file_last_updated(key, name=name, stype=stype)
        else:
            file_last_updated = None

        if not refresh and items is not None and name in items:
            entry = items.get(name)
            cached_last_updated = entry.get('last_updated', None)
            # Return data from cache if ...
            if file_last_updated is None or cached_last_updated is None or file_last_updated <= cached_last_updated:
                data = entry.get('value')
                if data is not None:
                    return data

        # read data from file
        data, stype = self._read(key, name, stype)
        if data is None:
            return None
        self.save(key, data,
                  name=name,
                  stype=stype,
                  last_updated=file_last_updated,
                  flush=False)
        return data

    def _add_future(self, fut):
        self.futures.append(fut)
        # if len(self.futures) > 100000:
        #     self.futures = [f for f in self.futures if not f.done()]

    def clean_futures(self):
        self.futures = [f for f in self.futures if not f.done()]

    def delete(self, key, name="data", stype=None):
        #TODO Fix slow deletes
        key = format_key(key)
        self.flush_all()
        items = self.data_store.get(key)
        path = self.get_path_from_key(key)

        #Remove file from disk
        if name is not None:
            items.pop(name, None)
            filepath_prefix = os.path.join(path, "{}".format(name))
            if stype is None:
                stype = get_filetype(filepath_prefix)

                #raise Exception("Not found")
            if stype is not None:
                filepath = os.path.join(path, "{}.{}".format(name, stype.lower()))
                if os.path.isfile(filepath):
                    os.remove(filepath)

        #Remove path
        if items is None or len(items) == 0:
            # Remove Path From memory
            self.data_store.pop(key, None)

            # If path is empty
            if len(os.listdir(path)) == 0:
                try:
                    if not os.path.isfile(path):
                        os.rmdir(path=os.path.join(os.getcwd(), path))
                except Exception as e:
                    print(e)
                if len(key) > 1:
                    self.delete(key[:-1], name=None)

    def prepvalue(self, value):
        return value

    def _flush(self, key, name='data', clear_cache=False):
        if self.read_only:
            return "not flushing {} {}".format(key, name)
        elif self.use_pool_executor:
            if self.read_only:
                return
            items = self.data_store[key]
            entry = items.get(name)
            if entry is not None:
                value = entry['value']
            if clear_cache:
                entry['value'] = None
            fut = self.executor.submit(self._write, key, self.prepvalue(value), name, entry['stype'])
            self._add_future(fut)
        elif self.use_write_thread:
            if self.read_only:
                return
            items = self.data_store[key]
            entry = items.get(name)
            if entry is not None:
                value = entry['value']
            if clear_cache:
                entry['value'] = None
            self.write_queue.put((key, name, self.prepvalue(value), entry['stype']))
        else:
            self._flush_sync(key, name, clear_cache)

    def delayed_write(self, key, name, value, stype):
        if self.use_write_thread:
            self.write_queue.put((key, name, self.prepvalue(value), stype))
        elif self.use_pool_executor:
            fut = self.executor.submit(self._write, key, name, self.prepvalue(value), stype)
            self._add_future(fut)
        else:
            raise Exception("Delayed write not supported with use_write_thread=False")

    def delayed_write_by_path(self, value, path):
        key, name, stype = path_to_key_name_stype(self.root_path, path)
        self.delayed_write(key, name, value, stype)

    def _flush_sync(self, key, name='data', clear_cache=False):
        if self.read_only:
            return
        items = self.data_store[key]
        entry = items.get(name)
        if entry is not None:
            value = entry['value']
            if clear_cache:
                entry['value'] = None
            self._write(key, name=name, value=value, stype=entry['stype'])

    # async def _flush_async(self, key, name='data', clear_cache=False):
    #     if self.read_only:
    #         return
    #     items = self.data_store[key]
    #     entry = items.get(name)
    #     if entry is not None:
    #         value = entry['value']
    #         if clear_cache:
    #             entry['value'] = None
    #         self._write(key, name=name, value=value, stype=entry['stype'])

    def get_path_from_key(self, key):
        if type(key) is tuple:
            path_parts = [str(k) for k in [self.root_path] + list(key)]
        else:
            path_parts = [str(k) for k in [self.root_path] + [key]]
        path = os.path.join(*path_parts)
        self.check_path(path)
        return path

    def close(self):
        if self.read_only:
            return
        if self.use_write_thread:
            while not self.write_queue.empty():
                time.sleep(0.1)
            self.writer_thread.running = False
            #self.writer_thread.terminate()
            self.writer_thread.join()
        elif self.use_pool_executor:

            self.executor.shutdown()
        else:
            return

    def flush_all(self):
        if self.read_only:
            return
        if self.use_write_thread:
            while not self.write_queue.empty():
                time.sleep(0.1)
        if self.use_pool_executor:
            concurrent.futures.wait(self.futures)
            self.clean_futures()
        else:
            return

    def _write(self, key, value, name='data', stype="json"):
        """
        saves value to file
        TODO: add autohandling of file type
        """
        if self.read_only:
            return

        path = self.get_path_from_key(key)
        filepath = os.path.join(path, "{}.{}".format(name, stype.lower()))
        filepath_tmp = os.path.join(
            path, "{}_tmp.{}".format(name, stype.lower()))
        try:
            if stype == "json":
                with open(filepath_tmp, 'w') as f:
                    json.dump(value, f, ignore_nan=True, cls=self.json_encoder)
            elif stype == "pkl":
                with open(filepath_tmp, 'wb') as f:
                    pickle.dump(value, f)
            elif stype == "png" or stype == "jpg":
                if type(value) == np.ndarray:
                    im = PIL.Image.fromarray(value)
                else:
                    im = value
                im.save(filepath_tmp)
            elif stype == "csv":
                with open(filepath_tmp, 'w') as f:
                    writer = csv.writer(f, delimiter='\t',
                                        quotechar='|',
                                        quoting=csv.QUOTE_MINIMAL)
                    writer.writerows(value)
            else:
                with open(filepath_tmp, 'w') as f:
                    f.write(value)
            shutil.copyfile(filepath_tmp, filepath)
            os.remove(filepath_tmp)
        except Exception as e:
            logging.error("Error key:{} name:{} type:{}".format(key, name, type(value)))
            raise e

    def _file_last_updated(self, key, name="data", stype=None):
        path = self.get_path_from_key(key)
        filepath_prefix = os.path.join(path, "{}".format(name))
        if stype is None:
            stype = get_filetype(filepath_prefix)
            if stype is None:
                return None
        filepath = filepath_prefix + "." + stype.lower()

        if os.path.isfile(filepath):
            return os.path.getmtime(filepath)
        else:
            return None

    def exists(self, key, name="data", stype=None):
        path = self.get_path_from_key(key)
        filepath_prefix = os.path.join(path, "{}".format(name))
        if stype is None:
            stype = get_filetype(filepath_prefix)

        if stype is None:
            return os.path.exists(filepath_prefix)
        else:
            fpath = filepath_prefix + "." + stype
            return os.path.exists(fpath)

    def _read(self, key, name="data", stype=None, default_value=None):
        path = self.get_path_from_key(key)
        filepath_prefix = os.path.join(path, "{}".format(name))
        if stype is None:
            stype = get_filetype(filepath_prefix)
            if stype is None:
                return None, None
        filepath = filepath_prefix + "." + stype.lower()
        try:
            if not os.path.isfile(filepath):
                return default_value, stype

            if stype.lower() == "json":
                with open(filepath, 'r') as f:
                    value = json.load(f, cls=self.json_decoder)
            elif stype == "pkl":
                with open(filepath, 'rb') as f:
                    value = pickle.load(f)
            elif stype == "csv":
                value = []
                with open(filepath, 'r') as f:
                    reader = csv.reader(f, delimiter='\t', quotechar='|')
                    for line in reader:
                        value.append(line)
            elif stype == "txt":
                value = []
                with open(filepath, 'r') as f:
                    value = f.readlines()
            else:
                raise Exception("Unsupported format {}".format(stype))
        except Exception as e:
            print("Error reading key:{}, name:{}, stype:{}".format(key, name, stype))
            print("Exception = {}".format(e))
            return None, None
        return value, stype
