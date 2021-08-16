import time
from uuid import uuid1
import platform
import os
import shortuuid
import hashlib
import time


def gen_shortuid():
    return shortuuid.uuid()


def gen_timestamp_str():
    current_time = time.time()
    millisecs = int((current_time - int(current_time)) * 1000)
    return time.strftime("%Y%m%d-%H%M%S-{}".format(millisecs))


def gen_uid(prefix=""):
    val = "%s_%s_%s" % (prefix, time.strftime("%Y%m%d-%H%M%S"), str(uuid1()))
    return val


def gen_instance_id():
    current_time = time.time()
    millisecs = int((current_time - int(current_time)) * 1000)
    val = "{}_{}_{}".format(time.strftime("%Y%m%d-%H%M%S-{}".format(millisecs)), platform.node(), os.getpid())
    return val


def get_timestamp_with_proc_info():
    current_time = time.time()
    millisecs = int((current_time - int(current_time)) * 1000)
    val = "{}_{}_{}".format(time.strftime("%Y%m%d-%H%M%S-{}".format(millisecs)), platform.node(), os.getpid())
    return val


def generate_seed(input: str, salt=None):
    if salt is None:
        salt = int(time.time())
    input_key = "{}_{}".format(input, salt)
    m = hashlib.sha256(input_key.encode())
    d = m.digest()
    seed = hex((int("0x{}".format(d.hex()), 0) % (2**32 - 1)))
    return seed

