import os
import subprocess
import logging
from filelock import FileLock


def run_bash_command(cmd):
    logging.info("BASH COMMAND \n\t{}".format(cmd))
    env = os.environ.copy()
    env['PYTHON_KEYRING_BACKEND'] = 'keyring.backends.null.Keyring'
    return subprocess.check_output(
        cmd, shell=True, text=True, env=env
    )
