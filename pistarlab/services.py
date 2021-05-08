import logging
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from .meta import DEFAULT_REDIS_PASSWORD
from pathlib import Path
from typing import Dict

import psutil

from .config import get_sys_config


def get_home():
    return str(Path.home())


class SystemServiceBase:

    def __init__(self, name, log_root):
        self.name = name
        self.log_root = log_root
        self.log_path = os.path.join(self.log_root, "{}_run.log".format(self.name))
        self.pid_path = os.path.join(self.log_root, "{}_run.pid".format(self.name))

    def get_info(self):
        raise NotImplementedError()

    def get_state(self):
        raise NotImplementedError()

    def stop(self, block=True):
        raise NotImplementedError()

    def start(self, block=True):
        raise NotImplementedError()

    def restart(self, block=True):
        raise NotImplementedError()

    def get_log(self):
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()


class ReaderThread(threading.Thread):

    def __init__(self, proc, msg_queue, ready_string, log_path, log_stdout=True):
        threading.Thread.__init__(self)
        self.proc = proc
        self.msg_queue = msg_queue
        self.ready_string = ready_string.lower()
        self.log_path = log_path
        self.log_stdout = log_stdout

    def run(self):
        try:
            self.msg_queue.put("STARTED")
            with open(self.log_path, 'w') as f:
                for line in self.proc.stdout:
                    if self.ready_string in line.lower():
                        self.msg_queue.put("READY")
                    f.write(line)
                    if self.log_stdout:
                        logging.info(line.strip('\n'))
                    f.flush()
        except Exception as e:
            self.msg_queue.put("TERMINATED")
            logging.info("Thread exception {}".format(e))


class ForegroundService(SystemServiceBase):

    def __init__(
            self,
            name,
            launch_args,
            log_root,
            ready_string="is READY",
            pkill_string=None,
            stdin_pipe=None,
            links={},
            env={},
            log_stdout=True):
        super().__init__(name=name, log_root=log_root)
        self.launch_args = launch_args
        self.ready_string = ready_string
        self.pkill_string = pkill_string
        self.stdin_pipe = stdin_pipe
        self.proc = None
        self.msg_queue = queue.Queue()
        self.thread = None
        self.state = "CREATED"
        self.wait_retries = 1000
        self.wait_retries_delay = 0.1
        self.links = links
        self.msg = ""
        self.env = env
        self.log_stdout = log_stdout

    def get_info(self):
        return {'state': self.get_state(), 'links': self.get_links()}

    def get_state(self):
        if self.is_running():
            return self._get_state()
        else:
            return "TERMINATED"

    def _get_state(self):
        while not self.msg_queue.empty():
            self.state = self.msg_queue.get(block=False)
        return self.state

    def _block_until(self, target_states):
        retries = self.wait_retries
        while self.get_state() not in target_states:
            time.sleep(self.wait_retries_delay)
            retries -= 1
            if retries == 0:
                break
        if self.get_state() not in target_states:
            raise Exception(f"Processed failed to change to state {target_states} current state is {self.get_state()}")

    def is_running(self):
        if self.proc:

            self.proc.poll()
            return self.proc.returncode is None and self.thread.is_alive()
        else:
            return False

    def stop(self, block=True):
        self.state = "STOP_REQUESTED"
        if self.proc is not None:
            self.proc.terminate()
            self.proc = None
        if block:
            self._block_until(['TERMINATED', 'FAILED', 'CREATED'])
            self.kill_all()

    def restart(self, block=True):
        try:
            self.stop(block)
        except:
            self.kill_all()
        self.proc = None
        self.start(block)

    def start(self, block=True):
        if self.get_state() not in ['READY', 'STARTED']:
            self.msg = ""
            try:
                env = os.environ.copy()
                env.update(self.env)
                cmd_str = " ".join([str(v) for v in self.launch_args])
                print(f"RUNNING COMMAND: {cmd_str}")
                self.proc = subprocess.Popen(
                    self.launch_args,
                    stdout=subprocess.PIPE,
                    stdin=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    env=env)

                if self.stdin_pipe is not None:
                    for line in self.stdin_pipe.split("\n"):
                        self.proc.stdin.write(line + "\n")
                    self.proc.stdin.close()
                    # self.proc.wait()

                self.thread = ReaderThread(
                    proc=self.proc,
                    msg_queue=self.msg_queue,
                    ready_string=self.ready_string,
                    log_path=self.log_path,
                    log_stdout=self.log_stdout,
                )

                self.thread.start()
                with open(self.pid_path, "w") as f:
                    f.write("{}".format(self.proc.pid))
                if block:
                    self._block_until(['READY'])
            except Exception as e:
                logging.error(e)
                self.state = "TERMINATED"
                self.msg = "failed to start with {}".format(e)
        else:
            logging.debug("Already Started")

    def kill_all(self):
        if self.pkill_string is not None:
            try:
                for p in psutil.process_iter(attrs=['pid', 'cmdline']):
                    cmdline = " ".join(p.info['cmdline'])
                    if (self.pkill_string in cmdline) \
                            and ('pistarlab' in cmdline):
                        os.kill(p.pid, signal.SIGTERM)
            except:
                pass

    def get_pid(self):
        with open(self.pid_path, 'r') as f:
            return f.read()

    def get_log(self):
        with open(self.log_path, 'r') as f:
            return f.read()

    def get_links(self):
        return self.links

    def close(self):
        if self.proc:
            self.proc.kill()
        self.thread.join(3)
        self.kill_all()


class RayService(SystemServiceBase):

    def __init__(self,
                 log_root,
                 address,
                 redis_port=6379,
                 redis_password=None,
                 additional_args={},  # TODO: num-cpu, num-gpu etc
                 dashboard_port=8265,

                 links={}):
        super().__init__(name="ray", log_root=log_root)
        self.log_level = "DEBUG"
        # self.num_gpus = num_gpus
        # self.num_cpus = num_cpus
        self.redis_port = redis_port
        self.dashboard_port = dashboard_port
        self.state = "CREATED"
        self.redis_password = redis_password
        self.address = address
        self.head = (self.address == "localhost")

        if self.head:
            print("Starting ray head node")
            self.start_cmd = f"""ray --logging-level={self.log_level} start --head --port={self.redis_port} \
--include-dashboard=true --dashboard-port {self.dashboard_port}"""
            if self.redis_password is not None:
                 self.start_cmd = f"{self.start_cmd} --redis-password=\'{self.redis_password}\'"
            self.links = links
        else:
            self.start_cmd = f"""ray --logging-level={self.log_level} start --address='{self.address}' --redis-password=\'{self.redis_password}\'"""
            self.links = links

    def get_state(self):
        if self.is_running():
            return self.state
        else:
            return "TERMINATED"

    def get_info(self):
        return {'state': self.get_state(), 'links': self.get_links()}

    def get_links(self):
        return self.links

    def is_running(self):

        match_list = ["raylet","plasma_store_server", "gcs_server"]

        if not self.head:
            match_list = ['raylet']

        for match_str in match_list:
            if (len([p for p in psutil.process_iter(attrs=['pid', 'name']) if match_str in p.info['name'] and 'ray' in "".join(p.cmdline())]) > 0):
                return True

        return False

    def stop(self, block=True):
        self.state = "STOP_REQUESTED"
        done = False
        retries = 10

        with open(self.log_path, "a") as f:
            while not done:
                results = subprocess.check_output("ray stop", shell=True, text=True)
                f.write(results)
                time.sleep(0.5)
                if not self.is_running() or not block:
                    self.state = "TERMINATED"
                    done = True
                elif retries == 0:
                    raise Exception("Unable to stop ray")
                else:
                    time.sleep(0.8)
                    retries -= 1

    def kill_all(self):
        self.stop()

    def restart(self, block=True):
        self.stop(block)
        self.start(block)

    def start(self, block=True):
        if not self.is_running():
            self.state = "STARTED"
            print(f"RUNNING COMMAND: {self.start_cmd}")
            result = subprocess.check_output(self.start_cmd, shell=True, text=True)
            with open(self.log_path, "w") as f:
                f.write(self.start_cmd)
                f.write("\n\n")
                f.write(result)
            self.state = "READY"
        else:
            logging.info("Ray has already been started")

    def get_log(self):
        with open(self.log_path, 'r') as f:
            return f.read()

    def close(self):
        pass


class XVFB(SystemServiceBase):

    def __init__(self, log_root):
        super().__init__("xvfb", log_root)
        self.vdisplay = None
        self.state = "CREATED"

    def kill_all(self):
        try:
            for p in psutil.process_iter(attrs=['pid', 'name']):
                if 'xvfb' in p.info['name'].lower():
                    os.kill(p.pid, signal.SIGTERM)
        except:
            pass

    def get_info(self):
        return {'state': self.get_state(), 'links': {}}

    def get_state(self):
        return self.state

    def start(self, block=True):
        self.state = "STARTED"
        try:
            from xvfbwrapper import Xvfb

            self.vdisplay = Xvfb(width=400, height=400, colordepth=16)
            self.vdisplay.start()
            self.display = self.vdisplay.new_display
            self.state = "READY"
        except:
            self.state = "TERMINATED"

    def stop(self, block=True):
        try:
            if self.vdisplay:
                self.vdisplay.stop()
        except:
            pass
        self.kill_all()
        self.state = "TERMINATED"

    def restart(self, block=True):
        self.stop(block)
        self.start(block)

    def get_log(self):
        return "Current display = {}".format(self.display)

    def close(self):
        self.stop()

import pkg_resources
class ServiceContext:

    def __init__(self):

        self.config = get_sys_config()

        self.log_root = self.config.log_root
        self.services: Dict[str, SystemServiceBase] = {}

        self.auto_restart_services = ['backend']
        self.auto_restart_enabled = True
        self.monitor_thread = None
        self.verbose=True
        self.monitor_thread_interval_sec = 3
        self.commandline_args = {}

    def set_commandline_args(self, args):
        self.commandline_args = args

    def _service_monitor(self):
        while True:
            time.sleep(self.monitor_thread_interval_sec)
            if self.auto_restart_enabled:
                for name in self.auto_restart_services:
                    try:
                        service = self.get_service(name)
                        if service is not None and service.get_state() == "TERMINATED":
                            logging.info(f"{name} not running start")
                            service.restart(block=False)
                    except:
                        logging.info(sys.exc_info())

    def start_service_monitor_thread(self):
        logging.info("Starting service monitor thread")
        self.monitor_thread = threading.Thread(
            target=self._service_monitor, daemon=True)
        self.monitor_thread.start()

    def get_service_instance(self, name):
        if name == "xvfb":
            return XVFB(log_root=self.log_root)
        elif name == "ray":
            return RayService(
                log_root=self.log_root,
                dashboard_port=8265,
                redis_password=self.commandline_args.get('ray_redis_password'),
                address=self.commandline_args.get('ray_address'),
                links={'dashboard': "http://localhost:8265"})
        elif name == "redis":
            redis_password = self.commandline_args.get('redis_password')
            redis_config_path = os.path.join(self.config.root_path,"tmp_redis.config")           
            with open(redis_config_path,"w") as f:
                f.write(f'appendonly no\nsave ""\nrequirepass {redis_password}')
  
            redis_path = pkg_resources.resource_filename(__name__,"thirdparty_lib/redis-server")
            if os.name == 'nt':
                redis_path = f"{redis_path}.exe"
            
            launch_args = [redis_path,redis_config_path, '--port', self.commandline_args.get('redis_port')]

            return ForegroundService(
                name="redis",
                launch_args=launch_args,
                ready_string="ready to accept connections",
                log_root=self.log_root,
                pkill_string="pistarlab/thirdparty_lib/redis-server",
                links={},
                env={})
        elif name == "backend":
            return ForegroundService(
                name="backend",  # TODO: make debug and autoreload options (only important for development)
                launch_args=['python', '-u', '-m', 'pistarlab.app_backend', '--debug', '--autoreload'],
                ready_string="Backend is Ready",
                log_root=self.log_root,
                pkill_string="pistarlab.app_backend",
                links={'main': "http://localhost:7777"},
                env={'PYTHON_KEYRING_BACKEND': 'keyring.backends.null.Keyring'})
        elif name == "streamer":
            return ForegroundService(
                name="streamer",
                launch_args=['python', '-u', '-m', 'pistarlab.streamer','--port',self.commandline_args.get('streamer_port')],
                ready_string="Streamer is Ready",
                log_root=self.log_root,
                pkill_string="pistarlab.streamer",
                links={'main': "http://localhost:{}".format(self.commandline_args.get('streamer_port'))})
        elif name == "theia_ide":
            return ForegroundService(
                name="theia_ide",
                launch_args=['yarn', '--cwd', 'theia_ide/', 'start', self.config.workspace_path, '--hostname', '0.0.0.0', '--port', '7781'],
                ready_string="Deploy plugins list took",
                log_root=self.log_root,
                pkill_string="theia_ide",
                links={'main': "http://localhost:7781"},
                log_stdout=self.verbose)
        elif name == "dev_ui":
            service = ForegroundService(
                name="dev_ui",
                launch_args=['npm', 'run', 'serve', '--prefix', 'ui/', '--port=8080'],
                ready_string="App running at",
                log_root=self.log_root,
                pkill_string=None,
                links={'main': "http://localhost:8080"},
                log_stdout=self.verbose)
            return service
        else:
            raise Exception(f"Service with name {name} not found")

    def prep_services(self, service_list):
        for service_name in service_list:
            logging.info("Adding service {}".format(service_name))
            self.services[service_name] = self.get_service_instance(service_name)

    def start_all(self):
        try:
            for name, service in self.services.items():
                logging.info(f"Starting {name}")
                service.start()
        except Exception as e:
            logging.error(e)
            logging.error("Failed to start services. Shuttting down")
            self.clean_up()

    def clean_up(self):
        for service in self.services.values():
            service.stop()

    def toggle_auto_restart(self):
        self.auto_restart_enabled = not self.auto_restart_enabled

    def disable_auto_restart(self):
        self.auto_restart_enabled = False

    def get_auto_restart_info(self):
        return {
            'enabled': self.auto_restart_enabled,
            'services': self.auto_restart_services}

    def restart_all(self):
        for service in self.services.values():
            service.restart()

    def get_service(self, name):
        return self.services.get(name)

    def get_service_info(self):
        return {name: data.get_info() for name, data in self.services.items()}
