import time
import logging
import json
from pistarlab import ctx
import socket as ssocket
import zmq
import msgpack
import msgpack_numpy as m
m.patch()
context = zmq.Context()


class AbortRequestedException(Exception):
    pass


class ServerConn:

    def __init__(self, task_id, client_ids, timeout=5000, timeout_abort_check_callback=lambda: False) -> None:
        self.redis_client = ctx.get_redis_client()
        self.client_ids = client_ids
        self.client_ports = {}
        self.client_sockets = {}
        self.task_id = task_id
        self.timeout = timeout
        self.timeout_abort_check_callback = timeout_abort_check_callback

        for client_id in client_ids:
            socket = context.socket(zmq.PUSH)
            port = socket.bind_to_random_port(f"tcp://*", min_port=49152, max_port=65536, max_tries=100)
            self.client_sockets[client_id] = socket
            self.client_ports[client_id] = port

        self.in_socket = context.socket(zmq.PULL)
        self.in_port = self.in_socket.bind_to_random_port(f"tcp://*", min_port=49152, max_port=65536, max_tries=100)
        self.publish_conn()

    def publish_conn(self):
        hostname = ssocket.gethostname()
        ip_addr = ssocket.gethostbyname(hostname)
        config = {
            'hostname': hostname,
            'ip_addr': ip_addr,
            'to_server_port': self.in_port,
            'client_ports': self.client_ports
        }
        self.redis_client.set(f"configs_for_server_{self.task_id}", json.dumps(config))

    def put(self, key, value):
        while True:
            if self.client_sockets[key].poll(self.timeout, flags=zmq.POLLOUT):
                self.client_sockets[key].send(m.packb(value))
                return
            else:
                logging.info("Server send timeout reached")
                if self.timeout_abort_check_callback():
                    raise AbortRequestedException("Abort after server timeout reached")

    def get(self):
        while True:
            if self.in_socket.poll(self.timeout):
                msg = self.in_socket.recv()
                return m.unpackb(msg)
            else:
                logging.info("Server recv timeout reached")
                if self.timeout_abort_check_callback():
                    raise AbortRequestedException("Abort after server timeout reached")

    def get_all(self):
        remaining = []
        done = False
        while done:
            if self.in_socket.poll(self.timeout):
                msg = self.in_socket.recv()
                remaining.append(m.unpackb(msg))
            else:
                done = True
        return remaining

    def close(self):
        for socket in self.client_sockets.values():
            socket.close()
        self.in_socket.close()


class ClientConn:

    def __init__(self, task_id, client_id, timeout=5000, timeout_abort_check_callback=lambda: False) -> None:
        self.redis_client = ctx.get_redis_client()
        self.client_id = client_id
        self.task_id = task_id
        self.timeout = timeout
        self.timeout_abort_check_callback = timeout_abort_check_callback

        self.conn_info = self.get_conn_info()
        self.server_ip = self.conn_info['ip_addr']

        self.in_port = self.conn_info['client_ports'][self.client_id]
        self.out_port = self.conn_info['to_server_port']

        self.in_socket = context.socket(zmq.PULL)
        self.in_socket.connect(f"tcp://{self.server_ip}:{self.in_port}")
        self.out_socket = context.socket(zmq.PUSH)
        self.out_socket.connect(f"tcp://{self.server_ip}:{self.out_port}")

    def get_conn_info(self):
        data = self.redis_client.get(f"configs_for_server_{self.task_id}")
        return json.loads(data.decode())

    def put(self, value):
        while True:
            if self.out_socket.poll(self.timeout, flags=zmq.POLLOUT):
                self.out_socket.send(m.packb(value))
                return
            else:
                logging.info("Client send timeout reached")
                if self.timeout_abort_check_callback():
                    raise AbortRequestedException("Abort after client timeout reached")

    def get(self):
        while True:
            if self.in_socket.poll(self.timeout):
                msg = self.in_socket.recv()
                return m.unpackb(msg)
            else:
                logging.info("Client recv timeout reached")
                if self.timeout_abort_check_callback():
                    raise AbortRequestedException("Abort after client timeout reached")

    def close(self):
        self.out_socket.close()
        self.in_socket.close()
