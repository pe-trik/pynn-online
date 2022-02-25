import socket
import threading
import logging
import time
import select
import queue

import sys
root = logging.getLogger()
root.setLevel(logging.DEBUG if True else logging.INFO)

handler = logging.StreamHandler(sys.stderr)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)12s %(levelname)s %(filename)s:%(lineno)3d] %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

class Server(threading.Thread):
    def __init__(self, host, port, data_callback, reset_callback):
        threading.Thread.__init__(self, daemon=True)
        self.data_callback = data_callback
        self.reset_callback = reset_callback
        self.host = host
        self.port = port
        self.is_running = True
        self.hypothesis_queue = queue.Queue()

    def send_hypothesis(self, h:str):
        self.hypothesis_queue.put(h.encode())

    def run(self):
        while True:
            try:
                server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                server.setblocking(False)
                server.bind((self.host, self.port))
                server.listen()
                break
            except:
                logging.error(f'port {self.port} in use, retrying to connect...')
                time.sleep(1)

        while self.is_running:
            try:
                ready = select.select([server],[],[],1)
                logging.debug(f'waiting for a connection on port {self.port}')
                if not ready[0]:
                    continue
                conn, _ = server.accept()
                logging.debug(f'got connection on port {self.port}')
                conn.setblocking(False)
                self.reset_callback()
                no_data_start = None
                with conn:
                    while self.is_running:

                        while True:
                            try:
                                d = self.hypothesis_queue.get(block=False)
                                conn.sendall(d)
                            except queue.Empty:
                                break

                        ready = select.select([conn],[],[],1)
                        if not ready[0]:
                            continue
                        data = conn.recv(16)
                        if not data:
                            if no_data_start:
                                if time.time() - no_data_start > 1:
                                    conn.close()
                                    break
                            else:
                                no_data_start = time.time()
                            continue
                        self.data_callback(-1, data)
            except socket.error:
                logging.error(f'connection error on port {self.port}')
                self.reset_callback()
                continue