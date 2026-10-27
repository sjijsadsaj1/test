#!/usr/bin/env python3
"""Compile production TSP helper and exercise failures on an isolated ROS master."""
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def compile_test(binary):
    flags = subprocess.check_output(['pkg-config', '--cflags', '--libs', 'roscpp'], text=True).split()
    subprocess.run(['g++', '-std=c++17', '-pthread', '-Wall', '-Wextra',
                    '-I' + str(ROOT / 'src/planner/exploration_manager/include'),
                    '-I' + str(ROOT / 'devel/include'), str(ROOT / 'tests/tsp_service_test.cpp'),
                    '-o', str(binary)] + flags, check=True)


def main():
    port = 11442
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(('127.0.0.1', port))
    root = Path(tempfile.mkdtemp(prefix='tsp-recovery-'))
    print('Test artifacts:', root, flush=True)
    binary = root / 'tsp_test'
    compile_test(binary)
    env = dict(os.environ, ROS_MASTER_URI=f'http://127.0.0.1:{port}', ROS_IP='127.0.0.1',
               ROS_HOME=str(root / 'ros'), ROS_LOG_DIR=str(root / 'ros_logs'))
    env.pop('ROS_HOSTNAME', None)
    with (root / 'master.log').open('w') as log:
        master = subprocess.Popen(['roscore', '-p', str(port)], env=env,
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                if master.poll() is not None:
                    raise RuntimeError('isolated master exited')
                try:
                    with socket.create_connection(('127.0.0.1', port), timeout=0.2):
                        break
                except OSError:
                    time.sleep(0.1)
            else:
                raise TimeoutError('isolated master did not start')
            with (root / 'test.log').open('w') as out:
                subprocess.run([str(binary), str(root / 'atsp_tour.tour')], env=env,
                               stdout=out, stderr=subprocess.STDOUT, timeout=30, check=True)
            print((root / 'test.log').read_text(), flush=True)
        finally:
            if master.poll() is None:
                os.killpg(master.pid, signal.SIGINT)
                master.wait(timeout=15)


if __name__ == '__main__':
    main()
