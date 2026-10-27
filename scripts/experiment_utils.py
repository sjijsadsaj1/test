#!/usr/bin/env python3
"""Shared experiment bookkeeping; independent of historical experiment drivers."""
import argparse
import json
from pathlib import Path
import re
import socket

ROOT = Path(__file__).resolve().parents[1]

def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')

def key(episode):
    return episode['scene_key'], str(episode['episode_id'])

def metrics(folder):
    final = None
    for path in folder.rglob('record.txt'):
        for block in path.read_text().split('Scene ID: ')[1:]:
            index = int(re.search(r'No\.(\d+) task', block).group(1))
            if final is None or index > final[0]:
                final = index, block
    if final is None:
        return {}
    result = {}
    for name in ('Average Success', 'Average SPL', 'Average Soft SPL', 'Average Distance to Goal'):
        match = re.search(r'\|\s*' + re.escape(name) + r'\s*\|\s*([0-9.]+)', final[1])
        if match:
            result[name] = float(match.group(1))
    elapsed = re.search(r'([0-9.]+) seconds spend', final[1])
    if elapsed:
        result['evaluation_elapsed_seconds'] = float(elapsed.group(1))
    return result

def read_records(folder):
    rows = {}
    for path in folder.rglob('record.txt'):
        for block in path.read_text().split('Scene ID: ')[1:]:
            scene = Path(block.splitlines()[0].strip()).stem
            eid = re.search(r'Episode ID:\s*(\S+)', block).group(1)
            key = (scene, eid)
            if key in rows:
                raise ValueError('Duplicate task outcome: ' + repr(key))
            rows[key] = re.search(r'success or not:\s*([^\r\n]+)',block).group(1).strip()
    return rows

def choose_ports(batch):
    def first_available(start):
        for port in range(start, start + 100):
            try:
                with socket.socket() as probe:
                    probe.bind(('0.0.0.0', port))
                return port
            except OSError as exc:
                import errno
                if exc.errno == errno.EADDRINUSE:
                    continue
                raise
        raise RuntimeError('No free port in range ' + str(start) + '..' + str(start + 99))
    ports = dict(bridge_port=first_available(8092), ros_port=first_available(12191))
    write(batch / 'ports.json', ports)
    return ports

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['ports'])
    parser.add_argument('--batch', type=Path, required=True)
    args = parser.parse_args()
    ports = choose_ports(args.batch)
    print(ports['bridge_port'], ports['ros_port'])
