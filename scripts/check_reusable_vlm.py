#!/usr/bin/env python3
"""Safe reuse checks; explicit HTTP mode probes inference, never loads models or signals processes."""
import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import struct
import urllib.request
import zlib

SERVICES = [(12181, 'vlm.detector.grounding_dino', 'gdino'),
            (12182, 'vlm.itm.blip2itm', 'blip2itm'),
            (12183, 'vlm.segmentor.sam', 'mobile_sam'),
            (12184, 'vlm.detector.yolov7', 'yolov7')]


def check_idle_evaluators():
    """Check visible evaluation clients; unreadable Python clients are unknown, not idle."""
    clients = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            args = (proc / 'cmdline').read_bytes().decode().split('\0')
        except (OSError, UnicodeError):
            try:
                comm = (proc / 'comm').read_text().strip()
            except FileNotFoundError:
                continue
            if comm.startswith(('python', 'habitat_eval')):
                raise RuntimeError(f'Cannot inspect possible evaluation PID {proc.name}; cannot assume idle')
            continue
        if any(Path(a).name in {'habitat_evaluation.py', 'evaluate_upstream.py'} for a in args):
            clients.append(int(proc.name))
    if clients:
        raise RuntimeError('Existing evaluation clients are active: ' + str(sorted(clients)))


def check_authorized_http_services():
    """Explicitly authorized deployment, no owner/loaded-weight identity claim."""
    check_idle_evaluators()
    def chunk(kind, data):
        return struct.pack('!I', len(data)) + kind + data + struct.pack('!I', zlib.crc32(kind + data))
    png = (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!2I5B', 64, 64, 8, 2, 0, 0, 0)) +
           chunk(b'IDAT', zlib.compress((b'\0' + b'\x80' * 192) * 64)) + chunk(b'IEND', b''))
    image = base64.b64encode(png).decode()
    params = {'gdino': dict(caption='chair .', box_threshold=.35, text_threshold=.25),
              'blip2itm': dict(txt='a chair'), 'mobile_sam': dict(bbox=[16, 16, 48, 48]),
              'yolov7': dict(agnostic_nms=True, conf_thres=.25, iou_thres=.45)}
    result = []
    # Avoid proxy environment accidentally redirecting loopback probes elsewhere.
    http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for port, module, route in SERVICES:
        url = f'http://127.0.0.1:{port}/{route}'
        with http.open(urllib.request.Request(url, method='OPTIONS'), timeout=5) as response:
            if response.status != 200 or 'POST' not in response.headers.get('Allow', ''):
                raise RuntimeError(f'Invalid HTTP interface: port {port} /{route}')
        check_idle_evaluators()
        body = json.dumps(dict(image=image, **params[route])).encode()
        print(f'  Contract probe: /{route}, one 64x64 image; no model loading', flush=True)
        with http.open(urllib.request.Request(url, data=body, headers={'Content-Type': 'application/json'}), timeout=60) as response:
            data = json.loads(response.read())
        if not isinstance(data, dict):
            raise RuntimeError(f'Invalid JSON response from /{route}')
        if route in ('gdino', 'yolov7'):
            boxes, scores, phrases = (data[k] for k in ('boxes', 'logits', 'phrases'))
            if not all(isinstance(v, list) for v in (boxes, scores, phrases)) or not len(boxes) == len(scores) == len(phrases):
                raise RuntimeError(f'Invalid detection schema from /{route}')
            if any(len(b) != 4 or not all(math.isfinite(float(v)) for v in b) for b in boxes):
                raise RuntimeError(f'Invalid detection boxes from /{route}')
            if not all(math.isfinite(float(v)) for v in scores) or not all(isinstance(v, str) for v in phrases):
                raise RuntimeError(f'Invalid detection confidence/classes from /{route}')
        elif route == 'blip2itm':
            cosine, itm = float(data['response']), float(data['itm score'])
            if not math.isfinite(cosine) or not math.isfinite(itm) or not 0 <= itm <= 1:
                raise RuntimeError('Invalid BLIP numeric output')
        elif len(base64.b64decode(data['cropped_mask'], validate=True)) != 64 * 64:
            raise RuntimeError('Invalid SAM mask byte count')
        result.append(dict(port=port, route=route, reused=True, mode='user-authorized-http-deployment',
                           probe='OPTIONS + one synthetic 64x64 inference',
                           owner_verified=False, loaded_weights_verified=False))
        print(f'  Contract OK: /{route}', flush=True)
    check_idle_evaluators()
    return result


def check(project):
    # Busy clients may share lockfiles/GPU and would be disturbed by concurrent
    # inference. This is an idle reuse facility, not concurrent evaluation.
    clients = []
    owners = {}
    inode_ports = {}
    for table in ('tcp', 'tcp6'):
        for line in Path('/proc/net/' + table).read_text().splitlines()[1:]:
            f = line.split()
            port = int(f[1].split(':')[-1], 16)
            if f[3] == '0A' and port in {x[0] for x in SERVICES}:
                inode_ports[f[9]] = port
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            args = (proc / 'cmdline').read_bytes().decode().split('\0')
        except (OSError, UnicodeError):
            continue
        if any(Path(a).name in {'habitat_evaluation.py', 'evaluate_upstream.py'} for a in args):
            clients.append(int(proc.name))
        try:
            for fd in (proc / 'fd').iterdir():
                try:
                    link = os.readlink(fd)
                except OSError:
                    continue
                if link.startswith('socket:[') and link[8:-1] in inode_ports:
                    owners.setdefault(inode_ports[link[8:-1]], set()).add(int(proc.name))
        except OSError:
            continue
    if clients:
        raise RuntimeError('Existing evaluation clients are active: ' + str(sorted(clients)) +
                           '. Wait for/stop their own launcher normally; no concurrent reuse.')
    result = []
    for port, module, route in SERVICES:
        with socket.create_connection(('127.0.0.1', port), timeout=2):
            pass
        pids = owners.get(port, set())
        if len(pids) != 1:
            raise RuntimeError(f'Cannot prove unique local process ownership for port {port}: {sorted(pids)}. '
                               'Run this check in the same host/PID namespace and under the service owner '
                               '(or a host administrator with /proc access). Container root or sudo inside '
                               'a restricted container may still lack that access; HTTP responsiveness alone '
                               'does not authorize reuse.')
        pid = next(iter(pids))
        proc = Path('/proc') / str(pid)
        args = (proc / 'cmdline').read_bytes().decode().split('\0')
        args = [a for a in args if a]
        expected = ['-u', '-m', module, '--port', str(port)]
        if args[1:] != expected or not Path(args[0]).name.startswith('python'):
            raise RuntimeError(f'Port {port} PID {pid} is not the expected default project model command')
        if (proc / 'cwd').resolve() != project:
            raise RuntimeError(f'Port {port} PID {pid} is outside this project working directory')
        environment = dict(item.split('=', 1) for item in
                           (proc / 'environ').read_bytes().decode().split('\0') if '=' in item)
        if module.endswith('.sam') and environment.get('MOBILE_SAM_CHECKPOINT', 'data/mobile_sam.pt') != os.environ.get('MOBILE_SAM_CHECKPOINT', 'data/mobile_sam.pt'):
            raise RuntimeError('MobileSAM checkpoint configuration mismatch')
        request = urllib.request.Request(f'http://127.0.0.1:{port}/{route}', method='OPTIONS')
        with urllib.request.urlopen(request, timeout=5) as response:
            if response.status != 200 or 'POST' not in response.headers.get('Allow', ''):
                raise RuntimeError(f'Port {port} does not expose the required POST /{route} interface')
        source = project / (module.replace('.', '/') + '.py')
        result.append(dict(port=port, pid=pid, module=module, cwd=str(project),
                           route=route, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                           probe='OPTIONS, no inference', reused=True))
        print(f'  Verified idle service: {module} PID {pid} port {port}', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--accept-existing-owner', action='store_true',
                        help='Explicitly authorize deployed HTTP services without /proc ownership proof; probes one image/model')
    args = parser.parse_args()
    try:
        services = check_authorized_http_services() if args.accept_existing_owner else check(args.project.resolve())
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        parser.exit(1, f'Refusing service reuse: {exc}\nNo existing processes have been stopped.\n')
    if args.output:
        args.output.write_text(json.dumps({'services': services,
            'limitation': 'Interface/schema probes do not validate semantic accuracy, process ownership in explicit HTTP mode, or loaded-weight hashes.'}, indent=2) + '\n')


if __name__ == '__main__':
    main()
