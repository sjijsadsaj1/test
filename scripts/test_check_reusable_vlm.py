#!/usr/bin/env python3
"""Mock-only reuse safety tests; no real /proc, sockets, inference or signals."""
import importlib.util
import base64
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location('reuse_check', Path(__file__).with_name('check_reusable_vlm.py'))
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class FakePath:
    files = {}
    directories = {}
    resolutions = {}

    def __init__(self, path): self.path = str(path)
    def __truediv__(self, item): return FakePath(self.path + '/' + str(item))
    @property
    def name(self): return self.path.rsplit('/', 1)[-1]
    def read_text(self): return self.read_bytes().decode()
    def read_bytes(self): return self.files[self.path]
    def iterdir(self): return iter(FakePath(p) for p in self.directories.get(self.path, []))
    def resolve(self): return self.resolutions.get(self.path, self)
    def __str__(self): return self.path
    def __eq__(self, other): return isinstance(other, FakePath) and self.path == other.path


class SafetyTests(unittest.TestCase):
    def setUp(self):
        FakePath.files = {'/proc/net/tcp6': b'header\n'}
        FakePath.directories = {'/proc': []}
        FakePath.resolutions = {}
        self.links = {}
        rows = []
        for i, (port, module, route) in enumerate(checker.SERVICES, 1):
            pid = 1000 + i
            rows.append(f'0: 0100007F:{port:04X} 00000000:0000 0A 0 0 0 0 0 {i}')
            proc = f'/proc/{pid}'
            FakePath.directories['/proc'].append(proc)
            FakePath.files[proc + '/cmdline'] = f'python\0-u\0-m\0{module}\0--port\0{port}\0'.encode()
            FakePath.files[proc + '/environ'] = b''
            FakePath.directories[proc + '/fd'] = [proc + '/fd/7']
            self.links[proc + '/fd/7'] = f'socket:[{i}]'
            FakePath.resolutions[proc + '/cwd'] = FakePath('/project')
            FakePath.files['/project/' + module.replace('.', '/') + '.py'] = b'mock source'
        FakePath.files['/proc/net/tcp'] = ('header\n' + '\n'.join(rows) + '\n').encode()

    def invoke(self):
        response = Mock(status=200, headers={'Allow': 'POST, OPTIONS'})
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        connection = Mock()
        connection.__enter__ = Mock(return_value=connection)
        connection.__exit__ = Mock(return_value=False)
        with patch.object(checker, 'Path', FakePath), \
             patch.object(checker.os, 'readlink', side_effect=lambda p: self.links[str(p)]), \
             patch.object(checker.socket, 'create_connection', return_value=connection), \
             patch.object(checker.urllib.request, 'urlopen', return_value=response), \
             patch.object(checker.os, 'getpid', return_value=9999):
            return checker.check(FakePath('/project'))

    def test_missing_owner_refused(self):
        FakePath.directories['/proc/1001/fd'] = []
        with self.assertRaisesRegex(RuntimeError, 'Cannot prove unique'):
            self.invoke()

    def test_active_evaluation_refused(self):
        FakePath.directories['/proc'].append('/proc/2000')
        FakePath.files['/proc/2000/cmdline'] = b'python\0/project/scripts/evaluate_upstream.py\0'
        with self.assertRaisesRegex(RuntimeError, 'evaluation clients are active'):
            self.invoke()

    def test_correct_idle_services_accepted(self):
        result = self.invoke()
        self.assertEqual([s['pid'] for s in result], [1001, 1002, 1003, 1004])
        self.assertTrue(all(s['probe'] == 'OPTIONS, no inference' for s in result))

    def test_wrong_module_refused(self):
        FakePath.files['/proc/1001/cmdline'] = b'python\0unknown_server.py\0'
        with self.assertRaisesRegex(RuntimeError, 'expected default project model command'):
            self.invoke()

    def test_wrong_cwd_refused(self):
        FakePath.resolutions['/proc/1001/cwd'] = FakePath('/other_project')
        with self.assertRaisesRegex(RuntimeError, 'outside this project'):
            self.invoke()


class AuthorizedHttpTests(unittest.TestCase):
    def invoke(self, bad_blip=False, idle_error=None):
        seen = []
        def open_request(request, timeout):
            route = request.full_url.rsplit('/', 1)[-1]
            response = Mock(status=200, headers={'Allow': 'POST, OPTIONS'})
            response.__enter__ = Mock(return_value=response)
            response.__exit__ = Mock(return_value=False)
            if request.get_method() == 'POST':
                payload = json.loads(request.data)
                self.assertTrue(base64.b64decode(payload['image']).startswith(b'\x89PNG'))
                seen.append(route)
                data = {'boxes': [], 'logits': [], 'phrases': []}
                if route == 'blip2itm':
                    data = {'response': .1, 'itm score': 2 if bad_blip else .5}
                elif route == 'mobile_sam':
                    data = {'cropped_mask': base64.b64encode(bytes(4096)).decode()}
                response.read.return_value = json.dumps(data).encode()
            return response
        opener = Mock()
        opener.open.side_effect = open_request
        with patch.object(checker, 'check_idle_evaluators', side_effect=idle_error), \
             patch.object(checker.urllib.request, 'build_opener', return_value=opener):
            result = checker.check_authorized_http_services()
        self.assertEqual(seen, ['gdino', 'blip2itm', 'mobile_sam', 'yolov7'])
        return result

    def test_authorized_contract_passes_without_owner_claim(self):
        result = self.invoke()
        self.assertTrue(all(not x['owner_verified'] and not x['loaded_weights_verified'] for x in result))

    def test_invalid_numeric_contract_refused(self):
        with self.assertRaisesRegex(RuntimeError, 'Invalid BLIP'):
            self.invoke(bad_blip=True)

    def test_unknown_or_busy_client_still_refused(self):
        with self.assertRaisesRegex(RuntimeError, 'cannot assume idle'):
            self.invoke(idle_error=RuntimeError('Cannot inspect possible evaluation; cannot assume idle'))


if __name__ == '__main__':
    unittest.main()
