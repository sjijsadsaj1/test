import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class TspServiceTests(unittest.TestCase):
    def test_tour_validation(self):
        spec = importlib.util.spec_from_file_location('tsp_regression', ROOT / 'scripts/test_tsp_recovery.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            binary = Path(folder) / 'test'
            module.compile_test(binary)
            subprocess.run([str(binary), '--parse-only'], check=True)
