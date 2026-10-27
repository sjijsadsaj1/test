from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class TargetConfirmationTests(unittest.TestCase):
    def test_cpp_confirmation_and_exclusion_lifecycle(self):
        with tempfile.TemporaryDirectory(prefix='target-confirm-test-') as temp:
            binary = str(Path(temp) / 'test')
            subprocess.run(['g++', '-std=c++17', '-Wall', '-Wextra', '-Werror',
                '-I/usr/include/eigen3',
                '-I' + str(ROOT / 'src/planner/exploration_manager/include'),
                str(ROOT / 'tests/target_confirmation_test.cpp'), '-o', binary], check=True)
            subprocess.run([binary], check=True)



if __name__ == '__main__':
    unittest.main()
