import os
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RosNativeEnvironmentTests(unittest.TestCase):
    def test_native_libraries_precede_inherited_conda_paths(self):
        env = dict(os.environ, LD_LIBRARY_PATH='/tmp/test-conda/lib',
                   APEXNAV_ROS_WORKSPACE='/tmp/test-planner')
        result = subprocess.check_output(['bash', str(ROOT / 'scripts/ros_native_env.sh'),
                                          '/usr/bin/printenv', 'LD_LIBRARY_PATH'], env=env, text=True)
        paths = result.strip().split(':')
        self.assertEqual(paths[0], '/tmp/test-planner/devel/lib')
        self.assertLess(paths.index('/usr/lib/x86_64-linux-gnu'), paths.index('/tmp/test-conda/lib'))
        self.assertEqual(env['LD_LIBRARY_PATH'], '/tmp/test-conda/lib')

    def test_command_required(self):
        result = subprocess.run(['bash', str(ROOT / 'scripts/ros_native_env.sh')], capture_output=True)
        self.assertEqual(result.returncode, 2)
