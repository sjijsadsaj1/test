"""Launcher environment regression: no ROS/model imports or services."""
import subprocess
import tempfile
import unittest
from pathlib import Path


class EvaluationInterpreterTest(unittest.TestCase):
    def test_fake_underlay_cannot_replace_saved_interpreter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / 'runtime'
            base = root / 'base'
            for prefix, marker in ((runtime, 'runtime'), (base, 'base')):
                (prefix / 'bin').mkdir(parents=True)
                executable = prefix / 'bin/python'
                executable.write_text('#!/bin/sh\nprintf "%s\\n" ' + marker + '\n')
                executable.chmod(0o755)
            setup = root / 'setup.bash'
            setup.write_text('export PATH="' + str(base) + '/bin:$PATH"\nexport PYTHONPATH=/fake/ros\n')
            result = subprocess.run(['bash', '-c', '''
set -eu
CONDA_PREFIX="$1"
readonly evaluation_prefix="$CONDA_PREFIX"
readonly evaluation_python="$evaluation_prefix/bin/python"
source "$2"
ros_pythonpath="${PYTHONPATH:-}"
export PATH="$evaluation_prefix/bin:$PATH"
export PYTHONPATH="/fake/runtime/site:${PYTHONPATH:-}"
"$evaluation_python"
python
printf '%s\n' "$ros_pythonpath" "$PYTHONPATH"
''', 'test', str(runtime), str(setup)], capture_output=True, text=True, check=True)
            self.assertEqual(result.stdout.splitlines(), ['runtime', 'runtime', '/fake/ros', '/fake/runtime/site:/fake/ros'])

    def test_launcher_has_no_unpinned_evaluation_command(self):
        source = Path(__file__).with_name('run_apexnav.sh').read_text()
        self.assertIn('readonly evaluation_python="$evaluation_prefix/bin/python"', source)
        self.assertIn('"$evaluation_python" -u habitat_evaluation.py', source)
        self.assertIn('eval_command=("$evaluation_python" -u', source)
        self.assertLess(source.index('import hydra, habitat, rospy'), source.index('start_background grounding_dino'))
        self.assertEqual(source.count('start_background ros env PATH="/usr/bin:$PATH" \\\n        PYTHONPATH="$ros_pythonpath"'), 2)


if __name__ == '__main__':
    unittest.main()
