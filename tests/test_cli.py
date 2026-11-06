import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def run_cli(self, command, path):
        return subprocess.run([sys.executable, '-m', 'cs2_money_round', command, str(path)],
                              cwd=ROOT, text=True, capture_output=True)

    def test_examples_run_end_to_end(self):
        for command, filename, field in [('evaluate', 'full_buy.json', 'full_buy_possible'),
                                         ('simulate', 'pistol_loss.json', 'rounds'),
                                         ('estimate', 'scenarios.json', 'full_buy_probability')]:
            result = self.run_cli(command, ROOT / 'examples' / filename)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(field, json.loads(result.stdout))

    def test_bad_json_reports_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.json'
            path.write_text('{"scenarios": [NaN]}')
            result = self.run_cli('estimate', path)
        self.assertEqual(result.returncode, 2)
        self.assertIn('Invalid JSON constant', result.stderr)
        self.assertNotIn('Traceback', result.stderr)

    def test_replay_continues_into_explicit_new_half(self):
        data = json.loads((ROOT / 'examples/pistol_loss.json').read_text())
        second = json.loads(json.dumps(data['rounds'][0]))
        second['reset_half'] = True
        data['rounds'].append(second)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'replay.json'
            path.write_text(json.dumps(data))
            result = self.run_cli('simulate', path)
        self.assertEqual(result.returncode, 0, result.stderr)
        final = json.loads(result.stdout)['final_team']
        self.assertEqual(final['side'], 'CT')
        self.assertEqual(final['loss_level'], 0)
        self.assertTrue(all(p['cash'] == 3500 for p in final['players']))
