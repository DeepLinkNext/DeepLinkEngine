"""CPU selection of admitted original/reused measurements; no plotting imports."""
import copy
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('official_plot_selection', ROOT / 'official_plots.py')
plots = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plots)


class PlotSelectionTests(unittest.TestCase):
    def document(self, status='reused', value=952.42):
        return {'metrics': {'mmlu': {'output_tokens_per_second': value}},
                'phases': {'mmlu': {'status': status, 'receipt.json': {'value':
                    {'status': 'completed', 'completed': True, 'quiescent': True}}}}}

    def test_original_completed_measurement_unchanged(self):
        self.assertEqual(plots.measured_profile(self.document('completed'), 'mmlu'), 952.42)

    def test_admitted_reuse_visible_without_changing_status_or_value(self):
        value = self.document();before = copy.deepcopy(value)
        self.assertEqual(plots.measured_profile(value, 'mmlu'), 952.42)
        self.assertEqual(value, before)

    def test_reuse_needs_original_terminal_receipt(self):
        for missing in ('status', 'completed', 'quiescent'):
            value = self.document();del value['phases']['mmlu']['receipt.json']['value'][missing]
            self.assertIsNone(plots.measured_profile(value, 'mmlu'))
        value = self.document();del value['phases']['mmlu']['receipt.json']
        self.assertIsNone(plots.measured_profile(value, 'mmlu'))

    def test_failed_running_or_unattempted_not_plotted(self):
        for status in ('failed', 'running', 'planned', 'not_attempted'):
            self.assertIsNone(plots.measured_profile(self.document(status), 'mmlu'))

    def test_missing_invalid_or_nonpositive_value_not_plotted(self):
        for number in (None, -1, 0, float('nan'), float('inf')):
            self.assertIsNone(plots.measured_profile(self.document(value=number), 'mmlu'))
        value = self.document();value['metrics'] = {}
        self.assertIsNone(plots.measured_profile(value, 'mmlu'))


if __name__ == '__main__':
    unittest.main()
