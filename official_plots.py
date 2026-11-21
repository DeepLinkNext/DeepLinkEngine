"""Plot only actual measured profiles and raw reconstructed context windows."""
import json
import math
from pathlib import Path
import sys


def measured_profile(result, key):
    """Select existing measurements, retaining their original completion status.

    The campaign admits reused sources before reporting. A reused measurement
    also needs its original completed, quiescent arm receipt; plotting does not
    re-audit raw payloads or turn a reuse into a new measurement.
    """
    phase = result['phases'].get(key, {})
    if phase.get('status') == 'reused':
        receipt = phase.get('receipt.json', {}).get('value', {})
        if (receipt.get('status') != 'completed' or receipt.get('completed') is not True
                or receipt.get('quiescent') is not True):
            return None
    elif phase.get('status') != 'completed':
        return None
    value = result['metrics'].get(key, {}).get('output_tokens_per_second')
    return value if isinstance(value, (int, float)) and math.isfinite(value) and value > 0 else None


def main():
    import matplotlib
    if matplotlib.__version__ != '3.10.8':
        raise ValueError('Pinned CPU matplotlib 3.10.8 required')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    path = Path(sys.argv[1])
    result = json.loads(path.read_bytes())
    metrics = result['metrics']
    labels = [('mmlu', 'MMLU 200'), ('hle', 'HLE 200'), ('frontier', 'Frontier 160'),
              ('swe', 'SWE 64'), ('terminal', 'Terminal 48')]
    names, values = [], []
    for key, label in labels:
        value = measured_profile(result, key)
        if value is not None:
            names.append(label)
            values.append(value)
    if values:
        fig, axis = plt.subplots(figsize=(8, 4.5))
        bars = axis.bar(names, values, color='#237b86')
        axis.bar_label(bars, fmt='%.2f', padding=3)
        axis.set_ylabel('Output throughput (token/s)')
        axis.set_title(result['baseline'] + ' | Qwen3.5-35B-A3B BF16 / TP2')
        axis.set_ylim(0, max(values) * 1.2)
        fig.text(.5, .015, 'Complete fixed-profile measurements; judge time excluded. Unmeasured profiles have no bar.', ha='center', fontsize=8)
        fig.tight_layout(rect=(0, .04, 1, 1))
        fig.savefig(path.parent / 'fixed-five-throughput.svg', metadata={'Date': None})
        fig.savefig(path.parent / 'fixed-five-throughput.png', dpi=180)
        plt.close(fig)
    context = metrics.get('context', {})
    phases = [row for row in context.get('phases', []) if row.get('phase', '').endswith('-formal')]
    if phases:
        fig, axis = plt.subplots(figsize=(8, 4.5))
        for phase in phases:
            windows = phase.get('windows', [])
            points = [(index + 1, row.get('metrics', {}).get('output_tokens_per_second')) for index, row in enumerate(windows)]
            points = [(x, y) for x, y in points if isinstance(y, (int, float)) and math.isfinite(y)]
            if points:
                axis.plot([p[0] for p in points], [p[1] for p in points], marker='o', label=phase['phase'])
        axis.set_xlabel('Original 60-second window')
        axis.set_ylabel('Output throughput (token/s)')
        axis.set_title(result['baseline'] + ' | context and four-tenant stability')
        if axis.lines:
            axis.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(path.parent / 'context-tenant-curves.svg', metadata={'Date': None})
        fig.savefig(path.parent / 'context-tenant-curves.png', dpi=180)
        plt.close(fig)


if __name__ == '__main__':
    main()
