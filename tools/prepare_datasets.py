"""Prepare fixed dataset bodies using only repository releases or offline archives."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset_bundle import prepare

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--archives', type=Path)
    parser.add_argument('--assets', type=Path)
    parser.add_argument('--datasets', default='mmlu,hle,frontier,swe,terminal')
    parser.add_argument('--context', action='store_true')
    a = parser.parse_args()
    prepare(a.cache, a.datasets.split(','), archives=a.archives, assets=a.assets, context=a.context)
