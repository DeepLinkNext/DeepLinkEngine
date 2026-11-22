"""Enter an unchanged SDK service child after preparing API logging."""
import json
from pathlib import Path
import runpy
import sys

import service_logging


def main():
    if len(sys.argv) < 5 or sys.argv[2] != '--':
        raise ValueError('Expected logging plan, --, SDK child and its arguments')
    plan = json.loads(Path(sys.argv[1]).read_bytes())
    target = Path(sys.argv[3]).resolve()
    if (service_logging.digest(__file__) != plan['entry_sha256']
            or str(target) != plan['child'] or not service_logging.inside(target, plan['sdk'])
            or service_logging.digest(target) != plan['child_sha256']):
        raise ValueError('SDK service child differs from logging plan')
    service_logging.install(plan)
    sys.argv = [str(target), *sys.argv[4:]]
    sys.path[0] = str(target.parent)
    runpy.run_path(str(target), run_name='__main__')


if __name__ == '__main__':
    main()
