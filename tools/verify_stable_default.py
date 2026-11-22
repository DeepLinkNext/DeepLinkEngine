"""Verify real default stable selection and both assessment configuration routes."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--download',action='store_true',help='Download all four installation inputs using this same selection')
    args=p.parse_args()
    from engine.release_selection import select,save,kit_path
    selected=select(args.cache)
    from packaging.version import Version
    if Version(selected['version'])<Version('2026.11.19.9'):
        raise ValueError('R9 stable publication must precede enabling this consumer default')
    from engine.engine_install import sources
    actual=sources()
    if actual!=[selected['sdist'],*selected['runtime_artifacts']]:
        raise ValueError('Default installation inputs differ from selected stable source')
    if args.download:
        from engine.engine_install import install
        install(args.cache,download_only=True)
    from one_click import verify_kit,configuration_arguments
    verify_kit()
    with tempfile.TemporaryDirectory(prefix='stable-client-contract-') as folder:
        folder=Path(folder)
        for mode in ('full','quick'):
            config=folder/(mode+'.json')
            config.write_text(json.dumps(dict(model=str(folder/'model'),deployment=str(folder/'installation/sdk'),
                physical_devices=[0,7],devices=[0,1],cache=str(args.cache),out=str(folder/mode),
                engine_profile=selected['engine_profile'],test_mode=mode)))
            for action in ('prepare','run'):
                arguments=configuration_arguments(action,['--config',str(config)])
                if arguments[arguments.index('--engine-profile')+1]!=selected['engine_profile']:
                    raise ValueError('Assessment profile differs from the selected stable engine')
                datasets=arguments[arguments.index('--datasets')+1]
                if datasets!=('mmlu,hle,frontier,swe,terminal' if mode=='full' else 'mmlu'):
                    raise ValueError('Stable selection changed the fixed assessment population')
    save(args.cache/'default-contract')
    print('DEFAULT_STABLE_VERIFIED',selected['version'],selected['release'],'4 pinned install inputs; prepare/run full/quick; no model loaded',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
