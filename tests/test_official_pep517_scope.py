from pathlib import Path
import base64, csv, hashlib, io, json, os, subprocess, sys, tarfile, time, zipfile, tempfile
from unittest.mock import patch

temporary=tempfile.TemporaryDirectory() if len(sys.argv)==1 else None
j=Path(temporary.name if temporary else sys.argv[1]);wheelhouse=j/'wheelhouse';wheelhouse.mkdir();traces=j/'trace.jsonl'
def make_wheel(name,version,module):
    path=wheelhouse/(name+'-'+version+'-py3-none-any.whl');dist=name+'-'+version+'.dist-info'
    files={name+'/__init__.py':module.encode(),dist+'/METADATA':('Metadata-Version: 2.1\nName: '+name+'\nVersion: '+version+'\n').encode(),dist+'/WHEEL':b'Wheel-Version: 1.0\nGenerator: real-pip-scope-fixture\nRoot-Is-Purelib: true\nTag: py3-none-any\n'}
    records=[]
    with zipfile.ZipFile(path,'w') as z:
        for name,body in files.items():z.writestr(name,body);records.append([name,'sha256='+base64.urlsafe_b64encode(hashlib.sha256(body).digest()).decode().rstrip('='),str(len(body))])
        records.append([dist+'/RECORD','','']);text=io.StringIO();csv.writer(text,lineterminator='\n').writerows(records);z.writestr(dist+'/RECORD',text.getvalue())
    return path
for version in ('1.0','2.0'):make_wheel('scope_build_dep',version,'VERSION='+repr(version)+'\n')

def source(name,version,requires,expected,nested=False):
    root=j/(name+'-source');root.mkdir()
    (root/'pyproject.toml').write_text('[build-system]\nrequires = '+json.dumps(requires)+'\nbuild-backend = "fixture_backend"\nbackend-path = ["."]\n')
    template=r'''from pathlib import Path
import os,json,zipfile,csv,io,hashlib,base64,sys
import scope_build_dep
NAME=__NAME__;VERSION=__VERSION__;EXPECTED=__EXPECTED__;NESTED=__NESTED__
def verify(hook):
    assert scope_build_dep.VERSION==EXPECTED,(NAME,scope_build_dep.VERSION,EXPECTED)
    if NESTED:
        import scope_nested
        assert scope_nested.BUILT_WITH=='1.0'
    with open(os.environ['SCOPE_FIXTURE_TRACE'],'a') as stream:
        stream.write(json.dumps(dict(project=NAME,hook=hook,dependency_version=scope_build_dep.VERSION,dependency_file=scope_build_dep.__file__,prefix=sys.prefix,base_prefix=sys.base_prefix,pid=os.getpid()))+'\n')
def get_requires_for_build_wheel(config_settings=None):verify('get_requires');return []
def prepare_metadata_for_build_wheel(metadata_directory,config_settings=None):
    verify('metadata');name=NAME+'-'+VERSION+'.dist-info';folder=Path(metadata_directory)/name;folder.mkdir(exist_ok=True);(folder/'METADATA').write_text('Metadata-Version: 2.1\nName: '+NAME+'\nVersion: '+VERSION+'\n');return name
def build_wheel(wheel_directory,config_settings=None,metadata_directory=None):
    verify('wheel');filename=NAME+'-'+VERSION+'-py3-none-any.whl';dist=NAME+'-'+VERSION+'.dist-info';files={NAME+'/__init__.py':('BUILT_WITH='+repr(EXPECTED)+'\n').encode(),dist+'/METADATA':('Metadata-Version: 2.1\nName: '+NAME+'\nVersion: '+VERSION+'\n').encode(),dist+'/WHEEL':b'Wheel-Version: 1.0\nGenerator: real-pip-scope-fixture\nRoot-Is-Purelib: true\nTag: py3-none-any\n'};records=[]
    with zipfile.ZipFile(Path(wheel_directory)/filename,'w') as z:
        for name,body in files.items():z.writestr(name,body);records.append([name,'sha256='+base64.urlsafe_b64encode(hashlib.sha256(body).digest()).decode().rstrip('='),str(len(body))])
        records.append([dist+'/RECORD','','']);text=io.StringIO();csv.writer(text,lineterminator='\n').writerows(records);z.writestr(dist+'/RECORD',text.getvalue())
    return filename
'''
    for key,value in {'__NAME__':repr(name),'__VERSION__':repr(version),'__EXPECTED__':repr(expected),'__NESTED__':repr(nested)}.items():template=template.replace(key,value)
    (root/'fixture_backend.py').write_text(template);archive=j/(name+'-'+version+'.tar.gz')
    with tarfile.open(archive,'w:gz') as t:
        for filename in ('pyproject.toml','fixture_backend.py'):t.add(root/filename,arcname=name+'-'+version+'/'+filename)
    return archive
child=source('scope_nested','0.1',['scope_build_dep==1.0'],'1.0')
parent=source('scope_parent','0.1',['scope_build_dep==2.0','scope_nested @ '+child.as_uri()],'2.0',True)
constraint=j/'root-constraint.txt';constraint.write_text('scope_build_dep==2.0\n')
# The production command deliberately uses -S. Expose only this test runner's pip
# package to its child, as a normal installed Python otherwise omits site-packages.
import pip
base=dict(os.environ)
base['PYTHONPATH']=str(Path(pip.__file__).resolve().parent.parent)
for key in list(base):
    if key.startswith('PIP_'):base.pop(key)
base.update(PIP_CONFIG_FILE=os.devnull,PIP_NO_INDEX='1',PIP_FIND_LINKS=str(wheelhouse),PIP_CACHE_DIR=str(j/'pip-cache'),PYTHONDONTWRITEBYTECODE='1',SCOPE_FIXTURE_TRACE=str(traces))
cases=[
 ('bad-environment-recursive',["install","--dry-run","--ignore-installed","--constraint",str(constraint),'scope_build_dep==2.0',str(child)],dict(PIP_BUILD_CONSTRAINT=str(constraint)),False),
 ('root-cli-constraint-metadata',["install","--dry-run","--ignore-installed","--report",str(j/'metadata-report.json'),"--constraint",str(constraint),'scope_build_dep==2.0',str(child)],{},True),
 ('cli-build-constraint-is-recursive',["wheel","--no-deps","--wheel-dir",str(j/'bad-built'),"--build-constraint",str(constraint),str(parent)],{},False),
 ('approved-find-links-preserves-nesting',["wheel","--no-deps","--wheel-dir",str(j/'good-built'),str(parent)],{},True),
]
proof=dict(schema='deeplinkengine-real-pip-nested-build-fixture/v1',pip_version=__import__('pip').__version__,python=sys.executable,base_prefix=sys.base_prefix,fixture_package_is_not_torch=True,framework_imported=False,model_started=False,started=time.time(),cases=[])
for name,args,changes,success in cases:
    offset=traces.stat().st_size if traces.exists() else 0
    q=subprocess.run([sys.executable,'-B','-S','-m','pip',*args],env=dict(base,**changes),capture_output=True,text=True,timeout=120)
    (j/(name+'.log')).write_text(q.stdout+q.stderr)
    rows=[]
    if traces.exists():
        with traces.open() as f:f.seek(offset);rows=[json.loads(line) for line in f if line.strip()]
    row=dict(name=name,command=args,returncode=q.returncode,expected_success=success,trace=rows,log_sha256=hashlib.sha256((j/(name+'.log')).read_bytes()).hexdigest());proof['cases'].append(row)
    assert (q.returncode==0)==success,(name,q.stdout[-2000:],q.stderr[-4000:])
    if name=='approved-find-links-preserves-nesting':
        assert {r['dependency_version'] for r in rows}=={'1.0','2.0'},rows
        assert len({r['dependency_file'] for r in rows})>=2,rows

# The build controller owns Linux process groups and /proc birth identities.
# Windows verifies actual pip's four portable isolation cases above; it does
# not claim support for installing or building an Ascend SDK.
if os.name == 'nt':
    proof.update(ended=time.time(),production_graph=dict(status='not-run',reason='Linux-only source-build controller'),source_runtime_or_upstream_requirements_modified=False)
    (j/'proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(dict(status='passed',cases=len(proof['cases']),production_graph='Linux-only',fixture_package_is_not_torch=True)))
    sys.exit(0)

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from official import enable_module_imports

enable_module_imports()
import official_source_candidate as candidate
import official_source_build_graph as graph
graph_folder=j/'production-graph';graph_folder.mkdir();cache=graph_folder/'cache';approved_cache=cache/'wheels';approved_cache.mkdir(parents=True)
registry={}
for package,version,original,stage in [
    ('scope_build_dep','1.0',wheelhouse/'scope_build_dep-1.0-py3-none-any.whl','nested'),
    ('scope_build_dep','2.0',wheelhouse/'scope_build_dep-2.0-py3-none-any.whl','01-core'),
    ('scope_nested','0.1',child,'nested-source')]:
    local=approved_cache/original.name
    import shutil
    shutil.copyfile(original,local);digest=hashlib.sha256(local.read_bytes()).hexdigest()
    registry['sha256:'+digest]=dict(package=package,version=version,stage=stage,requirement=package+'=='+version,
        filename=local.name,bytes=local.stat().st_size,sha256=digest,url=local.as_uri(),upstream_url=local.as_uri())
core=graph_folder/'original-metadata';core.mkdir();original=b'[build-system]\nrequires=["scope_build_dep==2.0","scope_nested==0.1"]\n';(core/'pyproject.toml').write_bytes(original)
build=graph_folder/'build';build.mkdir();graph_trace=graph_folder/'trace.jsonl';environment=dict(base,SCOPE_FIXTURE_TRACE=str(graph_trace))
with patch.object(candidate,'build_dependency_artifacts',return_value=registry),patch('transport_trust.configure',return_value=dict(offline_fixture_only=True)):
    links,graph_proof=graph.prepare(Path(sys.executable),core,cache,build,environment,sdk=graph_folder/'fixture-sdk',stage='01-core')
assert (core/'pyproject.toml').read_bytes()==original
environment.update(PIP_FIND_LINKS=str(links),PIP_CACHE_DIR=str(graph_folder/'actual-pip-cache'))
actual=subprocess.run([sys.executable,'-B','-S','-m','pip','wheel','--no-deps','--no-index','--no-cache-dir','--find-links',str(links),'--wheel-dir',str(graph_folder/'actual-built'),str(parent)],env=environment,capture_output=True,text=True,timeout=120)
(graph_folder/'actual-parent-build.log').write_text(actual.stdout+actual.stderr)
assert actual.returncode==0,actual.stdout[-2000:]+actual.stderr[-4000:]
trace=[json.loads(line) for line in graph_trace.read_text().splitlines()]
assert {row['dependency_version'] for row in trace}=={'1.0','2.0'}
assert len({row['dependency_file'] for row in trace})>=2
proof['production_graph']=dict(graph=graph_proof,actual_parent_build_returncode=actual.returncode,trace=trace,
    fixture_registry_not_engine_assets=True,tls_mock_applies_only_to_no_network_fixture=True)

assert 'torch' not in sys.modules
proof.update(ended=time.time(),source_runtime_or_upstream_requirements_modified=False)
(j/'proof.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(dict(status='passed',cases=len(proof['cases']),production_graph=True,proof=str(j/'proof.json'),fixture_package_is_not_torch=True)))
