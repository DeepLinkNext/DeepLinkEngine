"""Rebuild the metadata-only ARM64/CPython3.12/NumPy1.26 compatibility wheel."""
import argparse,base64,csv,hashlib,io,json
from pathlib import Path
from zipfile import ZipFile,ZipInfo,ZIP_DEFLATED
ROOT=Path(__file__).resolve().parents[1]
def build(source,out):
 record=json.loads((ROOT/'engine/r8-lock.json').read_text())['runtime_compatibility_sources']['opencv-python-headless']
 with source.open('rb') as stream:
  if hashlib.file_digest(stream,'sha256').hexdigest()!=record['upstream_sha256']:raise ValueError('Official source wheel SHA256 differs')
 if out.name!=record['filename']:raise ValueError('Keep the declared ARM64 compatibility filename and ABI tags')
 old='opencv_python_headless-5.0.0.93';new='opencv_python_headless-'+record['version'];entries=[]
 with ZipFile(source) as archive:
  for item in archive.infolist():
   if item.is_dir() or item.filename.endswith('.dist-info/RECORD'):continue
   raw=archive.read(item.filename);name=item.filename.replace(old+'.dist-info/',new+'.dist-info/',1).replace(old+'.data/',new+'.data/',1)
   if item.filename.endswith('.dist-info/METADATA'):
    lines=[];added=False
    for line in raw.decode().splitlines(keepends=True):
     if line.startswith('Version: '):line='Version: '+record['version']+'\n'
     elif line.startswith('Requires-Python: '):line='Requires-Python: >=3.12,<3.13\n'
     elif line.lower().startswith('requires-dist: numpy'):
      if not added:lines.append('Requires-Dist: numpy==1.26.4\n');added=True
      continue
     lines.append(line)
    if not added:raise ValueError('Original NumPy dependency missing')
    raw=''.join(lines).encode()
   info=ZipInfo(name,item.date_time);info.compress_type=item.compress_type;info.external_attr=item.external_attr;entries.append((info,raw))
 rows=io.StringIO(newline='');writer=csv.writer(rows,lineterminator='\n')
 for info,raw in entries:writer.writerow([info.filename,'sha256='+base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b'=').decode(),len(raw)])
 target=new+'.dist-info/RECORD';writer.writerow([target,'',''])
 with ZipFile(out,'w',compression=ZIP_DEFLATED,compresslevel=9) as archive:
  for info,raw in entries:archive.writestr(info,raw)
  info=ZipInfo(target,(2026,11,19,0,0,0));info.compress_type=ZIP_DEFLATED;info.external_attr=0o100644<<16;archive.writestr(info,rows.getvalue().encode())
 with out.open('rb') as stream:print('BUILT',out,hashlib.file_digest(stream,'sha256').hexdigest())
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--wheel',required=True,type=Path);parser.add_argument('--out',required=True,type=Path);args=parser.parse_args();build(args.wheel,args.out)
