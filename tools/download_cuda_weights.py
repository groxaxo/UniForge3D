"""Pinned, resumable public checkpoint downloads; never modifies Mac source weights."""
import concurrent.futures, hashlib, json, pathlib, subprocess, time
ROOT=pathlib.Path('/home/op/uniforge3d-work/weights')
MANIFEST=json.loads(pathlib.Path(__file__).with_name('source_verification.json').read_text())
REPOS={'uniforge3d':'xibushu/UniForge3D','trellis2':'microsoft/TRELLIS.2-4B','trellis1':'microsoft/TRELLIS-image-large'}
def download(job):
 group, repo, rev, file=job
 rel=file['path'].split('/weights/source/'+group+'/')[1]
 target=ROOT/group/rel
 target.parent.mkdir(parents=True,exist_ok=True)
 start=time.time()
 if not target.exists() or target.stat().st_size != file['bytes']:
  part=target.with_name(target.name+'.partial')
  subprocess.run(['curl','--fail','--location','--retry','5','--continue-at','-','--output',str(part),f'https://huggingface.co/{repo}/resolve/{rev}/{rel}'],check=True)
  part.replace(target)
 digest=hashlib.file_digest(target.open('rb'),'sha256').hexdigest()
 assert digest==file['sha256'],(target,digest)
 print(json.dumps({'file':str(target),'bytes':target.stat().st_size,'sha256':digest,'seconds':time.time()-start}),flush=True)
if __name__=='__main__':
 jobs=[(g,r,MANIFEST['repositories'][g]['revision'],f) for g,r in REPOS.items() for f in MANIFEST['repositories'][g]['files']]
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(download,jobs))
 print('PUBLIC_WEIGHTS_READY',flush=True)
