"""Collect recorded run facts without loading models or changing run artifacts."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--gpu-log',type=Path)
a=p.parse_args()
samples=[]
if a.gpu_log:
    for row in csv.DictReader(a.gpu_log.open()):
        row={k.strip():v.strip() for k,v in row.items()}
        timestamp=datetime.strptime(row['timestamp'],'%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=ZoneInfo('Pacific/Auckland'))
        mem=int(row['memory.used [MiB]'].split()[0]);samples.append((timestamp,mem))
result={}
for case in ['case01','facu-20261020']:
    candidates=sorted((a.root/case).glob('*/prediction.glb'),key=lambda x:x.stat().st_mtime)
    if not candidates:continue
    glb=candidates[-1]
    stages=[json.loads(x.read_text()) for x in glb.parent.glob('*.done.json')]
    stages.sort(key=lambda x:x['started_at'])
    start=datetime.fromisoformat(stages[0]['started_at']);end=datetime.fromisoformat(stages[-1]['finished_at'])
    stats=glb.with_suffix('.stats.json')
    result[case]={'glb':str(glb),'stages':stages,'wall_seconds':(end-start).total_seconds(),
        'sum_stage_seconds':sum(x['seconds'] for x in stages),
        'peak_allocated_bytes':max(x['peak_allocated_bytes'] for x in stages),
        'peak_reserved_bytes':max(x['peak_reserved_bytes'] for x in stages),
        'maximum_process_rss_kib':max(x['max_rss_kib'] for x in stages),
        'sampled_peak_gpu_mib':max([m for t,m in samples if start<=t<=end],default=None),
        'geometry':json.loads(stats.read_text()) if stats.exists() else None,
        'inputs':json.loads((glb.parent/'inputs.json').read_text())}
print(json.dumps(result,indent=2))
