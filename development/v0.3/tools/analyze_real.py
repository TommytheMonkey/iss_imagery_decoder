"""Full-recording analysis guarded by the independent validation and science freeze."""
import argparse,hashlib,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from robot36dec.candidates import analyze
from robot36dec.io import IQSpec
from tools.summarize_candidates import summarize

ROOT=Path(__file__).resolve().parents[1]
INPUTS=['/data/iss/iq/ISS_2026-10-04_16-54-00_437550k_1024ksps.iq',
        '/data/iss/iq/ISS_2026-10-04_18-30-00_437550k_1024ksps.iq']

def validate_freeze():
    freeze=json.loads((ROOT/'V03_SCIENCE_FREEZE.json').read_text())
    for name,digest in {**freeze['files'],**freeze['test_files']}.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    rows=json.loads((ROOT/'reports/validation/release-final/results.json').read_text())
    assert len(rows)==14,'Full independent RF validation required first'
    for row in rows:
        if row['case'] in ['colored_noise','unrelated_fm','cw','periodic_non_sstv','constant_tone']:
            assert not row['supported'],row
        if row['case'] in ['snr_+6','low_deviation','mixed_fm_cw']:assert row['supported'],row
        if row['case']=='invalid_vis':assert row['verdict']!='CONFIRMED ROBOT36',row
    for row in rows:
        r=json.loads((ROOT/'reports/validation/release-final'/row['case']/'candidates.json').read_text())
        for name,digest in r['source_sha256'].items():assert freeze['files']['robot36dec/'+name]==digest,name
    return freeze

def run(index):
    path=Path(INPUTS[index]);destination=ROOT/'reports/real'/['16-54','18-30'][index]
    destination.mkdir(parents=True,exist_ok=True)
    initial=path.stat();h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8<<20),b''):h.update(chunk)
    last=0
    def progress(stage,fraction):
        nonlocal last
        now=time.monotonic()
        if now-last>=20:
            print(path.name,stage,f'{fraction:.1%}',flush=True);last=now
    result=analyze(path,destination,IQSpec(1024000,437650000,437550000),progress=progress)
    final=path.stat();assert (initial.st_size,initial.st_mtime_ns)==(final.st_size,final.st_mtime_ns),'Input changed'
    (destination/'input-integrity.json').write_text(json.dumps(dict(sha256=h.hexdigest(),bytes=initial.st_size,mtime_ns=initial.st_mtime_ns),indent=2)+'\n')
    summarize(destination)
    print('COMPLETE',path.name,result['verdict'],len(result['candidates']),result['elapsed_s'],flush=True)
    return dict(input=str(path),output=str(destination),verdict=result['verdict'],candidates=len(result['candidates']))

if __name__=='__main__':
    validate_freeze()
    with ProcessPoolExecutor(max_workers=2) as pool:
        results=[f.result() for f in as_completed([pool.submit(run,i) for i in range(2)])]
    (ROOT/'reports/real/results.json').write_text(json.dumps(results,indent=2)+'\n')
