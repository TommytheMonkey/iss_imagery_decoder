"""Independent pre-real validation. Never opens real captures.
Run from v0.3: python -m tools.validate_v03 --out reports/validation
"""
import argparse,json,tempfile,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from tests.rf_reference import audio,write_iq
from robot36dec.candidates import analyze,_clean
from robot36dec.io import IQSpec


def run_case(out,name,cnr=6,kind='robot36',rf_kind='fm',deviation=3000,vis=True,parity=True,lines=240):
    destination=out/name;destination.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='v03-validation-') as temp:
        x,m=audio(lines=lines,vis=vis,parity_valid=parity,kind=kind)
        path=Path(temp)/'reference.u8'
        meta=write_iq(path,x,m['fs'],cnr_db=cnr,kind=rf_kind,deviation=deviation,
                      doppler=True,fades=True,oscillator_hz=6000,seed=37)
        print('START',name,flush=True)
        report=analyze(path,destination,IQSpec(1024000,437650000),render=False)
        result=dict(case=name,injected=meta,reference=m,verdict=report['verdict'],
                     n_candidates=len(report['candidates']),elapsed_s=report['elapsed_s'],
                     supported=[r['id'] for r in report['candidates'] if r['verdict'] in ('PROBABLE','CONFIRMED ROBOT36')])
        # Invalid VIS must not be promoted to a confirmed header, even with valid geometry.
        if not parity:assert report['verdict']!='CONFIRMED ROBOT36',result
        if kind!='robot36' or rf_kind not in ('fm','mixed'):assert report['verdict'] not in ('PROBABLE','CONFIRMED ROBOT36'),result
        if kind=='robot36' and rf_kind in ('fm','mixed') and cnr==6:assert result['supported'],result
        print('RESULT',json.dumps(result),flush=True)
        return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);parser.add_argument('--cases',default='all');parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    results=[]
    cases=[]
    if args.cases in ('all','sensitivity'):
        cases += [(f'snr_{cnr:+d}',dict(cnr=cnr)) for cnr in [6,0,-3,-6,-9,-12]]
    if args.cases in ('all','negative'):
        cases += [('colored_noise',dict(rf_kind='colored',lines=120)),('unrelated_fm',dict(rf_kind='unrelated',lines=120)),
                  ('cw',dict(rf_kind='cw',lines=120)),('periodic_non_sstv',dict(kind='periodic',vis=False,lines=120)),
                  ('constant_tone',dict(kind='tone',vis=False,lines=120)),('invalid_vis',dict(parity=False,lines=120)),
                  ('low_deviation',dict(deviation=1000,lines=120)),('mixed_fm_cw',dict(rf_kind='mixed',lines=120))]
    if args.workers==1:
        for name,kw in cases:
            results.append(run_case(out,name,**kw))
            (out/'results.json').write_text(json.dumps(_clean(results),indent=2,allow_nan=False)+'\n')
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(run_case,out,name,**kw):name for name,kw in cases}
            for future in as_completed(futures):
                results.append(future.result())
                (out/'results.json').write_text(json.dumps(_clean(results),indent=2,allow_nan=False)+'\n')
    print('DONE',len(results),flush=True)
if __name__=='__main__':main()
