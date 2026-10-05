"""Apply current verdict mapping to existing independent measured evidence.
No DSP measurements or thresholds are changed. Availability is not evidence.
This permits regression of reporting separately from expensive RF simulation.
"""
from pathlib import Path
import json
from robot36dec.protocol import classify_evidence
from robot36dec.candidates import _write


def reclassify(directory):
    rank={'NO EVIDENCE':0,'INCONCLUSIVE':1,'PROBABLE':2,'CONFIRMED ROBOT36':3}
    root=Path(directory);data=json.loads((root/'candidates.json').read_text())
    for r in data['candidates']:
        for e in r['evaluations']:e['verdict']=classify_evidence(e)
        if r['evaluations']:
            def key(e):return rank[e['verdict']],sum(v.get('z',0) for v in e.values() if isinstance(v,dict) and 'z'in v)
            best=max(r['evaluations'],key=key);r['best_evidence']=best;r['verdict']=best['verdict'];r['reason']=best['reason']
    data['verdict']=max((r['verdict'] for r in data['candidates']),key=rank.get,default='NO EVIDENCE')
    data['reporting_reclassification']='current mapping of measured checks; no DSP measurements changed'
    _write(data,root)
    return data
if __name__=='__main__':
    import sys
    for d in sys.argv[1:]:print(d,reclassify(d)['verdict'])
