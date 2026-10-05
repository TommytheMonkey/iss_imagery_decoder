"""Post-analysis illustration only; never feeds a classification or search."""
import json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'reports/real/18-30';r=json.loads((p/'candidates.json').read_text());assert 'elapsed_s'in r,'Whole-recording analysis must complete first'
fig,axes=plt.subplots(2,1,figsize=(12,8))
for c in r['candidates']:
 points=c['measurements'];t=np.array([q['time_s'] for q in points]);f=np.array([q['f_hz'] for q in points])
 if c['start_s']<=450 and c['end_s']>=400:axes[0].plot(t,f/1000,'.-',ms=2,lw=.5)
axes[0].set(xlim=(400,450),xlabel='Time from capture start (s)',ylabel='Residual frequency (kHz)',title='All retained RF trajectories crossing 400–450 s; not transmitter identifications')
rows=[]
for c in r['candidates']:
 for e in c['evaluations']:
  start=e.get('window_start_s',0)
  if start<=450 and min(start+40,c['audio_window_s'][1])>=400 and e.get('sync_profile'):
   rows.append((e.get('sync',{}).get('z',0)+e.get('pulse_geometry',{}).get('z',0),c,e))
rows.sort(key=lambda q:q[0],reverse=True);seen=set()
for score,c,e in rows:
 if c['id']in seen:continue
 profile=e['sync_profile'];axes[1].plot(profile['offset_ms'],profile['coherence'],'.-',label=f"{c['id']}, window {e['window_start_s']:.1f} s: {e['verdict']}")
 seen.add(c['id'])
 if len(seen)==3:break
axes[1].axvspan(0,9,color='gray',alpha=.12,label='Nominal 9 ms sync interval')
axes[1].set(xlabel='Offset from acquired line origin (ms)',ylabel='Held-out 1200 Hz coherence',title='Three largest sync/pulse scores from distinct RF hypotheses; selection shown, not a p-value')
for ax in axes:ax.grid(alpha=.2)
axes[1].legend(fontsize=8);fig.tight_layout();fig.savefig(p/'region_400_450.png',dpi=150)
