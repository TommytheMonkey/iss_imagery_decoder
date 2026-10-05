"""Export every RF hypothesis and every measured track point; no filtering."""
import csv,json,argparse
from pathlib import Path


def summarize(directory):
    root=Path(directory);data=json.loads((root/'candidates.json').read_text())
    fields=['id','start_s','end_s','n_measurements','median_offset_hz','linear_slew_hz_s','median_cnr_db','verdict','reason',
            'period_ms','sync_z','sync_parity_0_z','sync_parity_1_z','pulse_geometry_z','porch_z','separator_z','alternation_z','porch2_z','sync_width_ms','valid_vis_count','images']
    with (root/'candidates.csv').open('w',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=fields);writer.writeheader()
        for c in data['candidates']:
            e=c.get('best_evidence') or {};row={k:c.get(k) for k in fields[:9]}
            row.update(period_ms=1000*e['period_s'] if e.get('period_s') else None,
                       sync_width_ms=e.get('sync_profile',{}).get('fwhm_ms'),
                       valid_vis_count=sum(v['credible_robot36'] for v in e.get('vis',[])),images=';'.join(c.get('images',[])))
            for key in ['sync','sync_parity_0','sync_parity_1','pulse_geometry','porch','separator','alternation','porch2']:row[key+'_z']=e.get(key,{}).get('z')
            writer.writerow(row)
    with (root/'tracks.csv').open('w',newline='') as out:
        fields=['candidate_id','time_s','f_hz','rf_hz','cnr_db','cnr_reference_hz','cnr_48k_db','z','rf_width_hz','spectral_peak_hz','narrow']
        writer=csv.DictWriter(out,fieldnames=fields,extrasaction='ignore');writer.writeheader()
        for c in data['candidates']:
            for point in c['measurements']:writer.writerow(dict(point,candidate_id=c['id']))
    text=[f"Verdict: **{data['verdict']}**. {len(data['candidates'])} RF hypotheses.",
          'Trajectories can include sideband aliases and fragments of the same physical transmitter. They are not a count of transmitters.',
          '| Candidate | Time (s) | Median offset (Hz) | Slew (Hz/s) | Verdict | Reason |',
          '|---|---:|---:|---:|---|---|']
    for c in data['candidates']:
        slope=c['linear_slew_hz_s'];text.append(f"| {c['id']} | {c['start_s']:.2f}–{c['end_s']:.2f} | {c['median_offset_hz']:.1f} | {slope:.1f}"+f" | {c['verdict']} | {c['reason']} |" if slope is not None else f"| {c['id']} | {c['start_s']:.2f}–{c['end_s']:.2f} | {c['median_offset_hz']:.1f} | — | {c['verdict']} | {c['reason']} |")
    (root/'CANDIDATES.md').write_text('\n\n'.join(text[:2])+'\n\n'+'\n'.join(text[2:])+'\n')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(12,6))
        for c in data['candidates']:
            points=c['measurements'];ax.plot([p['time_s'] for p in points],[p['f_hz']/1000 for p in points],'.-',ms=2,lw=.6)
        ax.set(xlabel='Time from recording start (s)',ylabel='Residual relative to corrected nominal (kHz)',title=f"{Path(data['input']).name}: all RF hypotheses")
        ax.grid(alpha=.2);fig.tight_layout();fig.savefig(root/'frequency_tracks.png',dpi=150);plt.close(fig)
    except ImportError:pass
    return data

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');args=p.parse_args();summarize(args.directory)
