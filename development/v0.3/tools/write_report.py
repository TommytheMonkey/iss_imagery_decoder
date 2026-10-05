"""Summarize completed, immutable analyses; never changes classifications."""
import json,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
 rows=json.loads((ROOT/'reports/validation/release-final/results.json').read_text())
 texts=['# v0.3 validation and independent real-recording analysis','',
 'The v0.2.0 parent is preserved. v0.3 is an isolated development copy, not a production release. See [exact changes](v02-to-v03.patch), [implementation details](IMPLEMENTATION.md), [unit log](validation/unit-tests.log), and [RF validation results](validation/release-final/results.json).',
 '', '## Independent validation','',
 '25 unit tests passed. All 14 end-to-end RF cases completed. The RF generator imports no decoder modules. Tests use unsigned 8-bit I,Q at 1.024 MS/s, a −100 kHz capture offset, +6 kHz oscillator error, curved Doppler, fades and quantization. CNR is referenced to 48 kHz before fades; these numbers are not audio SNR.',
 '', '| Injected CNR (dB / 48 kHz) | Verdict | RF hypotheses | Supported hypotheses |','|---:|---|---:|---:|']
 by={r['case']:r for r in rows}
 for cnr in [6,0,-3,-6,-9,-12]:
  r=by[f'snr_{cnr:+d}'];texts.append(f"| {cnr:+d} | {r['verdict']} | {r['n_candidates']} | {len(r['supported'])} |")
 texts += ['','These are one seed/profile per SNR, not detection-probability estimates. Multiple supported RF hypotheses often represent sidebands/fragments of one injected transmitter. Detection does not measure recovered image quality.','',
 '| CNR (dB) | 11 kHz | 15 kHz | 25 kHz | 0 us | 75 us |', '|---:|---|---|---|---|---|']
 rank={'NO EVIDENCE':0,'INCONCLUSIVE':1,'PROBABLE':2,'CONFIRMED ROBOT36':3}
 for cnr in [6,0,-3,-6,-9,-12]:
  report=json.loads((ROOT/f'reports/validation/release-final/snr_{cnr:+d}/candidates.json').read_text())
  evaluations=[e for c in report['candidates'] for e in c['evaluations']]
  outcomes=[]
  for key,value in [('channel_bw_hz',11000),('channel_bw_hz',15000),('channel_bw_hz',25000),('deemphasis_us',0),('deemphasis_us',75)]:
   subset=[e['verdict'] for e in evaluations if (e.get(key) if key=='channel_bw_hz' else e['variant'].get(key))==value]
   outcomes.append(max(subset,key=rank.get,default='NO EVIDENCE'))
  texts.append(f'| {cnr:+d} | '+' | '.join(outcomes)+' |')
 texts += ['', 'Each column is the best outcome over all other tested choices; it is not an independently calibrated detection rate.', '',
 '| Independent control | Verdict | Supported hypotheses |','|---|---|---:|']
 for name in ['colored_noise','unrelated_fm','cw','periodic_non_sstv','constant_tone','invalid_vis','low_deviation','mixed_fm_cw']:
  r=by[name];texts.append(f"| {name} | {r['verdict']} | {len(r['supported'])} |")
 texts += ['','Invalid parity is never a valid VIS confirmation. A genuine Robot 36 body with intentionally corrupted parity may remain PROBABLE. The half-rate pulse regression, wrong porch, wrong sync width, constant-tone overrides and measured-confidence controls are also in the unit suite. No negative control was accepted as PROBABLE or CONFIRMED.','',
 '## Real recordings','',
 'Both files were scanned in full, after independent validation passed, with the same pre-frozen settings: 1,024,000 complex samples/s, u8 I,Q, center 437,650,000 Hz, nominal 437,550,000 Hz, ±30 kHz residual search and 20 ppm declared uncertainty. No timing hint, old classification, orbital trajectory, selected time region or image reference influenced acquisition. Input SHA-256 and original file size/mtime are recorded.',
 '', '| Recording | Duration (s) | RF hypotheses | Verdict | Images |','|---|---:|---:|---|---:|']
 real=[]
 for name in ['16-54','18-30']:
  d=ROOT/'reports/real'/name;r=json.loads((d/'candidates.json').read_text());real.append((name,r))
  count=sum(len(c.get('images',[])) for c in r['candidates'])
  texts.append(f"| {name} | {r['duration_s']:.3f} | {len(r['candidates'])} | {r['verdict']} | {count} |")
 for name,r in real:
  counts=dict(collections.Counter(c['verdict'] for c in r['candidates']))
  texts += ['',f'### {name}', '',f'Candidate verdict counts: {counts}.',
   f'Every RF hypothesis, rejection reason and best structural measurement: [candidate table](real/{name}/CANDIDATES.md), [candidate CSV](real/{name}/candidates.csv). Every measured time/frequency/CNR point: [track CSV](real/{name}/tracks.csv), [trajectory plot](real/{name}/frequency_tracks.png). Every bandwidth/preprocessing/window evaluation and measured VIS bit/parity decision: [full report](real/{name}/candidates.json). Input integrity: [SHA-256 and metadata](real/{name}/input-integrity.json).',
   '', 'Reported RF frequencies include oscillator offset. Linear slew is a summary of the full measured trajectory, not an ISS identity test. The list contains threshold-limited spectral placements, not necessarily distinct physical transmitters.']
  adc=json.loads((ROOT/'reports/real'/name/'adc-statistics.json').read_text())
  texts += ['',f"Whole-file ADC RMS about each component mean: I={adc['rms_about_mean_u8'][0]:.6f}, Q={adc['rms_about_mean_u8'][1]:.6f} byte codes; rail fractions {adc['rail_fraction']}. [All ADC histograms/statistics](real/{name}/adc-statistics.json). This very small variation makes low-level quantization an important sensitivity limitation. It does not by itself prove a receiver fault or absence of RF."]
  for c in r['candidates']:
   for img in c.get('images',[]):texts.append(f"Rendered for supported candidate {c['id']}: [{Path(img).name}]({Path(img).relative_to(ROOT/'reports')}).")
 r=real[1][1];region=[]
 for c in r['candidates']:
  for e in c['evaluations']:
   start=e.get('window_start_s',0)
   if start<=450 and min(start+40,c['audio_window_s'][1])>=400 and e.get('period_s'):
    region.append((c,e))
 region.sort(key=lambda pair:(pair[1].get('sync',{}).get('z',0)+pair[1].get('pulse_geometry',{}).get('z',0)),reverse=True)
 texts += ['', '### Independent review of 400–450 s in the 18:30 file','',
 'This region was reviewed only after the whole-recording scan. Below are the ten largest combined sync/pulse-geometry scores among overlapping evaluations, including rejected hypotheses; these rankings do not change any verdict.',
 '', '| Candidate | Window start (s) | RF BW (kHz) | De-emphasis (us), gate | Period (ms) | Sync z | Pulse z | Porch z | Separator z | Alternation z | Verdict / reason |',
 '|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|']
 for c,e in region[:10]:
  zs=[e.get(k,{}).get('z',0) for k in ['sync','pulse_geometry','porch','separator','alternation']]
  texts.append(f"| {c['id']} | {e['window_start_s']:.2f} | {e['channel_bw_hz']/1000:g} | {e['variant']['deemphasis_us']:g}, {e['variant']['gate']} | {e['period_s']*1000:.6f} | "+' | '.join(f'{z:.2f}' for z in zs)+f" | {e['verdict']}: {e['reason']} |")
 texts += ['','A near-150 ms best-fit period alone is insufficient: the algorithm explicitly searches that period neighborhood. The z values are engineering statistics after multiple searched hypotheses, not calibrated p-values.','',
 '## Remaining reasons a verdict could be wrong','',
 'Known injected Robot 36 is missed at sufficiently weak CNR. Noise-model errors, narrow filtering, FM threshold effects, front-end clipping, frequency uncertainty beyond the scanned range, IQ impairments, sample-rate error beyond the cadence search, short/partial frames, fades, simultaneous unresolved transmitters, sideband centroid bias, and greedy track association remain limitations. The synthetic tests use u8 quantization but do not establish sensitivity at the approximately one-code ADC RMS measured in these files; a separate amplitude/dither/gain sweep is still needed. The null suite is finite and does not calibrate false alarms across a long recording. RF motion alone cannot identify ISS without receiver calibration and ephemeris. The inherited tone/image reconstruction is heavily blurred and has not been validated across independent color reference patterns. A PROBABLE body without valid VIS has frame-origin ambiguity.','',
 '**I would not trust this decoder to establish that a real-world recording contains no recoverable Robot 36 image.** Its reports can establish what the tested hypotheses support; negative results remain bounded by measured sensitivity and search assumptions.']
 (ROOT/'reports/FINAL_REPORT.md').write_text('\n'.join(texts)+'\n')
if __name__=='__main__':main()
