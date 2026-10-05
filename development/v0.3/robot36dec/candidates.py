"""Candidate-preserving IQ analysis. Output includes every spectral trajectory,
all tested audio variants, measured VIS bits, geometry, and explicit decisions.
The input and all trajectories are independent of past decoder classifications.
"""
import json,math,tempfile,time,copy,hashlib
from pathlib import Path
from dataclasses import replace,asdict
import numpy as np
from scipy import signal
from . import rf,protocol,__version__
from .io import IQSpec
from .params import Params


def _clean(value):
    if isinstance(value,dict):return {k:_clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_clean(v) for v in value]
    if isinstance(value,np.generic):return _clean(value.item())
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


def _audio_variants(raw):
    # Both de-emphasis choices and automatic impulse gate; never use reconstructed
    # pixels to select a variant. Multiplicity is recorded and null-tested.
    from .preprocess import remove_dc,threshold_gate,bandpass
    from .iqdemod import _Deemphasis
    for deemph in (0.,75.):
        x=raw if deemph==0 else _Deemphasis(48000,deemph*1e-6).process(raw)
        p=Params.default();p.gate_mode='auto'
        dc=remove_dc(x,48000,p.dc_window_s)
        for gating in ('off','auto'):
            p.gate_mode=gating
            gated,info=threshold_gate(dc,48000,p)
            xb=bandpass(gated,48000,p.bp_lo_hz,p.bp_hi_hz)
            yield dict(deemphasis_us=deemph,gate=gating,gate_applied=bool(info.applied)),xb


def analyze(path,out,spec,params=None,bandwidths=(11000.,15000.,25000.),receiver_ppm=0.,uncertainty_ppm=20.,progress=None,render=True):
    params=params or Params.default();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if uncertainty_ppm<0 or not math.isfinite(receiver_ppm) or not math.isfinite(uncertainty_ppm):raise ValueError('finite receiver PPM and nonnegative uncertainty are required')
    if not bandwidths or any(not 1000<=bw<45000 for bw in bandwidths):raise ValueError('channel bandwidths must lie in [1000,45000) Hz')
    # Positive ppm means measured RF frequencies are shifted positive at the
    # nominal baseband; expose correction and remaining uncertainty separately.
    calibrated=replace(spec,signal_freq=spec.signal_freq+spec.signal_freq*receiver_ppm*1e-6)
    user_search=params.iq_search_hz
    required_search=11250+spec.signal_freq*uncertainty_ppm*1e-6+1000
    params=replace(params,iq_search_hz=max(params.iq_search_hz,required_search),iq_channel_bw_hz=max(bandwidths))
    started=time.time()
    package=Path(__file__).parent
    source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(package.glob('*.py'))}
    source_hashes['params_frozen.json']=hashlib.sha256((package/'params_frozen.json').read_bytes()).hexdigest()
    report=dict(version=__version__,source_sha256=source_hashes,parameters=asdict(params),input=str(Path(path).resolve()),metadata=dict(sample_rate=spec.fs,
        format='unsigned 8-bit interleaved I,Q',center_frequency_hz=spec.center_freq,
        nominal_frequency_hz=spec.signal_freq,receiver_ppm=receiver_ppm,uncertainty_ppm=uncertainty_ppm),
        verdict='NO EVIDENCE',candidates=[],limitations=[
        'RF candidate list is threshold-limited, not an exhaustive list of all physical transmitters.',
        'A spectral trajectory is not a carrier lock or identification.',
        'Weak evidence and negative outcomes do not prove absence of a recoverable image.',
        'No orbital ephemeris: reported residual includes Doppler and oscillator error.',
        'Confidence thresholds are engineering rules, not posterior probabilities.'])
    with tempfile.TemporaryDirectory(prefix='rf-cache-',dir=out) as directory:
        cache,plan,P,freqs,counts,duration=rf.extract_intermediate(path,calibrated,params,directory,progress)
        report.update(duration_s=duration,search_user_hz=user_search,search_required_hz=required_search,search_requested_hz=params.iq_search_hz,search_effective_hz=plan['search'],
                      intermediate_rate=plan['f_int'],bandwidths_hz=list(bandwidths),
                      front_end_delay_s=(len(plan['h1'])-1)/(2*spec.fs) if plan['h1'] is not None else 0.,
                      audio_timing_grid_s=.0005,rf_time_grid_s=params.iq_block_s,spectral_bin_hz=float(freqs[1]-freqs[0]))
        if plan['search']+1<params.iq_search_hz:raise ValueError('requested frequency uncertainty exceeds available capture/passband')
        np.savez_compressed(out/'rf_spectra.npz',power=P,freqs=freqs,counts=counts,block_s=params.iq_block_s)
        rows=rf.spectral_candidates(P,freqs,counts,plan['search'],plan['f_pass1'],(300.,)+tuple(bandwidths))
        records=rf.candidate_records(rows,calibrated,params.iq_block_s,duration)
        report['candidates']=records
        _write(report,out)
        for j,record in enumerate(records):
            if progress:progress('candidate '+record['id'],j/max(1,len(records)))
            # All candidates are demodulated, including single-block RF peaks.
            # Add context uniformly, not based on where a prior decoder found pulses.
            a=max(0,record['start_s']-2.);b=min(duration,record['end_s']+2.)
            record['audio_window_s']=[a,b]
            best=None;best_audio=None
            span=b-a
            windows=list(np.arange(0,max(0,span-40)+.001,20.))
            if not windows or windows[-1]+40<span-.01:windows.append(max(0,span-40))
            for bw in bandwidths:
                for start in windows:
                    # Bound RAM independently of recording/candidate duration.
                    raw=rf.demodulate_candidate(cache,plan,record,a+start,min(a+start+40,b),bw)
                    off_cache={}
                    for variant,chunk in _audio_variants(raw):
                        key=variant['deemphasis_us']
                        if variant['gate']=='auto' and not variant['gate_applied']:
                            ev=copy.deepcopy(off_cache[key]);ev['variant']=variant
                            ev['reused_identical_audio']=True
                        else:
                            ev=protocol.acquire_cadence(chunk,48000)
                            ev.update(variant=variant,channel_bw_hz=bw,window_start_s=float(a+start))
                            vis=protocol.inspect_vis(chunk,48000,a+start,params) if len(chunk)>=48000 else []
                            ev['vis']=vis
                            ev['verdict']=protocol.classify_evidence(ev)
                            if variant['gate']=='off':off_cache[key]=ev
                        record['evaluations'].append(ev)
                        score=sum(v.get('z',0) for v in ev.values() if isinstance(v,dict) and 'z'in v)
                        rank=({'NO EVIDENCE':0,'INCONCLUSIVE':1,'PROBABLE':2,'CONFIRMED ROBOT36':3}[ev['verdict']],score)
                        if best is None or rank>best[0]:best=(rank,ev);best_audio=(chunk.copy(),a+start)
            record['best_evidence']=best[1] if best else None
            record['verdict']=best[1]['verdict'] if best else 'NO EVIDENCE'
            record['reason']=best[1]['reason'] if best else 'no analyzable audio'
            record['rf_status']='protocol supported' if record['verdict'] in ('PROBABLE','CONFIRMED ROBOT36') else 'RF hypothesis, unverified'
            if render and record['verdict'] in ('PROBABLE','CONFIRMED ROBOT36'):
                _render(best_audio,best[1],params,out/record['id'],record)
            _write(report,out)
    rank={'NO EVIDENCE':0,'INCONCLUSIVE':1,'PROBABLE':2,'CONFIRMED ROBOT36':3}
    report['verdict']=max((r['verdict'] for r in report['candidates']),key=rank.get,default='NO EVIDENCE')
    report['elapsed_s']=time.time()-started;_write(report,out)
    return report


def _write(report,out):
    temp=out/'candidates.json.tmp';temp.write_text(json.dumps(_clean(report),indent=2,allow_nan=False)+'\n');temp.replace(out/'candidates.json')


def _render(audio,ev,params,out,record):
    from .decode import decode_robot36
    from .timing import Timing
    from PIL import Image
    out.mkdir(exist_ok=True)
    x,absolute=audio
    aligned=protocol.aligned_robot36_headers(ev)
    origin=aligned[0]['frame_origin_s'] if aligned else absolute+ev['t_line0_s']+(ev['period_s'] if ev.get('parity_flipped') else 0.)
    timing=Timing(t_line0=origin,period=ev['period_s'],source='v0.3 measured geometry',confidence=0.,agreement={'protocol_evidence':ev})
    result=decode_robot36(x,48000,absolute,timing,params,raw_audio=x,gate_applied=ev['variant']['gate_applied'])
    Image.fromarray(result.rgb).save(out/'robot36.png')
    Image.fromarray(np.uint8(np.clip(result.Y,0,255))).save(out/'luminance.png')
    record['images']=[str(out/'robot36.png'),str(out/'luminance.png')]
