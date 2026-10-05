"""Measured Robot 36 geometry. No pixel decoder or image smoothness enters a decision.

Cadence acquisition uses sync alone. Porch/separator validation uses disjoint data
and exact nominal geometry; overrides only select where to measure. Scores are
engineering evidence, not probabilities. Rejecting weak evidence is inconclusive.
"""
import numpy as np
from scipy import signal


def _coherence(x, fs, centres, duration, frequency):
    n = max(12, int(round(duration * fs)))
    offsets = np.arange(n) - (n - 1) / 2
    indices = np.rint(np.asarray(centres)[:, None] * fs + offsets).astype(int)
    valid = (indices[:, 0] >= 0) & (indices[:, -1] < len(x))
    y = x[np.clip(indices, 0, max(0, len(x)-1))].astype(float)
    w = np.hanning(n)
    osc = np.exp(-2j*np.pi*frequency*offsets/fs)
    p = np.abs((y*w) @ osc)**2
    den = w.sum() * (y*y*w).sum(axis=1)
    return np.where(valid, np.clip(2*p/np.maximum(den,1e-20),0,1),np.nan)


def _contrast(x, fs, centres, duration, expected, alternatives):
    good = _coherence(x,fs,centres,duration,expected)
    bad = np.max([_coherence(x,fs,centres,duration,f) for f in alternatives],axis=0)
    return good-bad, good


def _summary(values):
    v=np.asarray(values);v=v[np.isfinite(v)]
    if not len(v): return dict(mean=0.,se=1.,z=0.,n=0)
    # Cluster bootstrap analogue: adjacent line pairs share RF and chroma state.
    if len(v)%2:v=v[:-1]
    pairs=v.reshape(-1,2).mean(axis=1) if len(v) else np.zeros(0)
    se=max(float(np.std(pairs,ddof=1)/np.sqrt(len(pairs))) if len(pairs)>1 else 1.,.012)
    mean=float(np.mean(v)) if len(v) else 0.
    return dict(mean=mean,se=se,z=mean/se,n=int(len(v)))


def measure_geometry(x, fs, t0, period=.150, n_lines=240):
    """t0 relative to x. Exact line geometry, measured without tone reconstruction."""
    x=np.asarray(x,dtype=np.float32)
    n=min(n_lines,max(0,int((len(x)/fs-t0)/period)))
    line=t0+np.arange(n)*period; scale=period/.150
    if n<12:return dict(valid=False,reason='fewer than 12 complete lines',n_lines=n)
    centres=lambda ms: line+ms/1000*scale
    # Compare the sync to nearby off-pulse positions to require pulse geometry.
    syn,_=_contrast(x,fs,centres(4.5),.007*scale,1200,[1500,1900,2300])
    off1=_coherence(x,fs,centres(-4.5),.007*scale,1200)
    off2=_coherence(x,fs,centres(16.5),.007*scale,1200)
    pulse_coherence=_coherence(x,fs,centres(4.5),.007*scale,1200)
    shape=pulse_coherence-.5*(off1+off2)
    porch,_=_contrast(x,fs,centres(10.5),.0025*scale,1500,[1200,1900,2300])
    porch2,porch2_power=_contrast(x,fs,centres(105.25),.0013*scale,1900,[1200,1500,2300])
    cr,_=_contrast(x,fs,centres(102.25),.004*scale,1500,[1200,1900,2300])
    cb,_=_contrast(x,fs,centres(102.25),.004*scale,2300,[1200,1500,1900])
    parity=np.where(np.arange(n)%2==0,cr,cb)
    flipped=np.where(np.arange(n)%2==0,cb,cr)
    # Recovery line zero may be one line out; report that ambiguity explicitly.
    parity_flip=float(np.nanmean(flipped))>float(np.nanmean(parity))
    alternate=flipped if parity_flip else parity
    wrong=parity if parity_flip else flipped
    # Hold out alternating line pairs; paired chroma measurements stay together.
    hold=(np.arange(n)//2)%2==1
    # Presence selection uses training pairs only. Adjacent disjoint pairs test it.
    # This preserves partial frames without selecting favorable validation pixels.
    groups=np.arange(n)//4
    present=[g for g in range(n//4) if np.nanmean(syn[4*g:4*g+2])>.15]
    selection='all held-out pairs'
    if len(present)>=12:
        hold &= np.isin(groups,present)
        selection='training sync contrast >0.15; at least 12 groups of four lines'

    result={k:_summary(v[hold]) for k,v in dict(sync=syn,pulse_geometry=shape,porch=porch,
                 separator=alternate,alternation=alternate-wrong).items()}
    # Require held-out aggregate structure, not just tonal concentration.
    # Thresholds fixed before any real IQ is read; null suite includes searched maxima.
    checks={k:(result[k]['z']>=3.5 and result[k]['mean']>=.045) for k in result}
    # A 300 ms pulse train must not masquerade as 150 ms through pair averaging.
    # Require independently measured sync on both line parities.
    for parity_id in (0,1):
        key='sync_parity_'+str(parity_id)
        result[key]=_summary(syn[hold & (np.arange(n)%2==parity_id)])
        checks[key]=result[key]['mean']>=.045 and result[key]['z']>=3.5
    result['porch2']=_summary(porch2[hold])
    # The 1.5 ms second porch is intrinsically hard to resolve at weak SNR.
    # Strong contradiction vetoes all acceptance; confirmation requires positive evidence.
    wrong_fraction=float(np.mean(((porch2_power-porch2)>.65)[hold] & (porch2[hold]<-.15)))
    result['porch2']['strong_wrong_tone_fraction']=wrong_fraction
    checks['porch2_not_contradicted']=not (wrong_fraction>.6 and result['porch2']['mean']<-.08 and result['porch2']['z']<-3.5)
    checks['enough_lines']=n>=48 and hold.sum()>=24
    ok=all(checks.values())
    # Measure the aggregate sync FWHM and its centre independently of cadence.
    offsets=np.arange(-4.,17.)
    profile=np.array([np.nanmean(_coherence(x,fs,centres(ms),.002*scale,1200)[hold]) for ms in offsets])
    baseline=float(np.median(profile[(offsets<0)|(offsets>12)]))
    height=max(float(profile.max()-baseline),0.)
    above=offsets[profile>baseline+height*.5]
    width=float(len(above)) if height>.03 else None
    centre=float(np.mean(above)) if len(above) and height>.03 else None
    checks['sync_width']=width is not None and 6<=width<=12 and 2<=centre<=7
    ok=all(checks.values())
    result['sync_profile']={'offset_ms':offsets.tolist(),'coherence':profile.tolist(),
                            'fwhm_ms':width,'centre_ms':centre,'height':height}
    result.update(valid=ok,checks=checks,n_lines=n,period_s=float(period),t_line0_s=float(t0),
                  parity_flipped=parity_flip,held_out_lines=int(hold.sum()),
                  presence_selection=selection,training_present_groups=present,held_out_line_indices=np.where(hold)[0].tolist(),
                  reason='all held-out geometry checks passed' if ok else 'failed: '+', '.join(k for k,v in checks.items() if not v))
    return result


def acquire_cadence(x,fs,t0_hint=None,period_hint=None):
    """Fit sync on alternating line pairs, then validate disjoint pairs."""
    x=np.asarray(x,np.float32)
    if fs!=12000:
        from math import gcd
        g=gcd(int(fs),12000);x=signal.resample_poly(x,12000//g,int(fs)//g).astype(np.float32);fs=12000
    duration=len(x)/fs
    if duration<7.2:return dict(valid=False,reason='less than 48 lines',n_lines=int(duration/.15))
    if t0_hint is not None:
        return measure_geometry(x,fs,float(t0_hint),float(period_hint or .15))
    # A short-window STFT forms a sync contrast trace. Fit on alternating line pairs only.
    f,t,Z=signal.stft(x,fs,nperseg=96,noverlap=84,nfft=512,boundary=None)
    power=np.abs(Z)**2
    band=(f>=900)&(f<=2600);sb=(f>=1130)&(f<=1270)
    contrast=power[sb].sum(axis=0)/(power[band].sum(axis=0)+1e-20)
    contrast-=np.median(contrast)
    fit_end=min(duration,36.)
    ids=t<fit_end; tt=t[ids]; trace=contrast[ids]
    periods=np.arange(.14985,.1501501,.00000375) if period_hint is None else np.array([period_hint])
    phases=np.arange(0,.15,.0005)
    best=[]
    for P in periods:
        grid=phases[:,None]+P*np.arange(int(fit_end/P))[None,:]
        train=(np.arange(grid.shape[1])//2)%2==0
        scores=np.mean(np.interp(grid[:,train],tt,trace),axis=1)
        for i in signal.find_peaks(scores,distance=12)[0]:best.append((float(scores[i]),float(phases[i]),float(P)))
    if not best:return dict(valid=False,reason='no cadence hypothesis',n_lines=0)
    best.sort(reverse=True)
    hypotheses=[]
    for score,phase,P in best:
        t0=phase-.0045
        if t0<0:t0+=P
        if any(abs(t0-h['t_line0_s'])<.003 for h in hypotheses):continue
        h=measure_geometry(x,fs,t0,P);h['acquisition_score']=score;hypotheses.append(h)
        if len(hypotheses)>=5:break
    hypotheses.sort(key=lambda h:(h.get('valid',False),sum(v['z'] for v in h.values() if isinstance(v,dict) and 'z'in v)),reverse=True)
    result=hypotheses[0];result['cadence_hypotheses']=[{k:v for k,v in h.items() if k!='cadence_hypotheses'} for h in hypotheses[1:]]
    return result


def inspect_vis(x,fs,t0,params):
    """Independent measured-bit header check; invalid parity is never repaired."""
    if fs!=12000:
        from math import gcd
        g=gcd(int(fs),12000);x=signal.resample_poly(x,12000//g,int(fs)//g);fs=12000
    f,t,Z=signal.stft(x,fs,nperseg=240,noverlap=180,nfft=1024,boundary=None)
    p=np.abs(Z)**2;allp=p[(f>=900)&(f<=2600)].sum(axis=0)+1e-20
    leader=p[np.abs(f-1900)<60].sum(axis=0)/allp
    # Search the two independent leader plateaus, not parity-valid templates.
    length=60
    avg=np.convolve(leader,np.ones(length)/length,mode='valid')
    if len(avg)<63:return []
    score=np.minimum(avg[:-62],avg[62:])
    sync=p[np.abs(f-1200)<60].sum(axis=0)/allp
    origin=t[:len(score)]
    score *= np.sqrt(np.maximum(0,np.interp(origin+.615,t,sync)*np.interp(origin+.885,t,sync)))
    peaks=signal.find_peaks(score,height=.35,distance=180)[0]
    out=[]
    for k in peaks:
        leader0=float(t[k]-.010)
        # Refine uniformly +/-10 ms; select only leader/start/stop geometry.
        if leader0+.922>len(x)/fs or leader0<-.012:
            out.append(dict(t_leader_s=t0+leader0,measured_code=None,parity_ok=None,bits=[],llr_db=[],credible_robot36=False,reason='truncated header'))
            continue
        choices=[]
        for shift in np.arange(-.010,.0101,.001):
            a=leader0+shift
            centres=a+.625+np.arange(10)*.030
            startstop=_coherence(x,fs,centres[[0,9]],.024,1200)
            choices.append((float(np.nanmean(startstop)) if np.isfinite(startstop).any() else 0.,a,centres))
        geometry,a,centres=max(choices,key=lambda v:v[0])
        bits_centres=centres[1:9]
        p1=_coherence(x,fs,bits_centres,.024,1100)
        p0=_coherence(x,fs,bits_centres,.024,1300)
        llr=10*np.log10((p1+1e-8)/(p0+1e-8));bits=(llr>0).astype(int)
        code=int(sum(int(v)<<i for i,v in enumerate(bits[:7])))
        parity=bool(sum(bits)%2==0)
        confident=bool(geometry>=.35 and np.all(np.isfinite(llr)) and np.min(np.abs(llr))>=2.)
        out.append(dict(t_leader_s=t0+a,measured_code=code,parity_ok=parity,bits=bits.tolist(),
                        llr_db=llr.tolist(),leader_score=float(score[k]),start_stop_coherence=geometry,
                        credible_robot36=confident and parity and code==8,
                        reason='valid measured Robot36 VIS' if confident and parity and code==8 else
                        'invalid parity' if not parity else 'weak or other-mode VIS'))
    return out



def aligned_robot36_headers(evidence):
    """Associate measured VIS with recovered cadence and chroma line parity."""
    period=float(evidence.get('period_s',.150))
    origin=evidence.get('window_start_s',0.)+evidence.get('t_line0_s',0.)
    aligned=[]
    for header in evidence.get('vis',[]):
        if not header['credible_robot36']:continue
        end=header['t_leader_s']+.910*(period/.150)
        ordinal=int(round((end-origin)/period))
        residual=end-(origin+ordinal*period)
        if abs(residual)<.002 and bool(ordinal%2)==bool(evidence.get('parity_flipped',False)):
            aligned.append(dict(header,frame_origin_s=origin+ordinal*period,grid_residual_s=residual))
    return aligned


def classify_evidence(evidence):
    """Availability and a chance pulse width are not positive protocol evidence."""
    if evidence.get('valid',False):
        aligned=bool(aligned_robot36_headers(evidence))
        p2=evidence.get('porch2',{})
        second_porch=p2.get('mean',0)>=.02 and p2.get('z',0)>=2.5
        return 'CONFIRMED ROBOT36' if aligned and second_porch else 'PROBABLE'
    checks=evidence.get('checks',{})
    return 'INCONCLUSIVE' if checks.get('sync',False) and checks.get('pulse_geometry',False) else 'NO EVIDENCE'
