"""RF hypotheses, not carrier locks. Retain all local spectral candidates.

Noise subtraction is signed before integration. Hann-bin and Welch-overlap
covariance are included conservatively. Spectral z remains a model-dependent
screening statistic, never evidence of SSTV or a calibrated false-alarm claim.
"""
from dataclasses import dataclass,asdict,replace
import math,os,tempfile,json
from pathlib import Path
import numpy as np
from scipy import signal,ndimage
from . import iqdemod
from .io import iter_iq_u8,iq_num_samples


def noise_model(row,freqs,counts):
    # Average correlated adjacent bins first: efficient robust spatial background.
    # Effective degrees of freedom include the Hann power covariance.
    group=16; n=len(row)//group
    coarse=row[:n*group].reshape(n,group).mean(axis=1)
    cf=freqs[:n*group].reshape(n,group).mean(axis=1)
    m=max(float(counts)/1.12*group/(35/18),1.)
    correction=(1-1/(9*m))**3
    width=max(5,int(round(30000/(cf[1]-cf[0])))|1)
    floor=ndimage.median_filter(coarse,size=width,mode='nearest')/correction
    floor=ndimage.gaussian_filter1d(floor,max(1,width/8),mode='nearest')
    return np.maximum(np.interp(freqs,cf,floor),np.finfo(float).tiny)


def spectral_candidates(P,freqs,counts,search_hz,pass_hz,bandwidths=(300.,11000.,15000.,25000.),min_z=6.5):
    df=float(freqs[1]-freqs[0]); valid=np.abs(freqs)<pass_hz
    rows=[]
    for b,row in enumerate(P):
        floor=noise_model(row,freqs,counts[b]); residual=row-floor
        nodes=[]; m=max(float(counts[b])/1.12,1.)
        for bw in bandwidths:
            nb=max(3,int(round(bw/df))|1)
            excess=ndimage.uniform_filter1d(residual,nb,mode='constant')*nb
            noise=ndimage.uniform_filter1d(floor,nb,mode='constant')*nb
            # sum covariance of Hann power: 1+2*(4/9+1/36)=35/18.
            variance=ndimage.uniform_filter1d(floor**2,nb,mode='constant')*nb*(35/18)/m
            z=excess/np.sqrt(np.maximum(variance,1e-30))
            eligible=valid&(np.abs(freqs)<=search_hz)&(np.abs(freqs)+bw/2<pass_hz)
            z=np.where(eligible,z,-np.inf)
            peaks=signal.find_peaks(z,height=min_z,distance=max(2,int(min(bw/3,3500)/df)))[0]
            for k in peaks:
                a=max(0,k-nb//2);c=min(len(row),k+nb//2+1)
                # A clipped centroid is a placement estimator only, never CNR.
                weights=np.maximum(residual[a:c],0)
                centre=float(np.sum(weights*freqs[a:c])/weights.sum()) if weights.sum()>0 else float(freqs[k])
                centre=float(np.clip(centre,-search_hz,search_hz))
                cnr=float(10*np.log10(max(excess[k]/max(noise[k],1e-30),1e-12)))
                if cnr>=-15:
                    nodes.append(dict(block=b,f_hz=centre,z=float(z[k]),cnr_db=cnr,rf_width_hz=float(bw),integration_width_hz=float(nb*df),
                                      cnr_reference_hz=float(nb*df),cnr_48k_db=cnr-10*math.log10(48000/(nb*df)),
                                      spectral_peak_hz=float(freqs[k]),narrow=(bw<1000)))
        # Merge overlapping placements across bandwidths; retain separated transmitters.
        nodes.sort(key=lambda n:n['z'],reverse=True); retained=[]
        for node in nodes:
            if not any(abs(node['f_hz']-v['f_hz'])<max(500,min(node['rf_width_hz'],v['rf_width_hz'])/4) for v in retained):
                retained.append(node)
        rows.append(retained)
    return rows


def link_candidates(rows,block_s=.5,max_slew=300.,max_gap_s=2.):
    """One-to-one association. No candidate count cap; long gaps create new tracks."""
    tracks=[]; active=[]
    for b,nodes in enumerate(rows):
        active=[j for j in active if (b-tracks[j][-1]['block'])*block_s<=max_gap_s]
        edges=[]
        for j in active:
            hist=tracks[j];last=hist[-1];dt=(b-last['block'])*block_s
            slope=0.
            if len(hist)>1:
                older=hist[max(0,len(hist)-5)]
                slope=np.clip((last['f_hz']-older['f_hz'])/max((last['block']-older['block'])*block_s,block_s),-max_slew,max_slew)
            pred=last['f_hz']+slope*dt
            for i,node in enumerate(nodes):
                distance=abs(node['f_hz']-pred)
                if distance<=600+max_slew*dt:edges.append((distance,j,i))
        used_t=set();used_n=set()
        for _,j,i in sorted(edges):
            if j not in used_t and i not in used_n:tracks[j].append(nodes[i]);used_t.add(j);used_n.add(i)
        for i,node in enumerate(nodes):
            if i not in used_n:tracks.append([node]);active.append(len(tracks)-1)
    return tracks


def measure_compat(P,freqs,counts,bw,search,pass_hz,lock_snr_db,min_z):
    """Compatibility diagnostics. 'detected' means RF hypothesis, not signal lock."""
    rows=spectral_candidates(P,freqs,counts,search,pass_hz,(bw,),min_z)
    n=len(rows); f=np.full(n,np.nan);cnr=np.full(n,-120.);z=np.zeros(n);det=np.zeros(n,bool)
    floors=np.array([np.mean(noise_model(row,freqs,c)) for row,c in zip(P,counts)])
    for i,nodes in enumerate(rows):
        if nodes:
            node=max(nodes,key=lambda v:v['z']);f[i]=node['f_hz'];cnr[i]=node['cnr_db'];z[i]=node['z'];det[i]=cnr[i]>=lock_snr_db
    return dict(f_meas=f,snr_db=cnr,z=z,detected=det,floor=floors,k_peak=np.zeros(n,int),ok_block=counts>=2,
                n_box=max(3,int(round(bw/(freqs[1]-freqs[0])))),narrowband=np.zeros(n,bool),
                f_narrowband=np.full(n,np.nan),frac_narrowband=np.zeros(n))


def extract_intermediate(path,spec,params,directory,progress=None):
    """Single RF scan; disk cache bounds RAM. Actual effective search is explicit."""
    plan=iqdemod._plan(spec,params);n=iq_num_samples(path);duration=n/spec.fs
    count=int(math.ceil(duration/params.iq_block_s))
    fft=iqdemod._BlockSpectra(plan['f_int'],plan['n_fft'],params.iq_block_s,count)
    front=iqdemod._FrontEnd(spec.fs,plan['f_coarse'],plan['h1'],plan['D1'])
    cache=Path(directory)/'intermediate.c64';m=0;read=0
    with cache.open('wb') as stream:
        def consume(y):
            nonlocal m
            if len(y):fft.add(y,m);y.tofile(stream);m+=len(y)
        for z in iter_iq_u8(path,chunk_samples=1<<20):
            consume(front.process(z));read+=len(z)
            if progress:progress('RF scan',read/n)
        consume(front.finish())
    return cache,plan,fft.spectra(),fft.freqs(),fft.count,duration


def demodulate_candidate(cache,plan,track,start,end,bw):
    """Independent filter/discriminator state per candidate and bandwidth."""
    fs=plan['f_int'];source=np.memmap(cache,dtype=np.complex64,mode='r')
    pad=.08;a=max(0,int((start-pad)*fs));b=min(len(source),int((end+pad)*fs))
    t=(np.arange(a,b)/fs).astype(float)
    nodes=track['measurements'];nt=np.array([n['time_s'] for n in nodes]);nf=np.array([n['f_hz'] for n in nodes])
    f=np.interp(t,nt,nf);phase=np.cumsum(f)/fs
    mixed=np.asarray(source[a:b])*np.exp(-2j*np.pi*phase).astype(np.complex64)
    # Filter at the intermediate rate before conversion so large residuals cannot alias.
    taps=signal.firwin(161,bw/2,fs=fs,window=('kaiser',8.))
    mixed=signal.fftconvolve(mixed,taps,mode='same')
    from .io import rational_ratio
    up,down=rational_ratio(48000,fs);z=signal.resample_poly(mixed,up,down)
    audio=np.zeros(len(z),np.float32)
    audio[1:]=48000/(2*np.pi)*np.angle(z[1:]*np.conj(z[:-1]))
    offset=a/fs;lo=int(round((start-offset)*48000));hi=int(round((end-offset)*48000))
    return audio[max(0,lo):min(len(audio),hi)]


def candidate_records(rows,spec,block_s,duration):
    records=[]
    for i,hist in enumerate(link_candidates(rows,block_s)):
        measurements=[dict(v,time_s=min((v['block']+.5)*block_s,duration),
                           rf_hz=spec.signal_freq+v['f_hz']) for v in hist]
        ts=np.array([v['time_s'] for v in measurements]);f=np.array([v['f_hz'] for v in measurements])
        slope=float(np.polyfit(ts-ts.mean(),f,1)[0]) if len(ts)>1 else None
        records.append(dict(id=f'RF{i+1:04d}',start_s=max(0,float(ts[0]-block_s/2)),
            end_s=min(duration,float(ts[-1]+block_s/2)),n_measurements=len(hist),
            median_cnr_db=float(np.median([v['cnr_48k_db'] for v in hist])),
            median_cnr_reference_hz=48000,cnr_estimand='screened spectral component, 48 kHz equivalent under local noise model',
            median_offset_hz=float(np.median(f)),linear_slew_hz_s=slope,
            measurements=measurements,rf_status='spectral hypothesis; SSTV unverified',
            evaluations=[],verdict='NO EVIDENCE'))
    return records
