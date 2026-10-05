"""Independent encoder/RF simulator: deliberately imports no robot36dec code.
Units and waveform formulas are explicit. CNR reference bandwidth is 48 kHz.
"""
import numpy as np
from pathlib import Path


def audio(fs=12000,lines=240,vis=True,parity_valid=True,kind='robot36',porch2_hz=1900,sync_ms=9):
    duration=2+lines*.15+1
    n=int(duration*fs);freq=np.full(n,1900.)
    t0=1.91 if vis else 1.
    if vis:
        segments=[(1900,.3),(1200,.01),(1900,.3),(1200,.03)]
        bits=[0,0,0,1,0,0,0];bits.append(sum(bits)%2 if parity_valid else 1-sum(bits)%2)
        segments += [(1100 if b else 1300,.03) for b in bits]+[(1200,.03)]
        position=1.
        for f,d in segments:
            freq[round(position*fs):round((position+d)*fs)]=f;position+=d
    for row in range(lines):
        a=t0+row*.15
        if kind=='tone':continue
        segments=[(1200,sync_ms/1000),(1500,(12-sync_ms)/1000),(0,.088),(1500 if row%2==0 else 2300,.0045),(porch2_hz,.0015),(0,.044)]
        if kind=='periodic':segments=[(1200,.009),(1900,.141)]
        pos=a
        for f,d in segments:
            lo=round(pos*fs);hi=min(n,round((pos+d)*fs))
            if f:freq[lo:hi]=f
            else:
                # Independent image: sinusoidal bars, row gradients; chroma varies slowly.
                coordinate=np.linspace(0,1,hi-lo,endpoint=False)
                freq[lo:hi]=1900+260*np.sin(2*np.pi*(3*coordinate+row/51))+100*np.cos(row/23)
            pos+=d
    samples=np.sin(2*np.pi*np.cumsum(freq)/fs).astype(np.float32)
    return samples,dict(fs=fs,t_line0=t0,period=.15,duration=len(samples)/fs)


def write_iq(path,samples,fs_audio,cnr_db=6.,fs_iq=1024000,coarse=-100000.,deviation=3000.,
             doppler=True,fades=True,oscillator_hz=6000.,seed=7,kind='fm'):
    rng=np.random.default_rng(seed);n=int(len(samples)/fs_audio*fs_iq)
    amplitude=.015;noise_power=amplitude**2*fs_iq/(48000*10**(cnr_db/10))
    phase=0.;clipped=0;total=0
    with Path(path).open('wb') as stream:
        for a in range(0,n,1<<19):
            b=min(n,a+(1<<19));t=np.arange(a,b)/fs_iq
            m=np.interp(t,np.arange(len(samples))/fs_audio,samples)
            offset=coarse+oscillator_hz+(6500-180*t+1.5*(t-18)**2 if doppler else 0.)
            inst=offset+(deviation*m if kind in ('fm','mixed') else 0.)
            inc=np.cumsum(inst)/fs_iq;z=amplitude*np.exp(2j*np.pi*(phase+inc));phase=(phase+inc[-1])%1
            if fades:z*=1-.80*np.exp(-((t-14)/1.5)**2)-.65*np.exp(-((t-28)/2)**2)
            noise=np.sqrt(noise_power/2)*(rng.normal(size=b-a)+1j*rng.normal(size=b-a))
            if kind=='colored':
                # Broad noise bump; no coherent transmitter, independent of RF detector.
                F=np.fft.fft(noise);f=np.fft.fftfreq(len(noise),1/fs_iq)
                noise=np.fft.ifft(F*np.sqrt(1+.5*np.exp(-.5*((f-(coarse+3000))/4000)**2)));z=np.zeros_like(noise)
            if kind=='unrelated':
                unrelated=2.5*np.sin(2*np.pi*430*t)+.5*np.sin(2*np.pi*3110*t)
                inc=np.cumsum(offset+3000*unrelated)/fs_iq;z=amplitude*np.exp(2j*np.pi*(phase+inc));phase=(phase+inc[-1])%1
            if kind=='mixed':z+=2*amplitude*np.exp(2j*np.pi*(coarse+25000)*t)
            z+=noise
            components=np.column_stack((z.real,z.imag));clipped+=np.sum(np.abs(components)>1);total+=components.size
            np.uint8(np.rint(np.clip(components*127.5+127.5,0,255))).tofile(stream)
    return dict(cnr_db=cnr_db,cnr_reference_hz=48000,rf_deviation_hz=deviation,
                sample_rate=fs_iq,coarse_hz=coarse,oscillator_hz=oscillator_hz,clip_fraction=clipped/total)
