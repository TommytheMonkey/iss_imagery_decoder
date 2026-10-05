"""Audit regressions. unittest requires no optional pytest dependency."""
import unittest,tempfile
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from scipy import signal
from tests.rf_reference import audio,write_iq
from robot36dec import protocol,rf,iqdemod
from robot36dec.params import Params
from robot36dec.io import IQSpec,u8_to_complex64
from robot36dec.cli import _signal_decision

class ProtocolRegression(unittest.TestCase):
    def test_half_rate_sync_cannot_alias_robot36(self):
        x,m=audio(vis=False)
        for row in range(1,240,2):
            a=round((1+row*.15)*12000);x[a:a+108]=0
        e=protocol.acquire_cadence(x,12000)
        self.assertFalse(e['valid'])
        self.assertFalse(all(e['checks']['sync_parity_'+str(i)] for i in (0,1)))
    def test_independent_robot36(self):
        x,m=audio(vis=False);e=protocol.acquire_cadence(x,12000)
        self.assertTrue(e['valid'],e)
        self.assertLess(abs(e['period_s']-.15),5e-6)
    def test_periodic_non_sstv(self):
        x,_=audio(vis=False,kind='periodic');e=protocol.acquire_cadence(x,12000)
        self.assertFalse(e['valid']);self.assertFalse(e['checks']['alternation'])
    def test_constant_tone_override(self):
        x,m=audio(vis=False,kind='tone');e=protocol.acquire_cadence(x,12000,t0_hint=1.,period_hint=.15)
        self.assertFalse(e['valid'])
        timing=NS(confidence=.99,agreement={'confidence_signal':.99,'protocol_evidence':e})
        self.assertFalse(_signal_decision(timing,None,None,None,NS(frac_reliable=1.),Params(), 'auto')[0])
    def test_structure_veto(self):
        timing=NS(confidence=1.,agreement={'robot36_structure_ok':False,'protocol_evidence':{'valid':True}})
        self.assertFalse(_signal_decision(timing,None,None,None,NS(frac_reliable=1.),Params(),'auto')[0])
    def test_override_preserves_measured_confidence(self):
        from robot36dec.timing import resolve_timing
        from robot36dec.sync import PeriodEstimate
        for comb in [0.,12.]:
            pe=PeriodEstimate(None,comb,0.,.15,comb)
            baseline=resolve_timing([],pe,None,None,Params(),audio_span=(0,40))
            forced=resolve_timing([],pe,None,None,Params(),t0_override=1.,period_override=.15,audio_span=(0,40))
            self.assertLessEqual(forced.agreement['confidence_signal'],baseline.agreement['confidence_signal'])
    def test_availability_is_not_structural_evidence(self):
        e={'valid':False,'checks':{'enough_lines':True,'sync_width':True}}
        self.assertEqual(protocol.classify_evidence(e),'NO EVIDENCE')
    def test_partial_frame_with_noise(self):
        x,m=audio(vis=False);rng=np.random.default_rng(43)
        start=round((1+64*.15)*12000);x[start:]=rng.normal(0,.7,len(x)-start)
        e=protocol.acquire_cadence(x,12000)
        self.assertTrue(e['valid'],e)
        self.assertIn('training sync',e['presence_selection'])
        self.assertGreaterEqual(e['held_out_lines'],24)
    def test_noise_is_not_wrong_porch_evidence(self):
        from scipy.signal import butter,sosfiltfilt
        rng=np.random.default_rng(42);x=rng.normal(size=36*12000)
        x=sosfiltfilt(butter(4,[400,3200],btype='bandpass',fs=12000,output='sos'),x)
        e=protocol.measure_geometry(x,12000,.05,.15)
        self.assertTrue(e['checks']['porch2_not_contradicted'],e)
    def test_wrong_second_porch_rejected(self):
        x,m=audio(vis=False,porch2_hz=1500)
        e=protocol.acquire_cadence(x,12000)
        self.assertFalse(e['valid']);self.assertFalse(e['checks']['porch2_not_contradicted'])
    def test_wrong_sync_width_rejected(self):
        x,m=audio(vis=False,sync_ms=3)
        e=protocol.acquire_cadence(x,12000)
        self.assertFalse(e['valid']);self.assertFalse(e['checks']['sync_width'])
    def test_diagnostics_are_strict_json(self):
        import json
        from robot36dec.candidates import _write
        with tempfile.TemporaryDirectory() as temp:
            _write({'unknown':float('nan'),'infinite':float('inf'),'valid':np.bool_(True)},Path(temp))
            text=(Path(temp)/'candidates.json').read_text()
            def reject(token):raise AssertionError('non-standard JSON constant '+token)
            decoded=json.loads(text,parse_constant=reject)
            self.assertIsNone(decoded['unknown']);self.assertIsNone(decoded['infinite'])
    def test_vis_alignment_uses_measured_clock_and_chroma_parity(self):
        period=.15006;origin=.03
        e={'valid':True,'period_s':period,'t_line0_s':origin,'window_start_s':0,
           'parity_flipped':False,'porch2':{'mean':.1,'z':5},
           'vis':[{'credible_robot36':True,'t_leader_s':origin+100*period-.910*(period/.15)}]}
        self.assertEqual(protocol.classify_evidence(e),'CONFIRMED ROBOT36')
        self.assertAlmostEqual(protocol.aligned_robot36_headers(e)[0]['frame_origin_s'],origin+100*period)
        e['parity_flipped']=True
        self.assertEqual(protocol.classify_evidence(e),'PROBABLE')
    def test_header_alone_cannot_accept(self):
        timing=NS(confidence=1.,agreement={});vis=NS(vis_code=8,parity_ok=True,confidence=1.)
        self.assertFalse(_signal_decision(timing,vis,None,None,None,Params(),'auto')[0])
    def test_valid_and_invalid_parity(self):
        for valid in [True,False]:
            x,_=audio(parity_valid=valid)
            found=protocol.inspect_vis(x,12000,0.,Params())
            target=[v for v in found if abs(v['t_leader_s']-1)<.05]
            self.assertTrue(target,found)
            self.assertEqual(target[0]['parity_ok'],valid)
            self.assertEqual(target[0]['credible_robot36'],valid)
    def test_no_parity_repair_in_legacy_header(self):
        from robot36dec.header import _decode_bits,find_vis
        x,_=audio(parity_valid=False)
        d=_decode_bits(x,12000,0,1,30);self.assertFalse(d['parity_ok'])
        found=find_vis(x,12000,0,Params())
        target=[v for v in found if abs(v.t_leader-1)<.1]
        self.assertTrue(target);self.assertFalse(target[0].parity_ok)
        self.assertLess(target[0].confidence,.6)
    def test_noise_and_unrelated_audio(self):
        rng=np.random.default_rng(77)
        t=np.arange(39*12000)/12000
        for x in [rng.normal(size=len(t)),np.sin(2*np.pi*430*t)+.2*np.sin(2*np.pi*3110*t),np.sin(2*np.pi*1900*t)]:
            self.assertFalse(protocol.acquire_cadence(x,12000)['valid'])
    def test_clock_scales_pixel_offsets(self):
        from robot36dec.decode import pixel_times
        y,c=pixel_times(0,.150045)
        y0,c0=pixel_times(0,.15)
        self.assertAlmostEqual(y[0,-1]/y0[0,-1],1.0003)
        self.assertAlmostEqual(c[0,-1]/c0[0,-1],1.0003)

class RFRegression(unittest.TestCase):
    def test_u8_and_mixer_sign(self):
        np.testing.assert_allclose(u8_to_complex64(np.array([0,255,255,0],np.uint8)),[-1+1j,1-1j])
        fs=1024000;n=np.arange(20000);z=np.exp(2j*np.pi*(-100000+1700)*n/fs)
        mixed=z*iqdemod._lo_fixed(-100000/fs,0,len(z))
        hz=fs/(2*np.pi)*np.angle(mixed[1:]*np.conj(mixed[:-1]))
        self.assertLess(abs(np.mean(hz)-1700),.01)
    def test_zero_carrier_cnr_not_minus_10(self):
        rng=np.random.default_rng(4);freqs=np.linspace(-50000,50000,4096,endpoint=False)
        p=rng.gamma(24,1/24,size=(100,len(freqs)))
        result=rf.measure_compat(p,freqs,np.full(100,24),15000,30000,45000,-6,6.5)
        self.assertFalse(result['detected'].any())
        # Signed sums independently approach zero under the null; no per-bin clipping.
        floor=rf.noise_model(p[0],freqs,24)
        signed=(p[0]-floor)[np.abs(freqs)<7500].sum()/floor[np.abs(freqs)<7500].sum()
        self.assertLess(abs(signed),.04)
    def test_low_deviation_fm_preserved(self):
        fs=128000.;n=np.arange(int(fs*4));rng=np.random.default_rng(8)
        for deviation in [500,1000,3000]:
            phase=2*np.pi*np.cumsum(3000+deviation*np.sin(2*np.pi*1900*n/fs))/fs
            z=np.exp(1j*phase)+.03*(rng.normal(size=len(n))+1j*rng.normal(size=len(n)))
            spectra=iqdemod._BlockSpectra(fs,8192,.5,8);spectra.add(z.astype(np.complex64),0)
            rows=rf.spectral_candidates(spectra.spectra(),spectra.freqs(),spectra.count,30000,45000)
            self.assertGreater(sum(bool(r) for r in rows),5)
    def test_multiple_candidates_preserved(self):
        rows=[[dict(block=i,f_hz=f,z=12.,cnr_db=0.,rf_width_hz=300,spectral_peak_hz=f,narrow=True) for f in [-6000-i*50,8000+i*10]] for i in range(20)]
        tracks=rf.link_candidates(rows)
        self.assertEqual(len(tracks),2);self.assertEqual([len(t) for t in tracks],[20,20])
    def test_two_real_spectral_peaks_preserved(self):
        fs=128000.;t=np.arange(int(fs*4))/fs;rng=np.random.default_rng(19)
        z=np.exp(2j*np.pi*8000*t)+.8*np.exp(-2j*np.pi*10000*t)+.03*(rng.normal(size=len(t))+1j*rng.normal(size=len(t)))
        spectra=iqdemod._BlockSpectra(fs,8192,.5,8);spectra.add(z.astype(np.complex64),0)
        rows=rf.spectral_candidates(spectra.spectra(),spectra.freqs(),spectra.count,30000,45000)
        for row in rows:
            self.assertTrue(any(abs(p['f_hz']-8000)<100 for p in row))
            self.assertTrue(any(abs(p['f_hz']+10000)<100 for p in row))
    def test_single_carrier_api_cannot_discard_candidates(self):
        with self.assertRaisesRegex(RuntimeError,'single-carrier'):
            iqdemod.demodulate('unused',IQSpec(1024000,437650000),Params())
    def test_search_accounts_for_uncertainty(self):
        p=Params();p.iq_search_hz=40000;p.iq_channel_bw_hz=25000
        plan=iqdemod._plan(IQSpec(1024000,437650000),p)
        self.assertEqual(plan['search'],40000)
        self.assertGreater(plan['f_int'],102400)

if __name__=='__main__':unittest.main(verbosity=2)
