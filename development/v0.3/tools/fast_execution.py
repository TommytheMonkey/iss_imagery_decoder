"""Execution-only removal of duplicate off-gate diagnostics. No decision changes.
The off branch of threshold_gate returns the input unchanged. Compute the auto
branch once, and retain the same two bandpasses, flags and frozen candidate loop.
Require exact equality to the full serial independent RF regression before use.
"""
from robot36dec import candidates
from robot36dec.params import Params
from robot36dec.iqdemod import _Deemphasis
from robot36dec.preprocess import remove_dc,threshold_gate,bandpass
from tools import parallel_candidates
from tools.analyze_real import ROOT
import json,shutil,sys


def audio_variants(raw):
 for deemph in (0.,75.):
  x=raw if deemph==0 else _Deemphasis(48000,deemph*1e-6).process(raw)
  p=Params.default();p.gate_mode='auto'
  dc=remove_dc(x,48000,p.dc_window_s)
  gated,info=threshold_gate(dc,48000,p)
  yield dict(deemphasis_us=deemph,gate='off',gate_applied=False),bandpass(dc,48000,p.bp_lo_hz,p.bp_hi_hz)
  yield dict(deemphasis_us=deemph,gate='auto',gate_applied=bool(info.applied)),bandpass(gated,48000,p.bp_lo_hz,p.bp_hi_hz)


def evaluate(cache,plan,record,bandwidths,duration,params,out,render):
 from pathlib import Path
 fn=parallel_candidates.evaluator();fn.__globals__['_audio_variants']=audio_variants
 return fn(cache,plan,record,bandwidths,duration,params,Path(out),render)

if __name__=='__main__':
 parallel_candidates.evaluate=evaluate
 parallel_candidates.AUDIO_EXECUTION='Skip duplicate unused off-gate diagnostics; exact audio and complete RF output equivalence proven in fast-execution-equivalence.json and fast-audio-equivalence.json'
 if '--equivalence'in sys.argv:
  shutil.copy2(ROOT/'reports/validation/parallel-equivalence.json',ROOT/'reports/validation/parallel-original-equivalence.json')
  parallel_candidates.validate_freeze();parallel_candidates.equivalence()
  proof=json.loads((ROOT/'reports/validation/parallel-equivalence.json').read_text())
  proof['execution_only_change']='Off-gate returns original input directly; auto-gate statistics calculated once, not twice.'
  (ROOT/'reports/validation/fast-execution-equivalence.json').write_text(json.dumps(proof,indent=2)+'\n')
 else:
  assert json.loads((ROOT/'reports/validation/fast-execution-equivalence.json').read_text())['pass_']
  parallel_candidates.real()
