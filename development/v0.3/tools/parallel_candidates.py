"""Computational resume only. Execute the exact frozen candidate loop in workers.

AST extraction removes only progress and report checkpoint writes. DSP, windows,
variants, measurement, ranking and rendering statements are unchanged. This tool
is checked against the independent RF regression's full serial candidate output.
It never changes the frozen production source or selects candidates for omission.
"""
import ast,inspect,json,time,hashlib,tempfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
from robot36dec import candidates,iqdemod,rf
from robot36dec.params import Params
from robot36dec.io import IQSpec
from tools.analyze_real import ROOT,validate_freeze,INPUTS
from tools.summarize_candidates import summarize


def evaluator():
 tree=ast.parse(inspect.getsource(candidates.analyze))
 loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple)
           and [v.id for v in n.target.elts]==['j','record'])
 body=loop.body[:]
 assert isinstance(body[0],ast.If) and isinstance(body[0].test,ast.Name) and body[0].test.id=='progress'
 assert isinstance(body[-1],ast.Expr) and isinstance(body[-1].value,ast.Call) and body[-1].value.func.id=='_write'
 body=body[1:-1]+[ast.Return(value=ast.Name(id='record',ctx=ast.Load()))]
 names=['cache','plan','record','bandwidths','duration','params','out','render']
 fn=ast.FunctionDef(name='_frozen_candidate_loop',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=n) for n in names],kwonlyargs=[],kw_defaults=[],defaults=[]),body=body,decorator_list=[])
 module=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]));namespace=dict(candidates.__dict__)
 exec(compile(module,'<exact frozen candidate evaluation loop>','exec'),namespace)
 return namespace['_frozen_candidate_loop']


def evaluate(cache,plan,record,bandwidths,duration,params,out,render):
 return evaluator()(cache,plan,record,bandwidths,duration,params,Path(out),render)


def equivalence():
 from tests.rf_reference import audio,write_iq
 reference=json.loads((ROOT/'reports/validation/release-final/low_deviation/candidates.json').read_text())
 params=Params(**reference['parameters']);spec=IQSpec(1024000,437650000)
 with tempfile.TemporaryDirectory(prefix='parallel-equivalence-',dir=ROOT/'reports/validation') as temp:
  directory=Path(temp);x,m=audio(lines=120)
  path=directory/'reference.u8';write_iq(path,x,m['fs'],cnr_db=6,kind='fm',deviation=1000,doppler=True,fades=True,oscillator_hz=6000,seed=37)
  cache,plan,P,f,counts,duration=rf.extract_intermediate(path,spec,params,directory)
  rows=rf.spectral_candidates(P,f,counts,plan['search'],plan['f_pass1'],(300.,11000.,15000.,25000.))
  records=rf.candidate_records(rows,spec,params.iq_block_s,duration)
  with ProcessPoolExecutor(max_workers=4) as pool:
   futures={pool.submit(evaluate,cache,plan,c,(11000.,15000.,25000.),duration,params,directory,False):i for i,c in enumerate(records)}
   for future in as_completed(futures):records[futures[future]]=future.result()
  result=candidates._clean(records)
  assert result==reference['candidates'],'Parallel candidate evaluations differ from frozen serial RF regression'
  proof=dict(pass_=True,case='independent low-deviation RF regression',candidates=len(records),
             comparison='every candidate, every variant/window, every measurement and decision exactly equal to serial JSON')
  (ROOT/'reports/validation/parallel-equivalence.json').write_text(json.dumps(proof,indent=2)+'\n')
  print('EXACT EQUIVALENCE',len(records),flush=True)


def real():
 validate_freeze();assert json.loads((ROOT/'reports/validation/parallel-equivalence.json').read_text())['pass_']
 started=time.monotonic();jobs=[];reports={};plans={};destinations={}
 for name in ['16-54','18-30']:
  out=ROOT/'reports/real'/name;report=json.loads((out/'candidates.json').read_text());params=Params(**report['parameters'])
  metadata=report['metadata'];spec=IQSpec(metadata['sample_rate'],metadata['center_frequency_hz'],metadata['nominal_frequency_hz'])
  assert metadata['receiver_ppm']==0
  plan=iqdemod._plan(spec,params);cache=out/'intermediate.c64'
  assert cache.stat().st_size==int(report['duration_s']*plan['f_int'])*8
  reports[name]=report;plans[name]=(params,plan);destinations[name]=out
  for i,record in enumerate(report['candidates']):
   if 'best_evidence' not in record:jobs.append((name,i,record))
 report_source=inspect.getsource(candidates.analyze)
 with ProcessPoolExecutor(max_workers=4) as pool:
  # Long jobs first is scheduling only: retain original RF candidate order in output.
  jobs.sort(key=lambda job:job[2]['end_s']-job[2]['start_s'],reverse=True)
  futures={}
  for name,i,record in jobs:
   report=reports[name];params,plan=plans[name];out=destinations[name]
   future=pool.submit(evaluate,out/'intermediate.c64',plan,record,tuple(report['bandwidths_hz']),report['duration_s'],params,out,True)
   futures[future]=(name,i)
  last=time.monotonic()
  for future in as_completed(futures):
   name,i=futures[future];reports[name]['candidates'][i]=future.result()
   if time.monotonic()-last>20:
    for key,report in reports.items():
     done=sum('best_evidence'in c for c in report['candidates']);print('PROGRESS',key,done,'/',len(report['candidates']),flush=True)
     candidates._write(report,destinations[key])
    last=time.monotonic()
 for name,report in reports.items():
  rank={'NO EVIDENCE':0,'INCONCLUSIVE':1,'PROBABLE':2,'CONFIRMED ROBOT36':3}
  report['verdict']=max((c['verdict'] for c in report['candidates']),key=rank.get,default='NO EVIDENCE')
  report['elapsed_s']=time.monotonic()-started
  report['execution']=dict(mode='parallel resume of exact frozen candidate loop',workers=4,
   serial_checkpoint='serial-checkpoint.json',equivalence_proof='../../validation/parallel-equivalence.json',
   note='Retained completed serial evaluations; every remaining RF candidate evaluated with the same loop. Scheduling/checkpoint frequency only changed.',
   evaluator_source_sha256=hashlib.sha256(report_source.encode()).hexdigest())
  report['execution']['audio_execution']=globals().get('AUDIO_EXECUTION','unmodified production audio variants')
  report['execution']['elapsed_scope']='parallel resume only; initial RF scan and cached serial work excluded'
  candidates._write(report,destinations[name]);summarize(destinations[name])
  path=Path(report['input']);stat=path.stat();h=hashlib.sha256()
  with path.open('rb') as stream:
   for chunk in iter(lambda:stream.read(8<<20),b''):h.update(chunk)
  final=path.stat();assert (stat.st_size,stat.st_mtime_ns)==(final.st_size,final.st_mtime_ns)
  (destinations[name]/'input-integrity.json').write_text(json.dumps(dict(sha256=h.hexdigest(),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns),indent=2)+'\n')
  print('COMPLETE',name,report['verdict'],len(report['candidates']),flush=True)
 (ROOT/'reports/real/results.json').write_text(json.dumps([{k:r[k] for k in ['input','duration_s','verdict','elapsed_s']} for r in reports.values()],indent=2)+'\n')

if __name__=='__main__':
 import sys
 if '--equivalence' in sys.argv:validate_freeze();equivalence()
 else:real()
