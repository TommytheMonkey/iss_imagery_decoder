"""Read-only whole-file ADC diagnostics; no decoder selection or thresholds."""
import json,numpy as np
from pathlib import Path
from tools.analyze_real import INPUTS,ROOT,validate_freeze

def run(path,out):
 h=np.zeros((2,256),np.int64);cross=0;count=0
 with Path(path).open('rb') as stream:
  while True:
   raw=stream.read(2<<20)
   if not raw:break
   z=np.frombuffer(raw,np.uint8).reshape(-1,2)
   for k in (0,1):h[k]+=np.bincount(z[:,k],minlength=256)
   cross+=np.sum(z[:,0].astype(np.int32)*z[:,1].astype(np.int32),dtype=np.int64);count+=len(z)
 levels=np.arange(256);mean=h@levels/count;var=h@(levels**2)/count-mean**2
 covariance=cross/count-mean.prod();result=dict(complex_samples=count,channel_order=['I','Q'],
  mean_u8=mean.tolist(),rms_about_mean_u8=np.sqrt(var).tolist(),iq_correlation=float(covariance/np.sqrt(var.prod())),
  rail_fraction=((h[:,0]+h[:,-1])/count).tolist(),histogram=h.tolist(),
  caveat='Rail codes alone do not prove clipping; no receiver gain/calibration metadata available.')
 Path(out).write_text(json.dumps(result,indent=2)+'\n');print(Path(path).name,{k:v for k,v in result.items() if k!='histogram'},flush=True)
if __name__=='__main__':
 validate_freeze()
 for path,name in zip(INPUTS,['16-54','18-30']):run(path,ROOT/'reports/real'/name/'adc-statistics.json')
