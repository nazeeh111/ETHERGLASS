"""Source-bound exported I/Q waterfall comparison; no inference or fitting."""
from pathlib import Path
import json,hashlib,re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
P=Path(__file__).resolve().parent;V=json.loads((P/'provenance.json').read_text());D=np.load(P/'figure-data.npz')
for name,h in V['source_sha256'].items():assert hashlib.sha256((P.parents[1]/name).read_bytes()).hexdigest()==h
paper='#eeede6';ink='#252c2b';muted='#596052';copper='#a4472a'
def render(phone=False):
 w,h=(440,1250) if phone else (1000,930)
 with plt.rc_context({'font.family':['DejaVu Sans','Arial','sans-serif'],'svg.fonttype':'none','svg.hashsalt':'etherglass-exported-components'}):
  fig=plt.figure(figsize=(w/100,h/100),dpi=100,facecolor=paper);ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,w),ylim=(h,0));ax.axis('off')
  def txt(x,y,s,size=18,color=ink,weight='normal'):ax.text(x,y,s,fontsize=size*.72,color=color,weight=weight,va='center')
  m=24 if phone else 40;txt(m,32,'ETHERGLASS / FIXED SYNTHETIC REFERENCE',12,muted,'bold')
  if phone:txt(m,76,'Estimate a target.',30,weight='bold');txt(m,114,'Keep the residual.',30,weight='bold');txt(m,153,'16,384 complex samples · 1 MHz · Hybrid',16)
  else:txt(m,84,'Estimate a target. Keep the residual.',41,weight='bold');txt(m,128,'16,384 complex samples · 1 MHz · fixed synthetic seed 870001 · Hybrid checkpoint',19)
  times=D['time_s']*1000;freq=D['frequency_hz']/1000;dt=float(np.diff(times)[0]);df=float(np.diff(freq)[0]);ext=[times[0]-dt/2,times[-1]+dt/2,freq[0]-df/2,freq[-1]+df/2]
  for i,(name,label,detail) in enumerate([('mixture','INPUT MIXTURE','Known generated QPSK + overlapping chirp + noise'),('target','TARGET ESTIMATE','Conditioned on 0 Hz offset and 125 kbaud prior'),('residual','UNASSIGNED RESIDUAL','Interference, noise and reconstruction error')]):
   top=(245+i*270) if phone else (211+i*206);left=67 if phone else 82;pw=345 if phone else 865;ph=168 if phone else 144
   txt(m,top-48,label,15,copper,'bold');txt(m,top-23,detail,13 if phone else 16)
   plot=fig.add_axes([left/w,(h-top-ph)/h,pw/w,ph/h]);im=plot.imshow(D[name+'_relative_db'].T,origin='lower',extent=ext,aspect='auto',interpolation='none',cmap='magma',norm=Normalize(-55,0,clip=True));im.set_gid('field-'+name);plot.set(xlim=(0,16.384),ylim=(-500,500),xticks=[0,4,8,12,16],yticks=[-500,0,500]);plot.tick_params(labelsize=9 if phone else 10);plot.set_ylabel('Offset / kHz',fontsize=10);plot.set_xlabel('Time / ms',fontsize=10)
  cy=1090 if phone else 829;cbax=fig.add_axes([.18,(h-cy-13)/h,.68,13/h]);cb=fig.colorbar(im,cax=cbax,orientation='horizontal',ticks=[-55,-40,-20,0]);cb.ax.tick_params(labelsize=10);cb.set_label('dB re mixture STFT peak / shared scale' if phone else 'dB relative to the highest mixture STFT bin (shared reference)',fontsize=10 if phone else 11)
  if phone:
   txt(24,1175,'256-sample Hann window · 128-sample hop',14,muted);txt(24,1203,'Disagreement indicator, not a calibrated bound',14,muted);txt(24,1230,'Fixed generated example, not physical RF validation',13,muted)
  else:
   txt(40,896,'Shared color scale · 256-sample Hann window · 128-sample hop · 3.90625 kHz bins',16,muted)
   txt(40,922,'Residual = mixture − estimate; arithmetic closure does not prove correct separation. Ordinary demo: indicator only.',14,muted)
  name='exported-components-mobile' if phone else 'exported-components';fig.savefig(P/(name+'.png'),dpi=100,facecolor=paper);fig.savefig(P/(name+'.svg'),facecolor=paper,metadata={'Date':None});plt.close(fig)
  svg=(P/(name+'.svg')).read_text();svg=re.sub(r'font-family:[^;\"]+',"font-family: 'DejaVu Sans', Arial, sans-serif",svg);svg=svg.replace('<svg ','<svg role="img" aria-labelledby="eg-title eg-desc" ',1);i=svg.index('>',svg.index('<svg '))+1;svg=svg[:i]+'<title id="eg-title">Estimate a target. Keep the residual.</title><desc id="eg-desc">Actual exported fixed synthetic I/Q, target estimate and unassigned residual. All three native STFT grids share one mixture-referenced dB scale. Supplied target priors are zero Hz and125kbaud. The residual contains interference, noise and reconstruction error. This is an indicator-only synthetic example, not a calibrated error bound or physical RF efficacy claim.</desc>'+svg[i:];(P/(name+'.svg')).write_text(svg)
render();render(True)
