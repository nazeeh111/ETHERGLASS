"""Generate evidence graphics and original soundtrack. Run from any directory, stdlib only."""
from pathlib import Path
import json, math, struct, wave, hashlib
ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'composition' / 'assets'
x = json.loads((ASSETS / 'reference-vis.json').read_text())
COLORS = {'mixture':'#474b97','target':'#198a8e','residual':'#a4472a'}
def svg_plot(name, title):
    s=x['curves'][name]; w,h=800,113; left,right,top,bottom=62,18,12,24
    lo,hi=-105,-30
    pts=[]
    for f,p in zip(s['frequency_hz'],s['power_db']):
        px=left+(f+500000)/1000000*(w-left-right)
        py=top+(hi-max(lo,min(hi,p)))/(hi-lo)*(h-top-bottom)
        pts.append(f'{px:.2f},{py:.2f}')
    grid=''
    for db in [-100,-70,-40]:
        yy=top+(hi-db)/(hi-lo)*(h-top-bottom)
        grid+=f'<line x1="{left}" y1="{yy:.2f}" x2="782" y2="{yy:.2f}" stroke="#dfdfd4" stroke-width=".7"/><text x="50" y="{yy+3:.2f}" text-anchor="end">{db}</text>'
    for f in [-500,-250,0,250,500]:
        xx=left+(f+500)/1000*(w-left-right)
        grid+=f'<line x1="{xx}" y1="{top}" x2="{xx}" y2="89" stroke="#dfdfd4" stroke-width=".7"/><text x="{xx}" y="108" text-anchor="middle">{f}</text>'
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}"><style>text{{font:10px monospace;fill:#72776d}}</style>{grid}<polyline fill="none" stroke="{COLORS[name]}" stroke-width="1.7" points="{" ".join(pts)}"/><text x="69" y="24" fill="{COLORS[name]}">{title}</text></svg>'
for name,title in [('mixture','ORIGINAL MIXTURE'),('target','RECOVERED TARGET'),('residual','UNASSIGNED RESIDUAL')]:
    (ASSETS/f'{name}-spectrum.svg').write_text(svg_plot(name,title))
# Subtle original 20-second sine score. This is musical sound design, never RF audio.
sr=48000; duration=20
with wave.open(str(ASSETS/'instrument-score.wav'),'wb') as f:
    f.setnchannels(2);f.setsampwidth(2);f.setframerate(sr)
    pcm=bytearray()
    for n in range(sr*duration):
        t=n/sr
        fade=min(1,t/1.1)*min(1,max(0,(duration-t)/1.4))
        bed=.055*(math.sin(2*math.pi*110*t)+.44*math.sin(2*math.pi*164.8138*t)+.22*math.sin(2*math.pi*220*t))
        bed*=.86+.14*math.sin(2*math.pi*.13*t)
        hit=0
        for cue in [3,9,16]:
            z=t-cue
            if 0<=z<1.4:hit+=.035*math.sin(2*math.pi*440*z)*math.exp(-5*z)*min(1,z/.012)
        val=max(-1,min(1,(bed+hit)*fade))
        left=int(val*32767); right=int((val*.97)*32767)
        pcm.extend(struct.pack('<hh',left,right))
    f.writeframes(pcm)
print('Regenerated measured PSD graphics and original ambient score.')
# Actual time-frequency values, with the source UI's percentile and color map.
import zlib
rows=x['waterfall']['power_db']; allv=sorted(p for row in rows for p in row)
lo=allv[int(len(allv)*.04)];hi=max(lo+1,allv[int(len(allv)*.995)])
stops=[(0,(18,36,47)),(.24,(23,64,81)),(.48,(39,120,140)),(.68,(136,154,124)),(.84,(215,141,77)),(1,(249,232,171))]
def color(p):
    v=max(0,min(1,(p-lo)/(hi-lo))); s=0
    while s<len(stops)-2 and v>stops[s+1][0]:s+=1
    t=(v-stops[s][0])/(stops[s+1][0]-stops[s][0])
    return bytes(round(a+(b-a)*t) for a,b in zip(stops[s][1],stops[s+1][1]))
def chunk(tag,data):return struct.pack('>I',len(data))+tag+data+struct.pack('>I',zlib.crc32(tag+data)&0xffffffff)
w,h=len(rows[0]),len(rows)
raw=b''.join(b'\x00'+b''.join(color(v) for v in row) for row in rows)
(ASSETS/'field.png').write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',w,h,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b''))
print('Regenerated actual reference waterfall.')
