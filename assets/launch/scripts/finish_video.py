"""Extract selected poster, bake frame zero, copy public assets. Requires ffmpeg/ffprobe."""
from pathlib import Path
import subprocess, shutil
ROOT=Path(__file__).resolve().parents[1]
raw=ROOT/'launch-source.mp4';poster=ROOT/'brag.jpg';final=ROOT/'brag.mp4'
if poster.exists() or final.exists():raise SystemExit('Refusing to replace existing poster or final. Preserve earlier output before rebuilding.')
# Settled first-scene hook is selected because it carries the product and true field.
subprocess.run(['ffmpeg','-v','error','-ss','1.8','-i',str(raw),'-frames:v','1','-q:v','2',str(poster)],check=True)
subprocess.run(['ffmpeg','-v','error','-i',str(raw),'-i',str(poster),'-filter_complex',"[0:v][1:v]overlay=0:0:enable='eq(n,0)'[v]",'-map','[v]','-map','0:a?','-c:v','libx264','-threads','2','-crf','18','-preset','medium','-pix_fmt','yuv420p','-c:a','copy','-movflags','+faststart',str(final)],check=True)
subprocess.run(['ffmpeg','-v','error','-threads','2','-i',str(final),'-f','null','-'],check=True)
for src,dst in [(final,ROOT.parent/'launch.mp4'),(poster,ROOT.parent/'launch.jpg')]:
    if dst.exists():raise SystemExit(f'Refusing to replace existing {dst.name}')
    shutil.copyfile(src,dst)
print('Poster-baked final decoded and copied to public assets.')
