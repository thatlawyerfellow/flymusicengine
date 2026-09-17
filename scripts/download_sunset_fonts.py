"""Fetch original upstream GM SoundFonts and licenses; no global installation."""
import concurrent.futures, subprocess, pathlib, struct
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'soundfonts';OUT.mkdir(exist_ok=True)
MUSE='https://ftp.osuosl.org/pub/musescore/soundfont/MuseScore_General/'
GENERAL='https://raw.githubusercontent.com/mrbumpy409/GeneralUser-GS/main/'
def valid(p):
 if not p.exists():return False
 with p.open('rb') as f:head=f.read(12)
 return len(head)==12 and head[:4]==b'RIFF' and head[8:]==b'sfbk' and p.stat().st_size==struct.unpack('<I',head[4:8])[0]+8
def download(url,p):
 temp=p.with_suffix(p.suffix+'.partial')
 subprocess.run(['curl','-fsSL','--retry','3','--max-time','300',url,'-o',str(temp)],check=True);temp.replace(p)
def muse():
 target=OUT/'MuseScore_General.sf2'
 if valid(target):return
 size=215614036;chunk=4*1024*1024
 def part(i):
  a=i*chunk;b=min(size,a+chunk)-1;p=OUT/f'.musepart{i:03d}'
  subprocess.run(['curl','-fsSL','--retry','3','--max-time','300','-r',f'{a}-{b}',MUSE+'MuseScore_General.sf2','-o',str(p)],check=True)
  if p.stat().st_size!=b-a+1:raise RuntimeError('Unexpected range response')
  return p
 with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:parts=list(pool.map(part,range((size+chunk-1)//chunk)))
 temp=target.with_suffix('.partial')
 with temp.open('wb') as f:
  for p in parts:f.write(p.read_bytes());p.unlink()
 if not valid(temp):raise RuntimeError('Invalid SoundFont RIFF length')
 temp.replace(target)
if __name__=='__main__':
 muse()
 target=OUT/'GeneralUser_GS.sf2'
 if not valid(target):download(GENERAL+'GeneralUser-GS.sf2',target)
 if not valid(target):raise RuntimeError('Invalid GeneralUser SoundFont')
 for url,name in [(MUSE+'MuseScore_General_License.md','MuseScore_General_License.md'),(GENERAL+'documentation/LICENSE.txt','GeneralUser_GS_LICENSE.txt')]:
  if not (OUT/name).exists():download(url,OUT/name)
 print('Both licensed SoundFonts are present and RIFF lengths verified.')
