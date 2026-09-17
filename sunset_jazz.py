"""Reproducible image -> existing TorchModel -> neural features -> jazz pipeline.
Proxy channels are functional assignments, not anatomical photoreceptor claims.
RGB contains no measured UV; UV is explicitly unavailable (zero).
"""
from pathlib import Path
import argparse, hashlib, json, sys, time, subprocess, shutil
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image
from scipy.ndimage import uniform_filter, sobel
import mido
import soundfile as sf
ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data/sunset_jazz'
sys.path.insert(0, str(ROOT / 'code'))
from run_pytorch import TorchModel, MODEL_PARAMS, DT, get_hash_tables, get_weights
import torch
SEED = 20260917
CHANNELS = ['luminance','warm','cool','green','edge_contrast','left','right','upper','lower']
MAPPINGS = {'total_spikes':'bounded note and bass density', 'active_neurons':'3-5 voice chord richness',
 'synchrony':'accent strength and rhythmic regularity', 'temporal_burstiness':'swing 62-67% and syncopation',
 'entropy':'harmonic tension', 'response_latency':'phrase entry delay', 'persistent_activity':'sustain',
 'left_right_balance':'bounded stereo pan', 'trial_variance':'timing looseness',
 'recurrent_fraction':'register width and orchestration', 'population_rank':'motif scale degrees',
 'burst_phase':'phrase onset', 'firing_rate_variance':'harmonic ambiguity'}
def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def discover():
 images=[]
 for p in sorted(ROOT.iterdir(),key=lambda p:p.name.lower()):
  if p.suffix.lower() in {'.png','.jpg','.jpeg','.webp'} and not any(s in p.stem.lower() for s in ['plot','icon','generated','spectrum']):
   with Image.open(p) as im:
    if min(im.size)>=128: images.append(p)
 if len(images)<4: raise RuntimeError('Need four source sunset images in repository root; found '+str(len(images)))
 for i,p in enumerate(images[:4]): print(f'Source image {i+1}: {p.name}',flush=True)
 return images[:4]
def robust(a):
 lo,hi=np.percentile(a,[2,98]); return np.clip((a-lo)/max(hi-lo,1e-6),0,1)
def encode(p):
 rgb=np.asarray(Image.open(p).convert('RGB').resize((16,12),Image.Resampling.LANCZOS),dtype=float)/255
 lum=rgb @ np.array([.2126,.7152,.0722]); contrast=robust(abs(lum-uniform_filter(lum,3)))
 edge=robust(np.hypot(sobel(lum,0),sobel(lum,1)))
 y,x=np.mgrid[:12,:16]; l=robust(lum)
 channels=[l, np.clip(rgb[:,:,0]-.5*(rgb[:,:,1]+rgb[:,:,2]),0,1),rgb[:,:,2],rgb[:,:,1],
  .5*(edge+contrast),l*(1-x/15),l*x/15,l*(1-y/11),l*y/11]
 return np.stack(channels).reshape(9,-1), {'uv_available':False,'uv_input':0,'mean_luminance':float(lum.mean()),'grid':[16,12]}
def select_visual_input_neurons(n,count):
 # Dataset inspected: only Completed column. Uniform valid IDs, fixed seed.
 return np.random.default_rng(SEED).choice(n,min(count,n),replace=False)
def feature(df,n,trials,duration,selected):
 counts=np.bincount(df.neuron_index.to_numpy(),minlength=n); rates=counts/(trials*duration)
 bins=np.arange(0,duration*1000+10,10); hist=np.histogram(df.time_ms,bins)[0]
 non=counts[counts>0]; probs=non/max(non.sum(),1)
 recurrent=~df.neuron_index.isin(selected)
 tc=np.bincount(df.trial.to_numpy(),minlength=trials)
 left=counts[selected[np.arange(len(selected))%9==5]].sum(); right=counts[selected[np.arange(len(selected))%9==6]].sum()
 pops=np.bincount(df.loc[recurrent,'neuron_index'].to_numpy()%7,minlength=7)
 # Synchrony is excess population-bin variability proxy, not pairwise synchrony.
 f=dict(total_spikes=len(df),active_neurons=int((counts>0).sum()),mean_firing_rate=float(rates.mean()),
 firing_rate_variance=float(rates.var()),temporal_burstiness=float(hist.std()/max(hist.mean(),1)),
 synchrony=float(hist.var()/max(hist.mean(),1)),response_latency=float(df.time_ms.min()) if len(df) else duration*1000,
 early_response_strength=float((df.time_ms<duration*250).sum()/(trials*duration*.25)),
 late_response_strength=float((df.time_ms>=duration*750).sum()/(trials*duration*.25)),
 persistent_activity=float((df.time_ms>=duration*850).sum()/max(len(df),1)/.15),
 trial_variance=float(tc.var()/max(tc.mean()**2,1)),entropy=float(-(probs*np.log(probs)).sum()/np.log(n)),
 left_right_balance=float((right-left)/max(right+left,1)),recurrent_fraction=float(recurrent.mean()),
 burst_phase=float(np.argmax(hist)/max(len(hist)-1,1)))
 f.update({f'population_{i}':int(v) for i,v in enumerate(pops)})
 return f

def simulate(images,args):
 device='cuda' if torch.cuda.is_available() else 'cpu'; torch.set_num_threads(args.threads)
 f2i,i2f=get_hash_tables(ROOT/'data/2025_Completeness_783.csv'); n=len(f2i)
 selected=select_visual_input_neurons(n,args.inputs); print(f'Device: {device}; neurons: {n}; inputs: {len(selected)}',flush=True)
 con=pq.read_metadata(ROOT/'data/2025_Connectivity_783.parquet'); print('Connectivity rows:',con.num_rows,flush=True)
 config={'duration':args.duration,'trials':args.trials,'inputs':args.inputs,'seed':SEED,'device':device,'dt_ms':DT,
 'model_sha256':digest(ROOT/'code/run_pytorch.py'),'connectivity_sha256':digest(ROOT/'data/2025_Connectivity_783.parquet'),
 'completeness_sha256':digest(ROOT/'data/2025_Completeness_783.csv'),'pipeline_sha256':digest(Path(__file__)), 'cpu_adapter_sha256':digest(ROOT/'sunset_cpu_weights.py')}
 weights=None; features=[]; seeds=[]; encodings=[]
 for image_i,p in enumerate(images):
  enc,meta=encode(p); encodings.append(meta)
  key=dict(config,image_sha256=digest(p)); out=DATA/f'image_{image_i+1:02d}_spikes.parquet'; cfg=out.with_suffix('.json')
  seed=SEED+1000*(image_i+1); seeds.append(seed)
  if args.resume and out.exists() and cfg.exists() and json.loads(cfg.read_text())==key:
   print('Verified cache:',out.name,flush=True); df=pd.read_parquet(out)
  else:
   if weights is None:
    weights=get_weights(ROOT/'data/2025_Connectivity_783.parquet',ROOT/'data/2025_Completeness_783.csv',DATA,csr=True).to(device)
    if device=='cpu':
     from sunset_cpu_weights import CPUEventWeights
     weights=CPUEventWeights(weights)
   model=TorchModel(args.trials,n,DT,MODEL_PARAMS,weights,exc_indices=selected.tolist(),device=device)
   state=model.state_init(); rates=torch.zeros(args.trials,n,device=device)
   k=np.arange(len(selected)); values=np.clip(5+130*enc[k%9,((k//9)*37)%192]+25*enc[4,((k//9)*37)%192],0,180)
   rates[:,selected]=torch.tensor(values,dtype=torch.float32,device=device)
   gen=torch.Generator(device=device).manual_seed(seed); chunks=[]; start=time.monotonic(); steps=round(args.duration*1000/DT)
   with torch.no_grad():
    for step in range(steps):
     # Last 15% has no external drive: measure recurrent persistence.
     if step==int(.85*steps): rates.zero_()
     state=model(rates,*state,generator=gen)
     b,neur=(state[2]>0).nonzero(as_tuple=True)
     if len(b): chunks.append(np.column_stack((np.full(len(b),step*DT),b.cpu().numpy(),neur.cpu().numpy())))
     if (step+1)%500==0: print(f'Image {image_i+1}: {step+1}/{steps} steps; {time.monotonic()-start:.1f}s',flush=True)
   a=np.concatenate(chunks) if chunks else np.empty((0,3))
   df=pd.DataFrame({'time_ms':a[:,0],'trial':a[:,1].astype('int32'),'neuron_index':a[:,2].astype('int32')})
   df['flywire_id']=[i2f[int(i)] for i in df.neuron_index]; df['image_index']=image_i+1; df['image_filename']=p.name
   df.to_parquet(out,index=False); cfg.write_text(json.dumps(key,indent=2))
  if not len(df): raise RuntimeError('No spikes: refusing to fabricate neural features')
  f=feature(df,n,args.trials,args.duration,selected); features.append(f)
  print(f'Image {image_i+1}: {f["total_spikes"]} spikes, {f["active_neurons"]} active neurons',flush=True)
 raw=pd.DataFrame(features); norm=(raw-raw.min())/(raw.max()-raw.min()).replace(0,np.nan); norm=norm.fillna(.5)
 pd.concat([pd.Series([p.name for p in images],name='image_filename'),raw,norm.add_prefix('normalized_')],axis=1).to_csv(DATA/'features.csv',index=False)
 return features,norm.to_dict('records'),dict(config,brain_backend='code/run_pytorch.py:TorchModel; CPU event-sparse recurrent matmul adapter' if device=='cpu' else 'code/run_pytorch.py:TorchModel',visual_input_type='proxy; no visual annotations in completeness dataset',
 number_of_visual_input_neurons=len(selected),visual_input_ids=[int(i2f[int(i)]) for i in selected],image_batch_seeds=seeds,
 trial_seed_policy='One seeded generator per image; independent draws across batched trials',visual_encoding=encodings,model_params=MODEL_PARAMS,
 total_network_neurons=n,feature_definitions={'synchrony':'10 ms population spike count Fano factor','persistent_activity':'fraction of spikes during final 15% with external drive off, divided by .15','population_groups':'non-input neuron tensor index modulo 7','left_right_balance':'response of assigned proxy field channels; not anatomical hemispheres'})

CHORDS={'Dm9':(38,[53,60,64,69,76]),'G13':(43,[53,59,64,69,76]),'Cmaj9':(36,[52,59,62,67,74]),
 'Fmaj9':(41,[52,57,60,67,74]),'Bbmaj7':(46,[53,57,62,65,72]),'Em7b5':(40,[55,58,62,65,69]),
 'A7alt':(45,[55,61,65,70,74]),'Am7':(45,[55,60,64,67,74]),'Gm9':(43,[53,57,62,65,69]),'Dm6/9':(38,[53,59,64,69,74])}
def compose(images,raw,norm):
 rng=np.random.default_rng(SEED); mid=mido.MidiFile(ticks_per_beat=960); end=160*960
 specs=[('Piano',0,0),('Acoustic bass',1,32),('Vibraphone',2,11),('Muted trumpet',3,59),('Jazz drums',9,0)]
 events=[[] for _ in specs]; meta=[(0,mido.MetaMessage('set_tempo',tempo=750000)),(0,mido.MetaMessage('time_signature',numerator=4,denominator=4))]
 def note(t,p,on,dur,vel):
  off=min(158.2,on+dur); on=max(0,on)
  if off<=on:return
  ch=specs[t][1]; events[t].extend([(round(on*960),mido.Message('note_on',channel=ch,note=int(p),velocity=int(np.clip(vel,1,110)))),(round(off*960),mido.Message('note_off',channel=ch,note=int(p),velocity=0))])
 for i,p in enumerate(images):meta.append((i*40*960,mido.MetaMessage('marker',text=f'SUNSET_{i+1}: {p.name}')))
 scale=np.array([62,64,65,67,69,71,72]); motif=np.argsort([raw[0][f'population_{i}'] for i in range(7)])[:5]
 prev=np.array([53,60,64,69]); harmony=[]
 for bar in range(40):
  s=bar//10; f=norm[s]; r=raw[s]; local=bar%10; beat=bar*4; act=f['total_spikes']; persist=f['persistent_activity']; sync=f['synchrony']; burst=f['temporal_burstiness']
  arc=[.65,.85,1,.68][s]*(1 if bar<36 else .72); swing=.62+.05*burst; loosen=.005+.018*f['trial_variance']
  pool=['Dm9','G13','Cmaj9','Fmaj9','Am7','Gm9'];
  if f['entropy']>.5:pool+=['Bbmaj7','Em7b5']
  if f['firing_rate_variance']>.65:pool+=['Gm9','Fmaj9']
  name=pool[int((r[f'population_{local%7}']+local*3)%len(pool))]
  if local==0:name='Dm9' if s in (0,3) else ('Fmaj9' if s==1 else 'G13')
  if local==8 and sync>.4:name='A7alt' if f['entropy']>.5 else 'G13'
  if bar==36:name='Em7b5'
  if bar==37:name='A7alt'
  if bar>=38:name='Dm6/9'
  root,voices=CHORDS[name]; count=3+int(2*f['active_neurons']); voices=np.array(voices[:count])
  choices=[voices+shift for shift in [-12,0,12] if min(voices+shift)>=48 and max(voices+shift)<=81]
  voices=min(choices,key=lambda v:abs(np.mean(v)-np.mean(prev))); prev=voices; harmony.append(name)
  for t,(_,ch,_) in enumerate(specs):
   pan=int(np.clip([52,61,78,69,64][t]+12*r['left_right_balance'],30,98))
   events[t].append((beat*960,mido.Message('control_change',channel=ch,control=10,value=pan)))
  onset=beat+(0 if local==0 else (.15+.5*burst))+rng.uniform(-loosen,loosen)
  if bar!=39:
   for j,p in enumerate(voices): note(0,p,onset+j*.018,1.7+1.4*persist if bar<38 else 5.4,47+12*act+4*arc-j)
  if act>.5 and local%3==1 and bar<36:
   for j,p in enumerate(voices[:3]):note(0,p,beat+2+swing+j*.012,.65,43+10*act)
  if bar>=2 and bar!=39:
   note(1,root,beat+.025,2.8 if act<.5 else 1.65,55+12*arc)
   if act>.3 and bar<37:note(1,min(root+7,55),beat+2.02,1.6,50+8*act)
   if s==2 and act>.65 and local%3==2:note(1,root+1,beat+3.05,.7,47)
  if local in [0,3,6] and bar<37:note(2,int(voices[-1]),beat+1.0+swing,2+persist,40+9*act)
  if local in [1,2,4,6,7] and bar<36:
   # Neural rank-derived motif, maintained and varied across sections.
   rank=np.argsort([r[f'population_{i}'] for i in range(7)]); degrees=motif.copy()
   if local in [4,7]:degrees[-2:]=rank[:2]
   length=3+int(2*act); entry=.15+.5*f['response_latency']+.35*r['burst_phase']
   for j in range(length):
    on=beat+entry+j*.66+(swing-.5 if j%2 else 0)+rng.uniform(-loosen,loosen)
    pitch=int(scale[degrees[j%5]])
    if f['recurrent_fraction']>.7 and j==length-1:pitch=min(81,pitch+7)
    track=3 if s in [1,2] and local in [2,6,7] else 0
    note(track,pitch,on,.38+.5*persist+( .35 if j==length-1 else 0),53+14*act+4*arc)
  if 10<=bar<37:
   for off in [0,2,2+swing]:note(4,51,beat+off,.16,29+10*sync+(5 if off==2 else 0))
   for off in [1,3]:note(4,44,beat+off,.1,24+7*sync)
   if local%2==0:note(4,36,beat,.12,23)
   if burst>.65 and local==7:note(4,38,beat+3+swing,.12,25)
  if bar==38: note(2,74,beat+.05,5.1,37)
 conductor=mido.MidiTrack(); mid.tracks.append(conductor); conductor.append(mido.MetaMessage('track_name',name='Sunset Fly Jazz - 40 bars'))
 def write(track,ev):
  last=0
  for tick,msg in sorted(ev,key=lambda e:(e[0],0 if e[1].type=='note_off' else 1)):
   track.append(msg.copy(time=tick-last));last=tick
  track.append(mido.MetaMessage('end_of_track',time=end-last))
 write(conductor,meta)
 for i,(name,ch,program) in enumerate(specs):
  # Avoid overlapping same-pitch voices on one MIDI channel.
  ev=sorted(events[i],key=lambda e:(e[0],0 if e[1].type=='note_off' else 1)); pending={}; paired=[]
  for tick,msg in ev:
   if msg.type=='note_on':
    key=(msg.channel,msg.note); pending.setdefault(key,[]).append((tick,msg))
   elif msg.type=='note_off':
    key=(msg.channel,msg.note); on,onmsg=pending[key].pop(0); paired.append([on,tick,onmsg,msg])
  for key in {(x[2].channel,x[2].note) for x in paired}:
   notes=sorted([x for x in paired if (x[2].channel,x[2].note)==key],key=lambda x:x[0])
   for first,second in zip(notes,notes[1:]):first[1]=min(first[1],second[0])
  events[i]=[(tick,msg) for tick,msg in ev if msg.type not in ('note_on','note_off')]
  for on,off,onmsg,offmsg in paired:
   if off>on:events[i].extend([(on,onmsg),(off,offmsg)])
  tr=mido.MidiTrack();mid.tracks.append(tr);tr.append(mido.MetaMessage('track_name',name=name));tr.append(mido.Message('program_change',channel=ch,program=program));tr.append(mido.Message('control_change',channel=ch,control=7,value=[88,95,70,73,68][i]));write(tr,events[i])
 mid.save(ROOT/'sunset_fly_jazz.mid');return harmony

def verify_midi():
 mid=mido.MidiFile(ROOT/'sunset_fly_jazz.mid');count=0;markers=[];tempos=[];meters=[]
 for tr in mid.tracks:
  active={}
  for m in tr:
   assert m.time>=0
   if m.type=='set_tempo':tempos.append(m.tempo)
   if m.type=='time_signature':meters.append((m.numerator,m.denominator))
   if m.type=='marker':markers.append(m.text)
   if m.type in ['note_on','note_off']:
    assert 0<=m.note<=127;k=(m.channel,m.note)
    if m.type=='note_on' and m.velocity>0:
     assert 1<=m.velocity<=127;active[k]=active.get(k,0)+1;count+=1
    else:assert active.get(k,0)>0;active[k]-=1
  assert not any(active.values()),'stuck notes'
 assert abs(mid.length-120)<.001 and tempos==[750000] and meters==[(4,4)] and len(markers)==4 and count>100
 return dict(midi_duration_seconds=mid.length,midi_note_count=count,midi_tracks=[next(m.name for m in t if m.type=='track_name') for t in mid.tracks])
def render():
 fluid=ROOT/'.venv/native/bin/fluidsynth'; ffmpeg=ROOT/'.venv/native/bin/ffmpeg'
 if not fluid.exists():fluid=Path(shutil.which('fluidsynth') or 'MISSING_FLUIDSYNTH')
 if not ffmpeg.exists():
  import imageio_ffmpeg
  ffmpeg=Path(imageio_ffmpeg.get_ffmpeg_exe())
 candidates=[('MuseScore_General.sf2','https://ftp.osuosl.org/pub/musescore/soundfont/MuseScore_General/MuseScore_General.sf2','MuseScore_General_License.md'),('GeneralUser_GS.sf2','https://github.com/mrbumpy409/GeneralUser-GS','GeneralUser_GS_LICENSE.txt')]
 tests=[];selected=None
 for name,url,lic in candidates:
  font=ROOT/'soundfonts'/name
  if not font.exists() or not (font.parent/lic).exists():continue
  raw=DATA/(font.stem+'_raw.wav');log=DATA/(font.stem+'_render.log')
  cmd=[str(fluid),'-ni','-g','0.45','-r','48000','-o','synth.reverb.active=1','-o','synth.reverb.room-size=0.35','-o','synth.reverb.level=0.18','-o','synth.chorus.active=0','-T','wav','-O','float','-F',str(raw),str(font),str(ROOT/'sunset_fly_jazz.mid')]
  with log.open('w') as f: result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
  if result.returncode:tests.append(dict(font=name,success=False));continue
  audio,sr=sf.read(raw);peak=float(abs(audio).max());rms=float(np.sqrt(np.mean(audio**2)))
  ok=audio.ndim==2 and audio.shape[1]==2 and rms>1e-5 and np.isfinite(audio).all()
  tests.append(dict(font=name,success=bool(ok),peak=peak,rms=rms))
  if ok and selected is None:selected=(name,url,lic,raw)
 if selected is None:raise RuntimeError('No licensed SoundFont rendered successfully')
 name,url,lic,raw=selected
 # Measure loudness, then use a single static gain bounded by peak headroom.
 proc=subprocess.run([str(ffmpeg),'-hide_banner','-i',str(raw),'-af','atrim=duration=120,loudnorm=I=-16:TP=-1:LRA=11:print_format=json','-f','null','-'],capture_output=True,text=True,check=True)
 measurement=json.loads(proc.stderr[proc.stderr.rfind('{'):proc.stderr.rfind('}')+1]);audio,sr=sf.read(raw)
 audio=audio[:48000*120];audio=np.pad(audio,((0,max(0,48000*120-len(audio))),(0,0)))
 peak=float(abs(audio).max());gain_db=min(-16-float(measurement['input_i']),-1-20*np.log10(max(peak,1e-12)))
 audio*=10**(gain_db/20);audio[:2400]*=np.linspace(0,1,2400)[:,None];audio[-144000:]*=np.linspace(1,0,144000)[:,None]
 sf.write(ROOT/'sunset_fly_jazz.wav',audio,48000,subtype='PCM_24')
 a,sr=sf.read(ROOT/'sunset_fly_jazz.wav');peak=float(abs(a).max());rms=float(np.sqrt(np.mean(a*a)))
 assert sr==48000 and a.shape==(5760000,2) and 1e-5<rms<.5 and 0<peak<1 and abs(a[-1]).max()<1e-6
 return dict(soundfont=name,soundfont_source=url,soundfont_license='soundfonts/'+lic,soundfont_sha256=digest(ROOT/'soundfonts'/name),soundfont_tests=tests,
 soundfont_selection='First successful licensed font in preferred order; objective rendering tests, no subjective listening claim',audio_renderer='FluidSynth',sample_rate=sr,channels=2,wav_duration_seconds=len(a)/sr,
 audio_peak_dbfs=20*np.log10(peak),audio_rms_dbfs=20*np.log10(rms),silent_frame_percentage=float(100*np.mean(np.max(abs(a),axis=1)<1e-5)),mastering_gain_db=gain_db,raw_loudness_measurement=measurement)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--trials',type=int,default=8);parser.add_argument('--duration',type=float,default=1.0);parser.add_argument('--inputs',type=int,default=1000);parser.add_argument('--threads',type=int,default=2);parser.add_argument('--resume',action='store_true');parser.add_argument('--render-only',action='store_true');args=parser.parse_args()
 DATA.mkdir(parents=True,exist_ok=True)
 if args.render_only:
  provenance=json.loads((DATA/'composition_provenance.json').read_text())
 else:
  images=discover();raw,norm,provenance=simulate(images,args);harmony=compose(images,raw,norm)
  provenance.update(source_images=[dict(filename=p.name,sha256=digest(p)) for p in images],base_seed=SEED,simulation_duration_per_image=args.duration,trials_per_image=args.trials,
   neural_features=raw,normalized_neural_features=norm,feature_music_mapping=MAPPINGS,tempo_bpm=80,time_signature='4/4',bars=40,duration_seconds=120,harmony=harmony,
   scientific_limitations='Approximate RGB encoding; no UV inference; proxy functional neurons, not a complete Drosophila retina. Input files have ChatGPT Image filenames; photographic origin is not asserted.')
  (DATA/'composition_provenance.json').write_text(json.dumps(provenance,indent=2))
 provenance.update(verify_midi());provenance.update(render());provenance['output_sha256']={name:digest(ROOT/name) for name in ['sunset_fly_jazz.mid','sunset_fly_jazz.wav']}
 (ROOT/'sunset_fly_jazz_provenance.json').write_text(json.dumps(provenance,indent=2,allow_nan=False));print(json.dumps({k:v for k,v in provenance.items() if k in ['device','trials_per_image','simulation_duration_per_image','midi_note_count','midi_duration_seconds','soundfont','wav_duration_seconds','audio_peak_dbfs','audio_rms_dbfs','silent_frame_percentage']},indent=2),flush=True)
if __name__=='__main__':main()
