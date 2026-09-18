"""Compose and render Days: seven developing, neural-derived scene studies.

The composer reads recorded spike features, never pixels. Long form comes from
heads, contrasting solo sections, interludes, thematic returns, and codas; audio
is synthesized from the generated MIDI, never looped from an existing recording.
"""
from pathlib import Path
from itertools import product
from collections import defaultdict
import argparse, hashlib, json, math, subprocess
import numpy as np
import mido
import soundfile as sf
ROOT=Path(__file__).resolve().parent
ALBUM=ROOT/'albums/Days_v2'
RATE=48000
SEED=20260918
TICKS=960
# Root pitch class, third/seventh and colour tones for rootless comping.
CHORDS={
 'Fmaj9':(5,[9,4,7,0]),'Cmaj9':(0,[4,11,2,7]),'Bbmaj9':(10,[2,9,0,5]),
 'Gmaj9':(7,[11,6,9,2]),'Ebmaj9':(3,[7,2,5,10]),
 'Dm9':(2,[5,0,4,9]),'Dm6/9':(2,[5,11,4,9]),'Gm9':(7,[10,5,9,2]),
 'Am9':(9,[0,7,11,4]),'Am7':(9,[0,7,4,11]),'Em7':(4,[7,2,11,6]),
 'Em7b5':(4,[7,2,10,5]),'Bbm6':(10,[1,7,5,0]),
 'G13':(7,[11,5,9,4]),'C13':(0,[4,10,2,9]),'D13':(2,[6,0,4,11]),
 'Eb13':(3,[7,1,5,0]),'A7sus':(9,[2,7,11,4]),
 'A7alt':(9,[1,7,10,5]),'D7alt':(2,[6,0,3,10]),
}
BASE_CONFIGS=[{'tonic': 5,
  'scale': [5, 7, 9, 11, 0, 2, 4],
  'progressions': [['Fmaj9', 'Fmaj9', 'Em7', 'A7alt', 'Dm9', 'G13', 'Gm9', 'C13'],
                   ['Fmaj9', 'Am9', 'Bbmaj9', 'Bbm6', 'Am7', 'D7alt', 'Gm9', 'C13'],
                   ['Dm9', 'G13', 'Cmaj9', 'Cmaj9', 'Em7', 'A7alt', 'Dm9', 'G13'],
                   ['Bbmaj9', 'Eb13', 'Am7', 'D7alt', 'Gm9', 'C13', 'Fmaj9', 'C13']]},
 {'tonic': 2,
  'scale': [2, 4, 5, 7, 9, 11, 0],
  'progressions': [['Dm9', 'G13', 'Dm9', 'G13', 'Em7b5', 'A7alt', 'Dm9', 'A7sus'],
                   ['Dm9', 'Dm9', 'Gm9', 'C13', 'Fmaj9', 'Bbmaj9', 'Em7b5', 'A7alt'],
                   ['Cmaj9', 'A7alt', 'Dm9', 'G13', 'Em7', 'A7sus', 'Dm9', 'A7alt'],
                   ['Am9', 'D13', 'Gmaj9', 'Cmaj9', 'Fmaj9', 'Em7b5', 'A7alt', 'A7alt']]},
 {'tonic': 2,
  'scale': [2, 4, 5, 7, 9, 11, 0],
  'progressions': [['Dm9', 'Dm9', 'Bbmaj9', 'Gm9', 'Em7b5', 'A7alt', 'Dm6/9', 'Dm6/9'],
                   ['Fmaj9', 'Am7', 'Bbmaj9', 'Eb13', 'Dm9', 'Gm9', 'Em7b5', 'A7alt'],
                   ['Gm9', 'C13', 'Fmaj9', 'Bbmaj9', 'Em7b5', 'A7alt', 'Dm9', 'Dm9'],
                   ['Dm6/9', 'Am7', 'Bbmaj9', 'Bbm6', 'Dm9', 'G13', 'Em7b5', 'A7alt']]}]
from copy import deepcopy
# Exact target duration is set by bar count and tempo, never by stretching audio.
TRACKS=[
 ('Sunrise','sunrise',280,84,98,0,59,.62,[8,16,24,16,10,16,8]),
 ('Traffic','traffic',240,120,120,1,65,.66,[8,24,24,16,16,24,8]),
 ('High Noon','high_noon',288,100,120,1,66,.64,[8,24,24,16,16,24,8]),
 ('Rainy Evening Drizzle','rainy_evening_drizzle',320,72,96,2,64,.61,[8,16,24,16,8,16,8]),
 ('Sunset','sunset',312,80,104,2,59,.62,[8,24,24,16,8,16,8]),
 ('Crescent Moon','crescent_moon',328,60,82,2,11,.61,[8,16,16,10,8,16,8]),
 ('Bombolini','bombolini',272,120,136,0,71,.66,[8,24,32,24,16,24,8]),
]
SECTION_TITLES=[
 ['First light','Open windows','Morning conversation','Reflections','A breath of air','Full daylight','The road ahead'],
 ['Ignition','Crosswalk theme','Lane changes','Signals','Under the bridge','Green light','Turn the corner'],
 ['Overhead sun','Short shadows','Hot pavement','Bright glass','Shade','The square awakens','After noon'],
 ['First drops','Window theme','Pavement reflections','Drifting umbrellas','Shelter','The rain returns','Last drops'],
 ['Low sun','Amber theme','Long shadows','Copper water','The horizon','Afterglow','Violet air'],
 ['Blue hour','Silver crescent','Night windows','Still water','Dark sky','Moonlight returns','Goodnight'],
 ['Bakery door','Sugar theme','Custard conversation','Espresso break','Crumbs','One more bite','Sweet goodbye'],
]
CONFIGS=[]
for i,(title,scene,duration,tempo,bars,base,lead,swing,lengths) in enumerate(TRACKS):
 cfg=deepcopy(BASE_CONFIGS[base]);cfg.update(title=title,scene=scene,slug=f'{i+1:02d}_{title.replace(" ","_")}',duration=duration,tempo=tempo,bars=bars,lead=lead,swing=swing)
 energies=([.32,.48,.59,.48,.25,.5,.22] if i in (3,4,5) else [.4,.65,.8,.68,.45,.72,.35])
 roles=['intro','head','piano','lead' if i!=6 else 'vibes','interlude','reprise','coda']
 cfg['sections']=list(zip(SECTION_TITLES[i],lengths,roles,energies));assert sum(lengths)==bars
 assert abs(bars*240/tempo-duration)<1e-6
 cfg['transpose']=[0,-2,0,-5,0,2,5][i]
 CONFIGS.append(cfg)

SPECS=[('Piano harmony',0,0,83,46),('Piano melody',1,0,88,57),('Upright bass',2,32,92,63),('Vibraphone',3,11,67,80),('Lead',4,59,70,69),('Electric piano',5,4,54,35),('Jazz drums',9,0,65,65)]
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def write_json(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False))
def closest(pcs,target,low,high):
 return min((n for n in range(low,high+1) if n%12 in pcs),key=lambda n:(abs(n-target),n))
def voicing(chord,previous,rich):
 pcs=CHORDS[chord][1][:3+int(rich>.32)]
 choices=[[n for n in range(50,77) if n%12==pc] for pc in pcs]
 candidates=[]
 for notes in product(*choices):
  v=sorted(notes)
  if len(set(v))!=len(v) or max(v)-min(v)>21 or any(b-a<2 for a,b in zip(v,v[1:])):continue
  cost=sum(min(abs(n-old) for old in previous) for n in v)+.2*abs(np.mean(v)-63)+.12*(max(v)-min(v))
  candidates.append((cost,v))
 return min(candidates,key=lambda x:x[0])[1]
def harmony_plan(cfg,raw,norm,profiles):
 result=[];section_meta=[];bar=0
 for index,(title,length,role,energy) in enumerate(cfg['sections']):
  section_meta.append(dict(title=title,start_bar=bar+1,bars=length,role=role,energy=energy))
  for local in range(length):
   profile=profiles[min(len(profiles)-1,bar*len(profiles)//cfg['bars'])]
   selector=(int(profile['population_activity'][(local//8)%7])//37+index+int(norm['entropy']*3))%len(cfg['progressions'])
   if role in ('head','reprise'):selector=0 if local%32<16 else 1
   # Two-bar harmonic rhythm in spacious passages; one bar in active solos.
   harmonic_step=local//2 if role in ('intro','interlude','coda') else local
   chord=cfg['progressions'][selector][harmonic_step%8]
   if role=='coda' and local>=length-8:
    cadence=['Gm9','Gm9','C13','C13','Fmaj9','Fmaj9','Fmaj9','Fmaj9'] if cfg['tonic']==5 else ['Gm9','Em7b5','A7alt','A7alt','Dm6/9','Dm6/9','Dm6/9','Dm6/9']
    chord=cadence[local-(length-8)]
   result.append(dict(bar=bar,chord=chord,section=index,local=local,role=role,energy=energy,profile=min(len(profiles)-1,bar*len(profiles)//cfg['bars'])))
   bar+=1
 assert len(result)==cfg['bars']
 return result,section_meta

def compose(index,neural,destination=None,perturb=False):
 cfg=CONFIGS[index];raw=neural['raw_features'][index];norm=neural['normalized_features'][index];profiles=neural['temporal_profiles'][index]
 rng=np.random.default_rng(SEED+index*100);total_beats=cfg['bars']*4;tempo=mido.bpm2tempo(cfg['tempo']);end=total_beats*TICKS
 harmony,sections=harmony_plan(cfg,raw,norm,profiles);notes=[[] for _ in SPECS];controls=[[] for _ in SPECS]
 # Album identity: neural population ranks from Sunrise, adapted to each mode.
 album_rank=np.argsort([-neural['raw_features'][0][f'population_{i}'] for i in range(7)])
 motif=[int(album_rank[i]) for i in [0,2,1,4,3,1,5,0]]
 activity=norm['total_spikes'];persistence=norm['persistent_activity'];synchrony=norm['synchrony'];burst=norm['temporal_burstiness'];variance=norm['trial_variance']
 swing=float(np.clip(cfg['swing']+.012*(burst-.5),.61,.67));jitter=.007+.018*variance
 last_pitch=67;previous=[53,60,64,69]
 def note(track,pitch,start,length,velocity):
  pitch+=cfg['transpose'] if track!=6 else 0
  start=max(0.,float(start));stop=min(float(start+length),total_beats-2.4)
  if stop<=start:return
  notes[track].append([round(start*TICKS),round(stop*TICKS),int(np.clip(pitch,0,127)),int(np.clip(round(velocity),1,110))])
 def swing_time(t):
  frac=t-math.floor(t)
  return t+(swing-.5 if abs(frac-.5)<.025 else 0)
 def melody_phrase(start_bar,role,energy,section,phrase_index):
  nonlocal last_pitch
  if role=='bass':return
  profile=profiles[min(len(profiles)-1,start_bar*len(profiles)//cfg['bars'])];rank=profile['population_rank'];burst_bins=np.array(profile['burst_profile'])
  if role in ('head','reprise'):
   degrees=motif.copy();rhythm=[0.,1.,2.5,4.,5.5,8.,10.,12.5]
   if phrase_index%4==3:degrees[-3:]=[2,1,0]
   if role=='reprise':degrees[3]=int(rank[phrase_index%7])
  else:
   count=7+int(8*energy+4*activity)
   if role in ('intro','interlude','coda'):count=3+int(4*energy)
   durations=rng.choice([.5,1.,1.5,2.],size=count,p=[.32,.4,.2,.08] if role in ('piano','lead','trading') else [.1,.35,.3,.25])
   rhythm=np.cumsum(np.r_[0,durations[:-1]]).tolist();rhythm=[t for t in rhythm if t<13.8]
   degrees=[int(rank[(j+phrase_index)%7]) for j in range(len(rhythm))]
   # Rest in every phrase; burst timing chooses a gentle phrase displacement.
   rhythm=[t+.25*int(np.argmax(burst_bins)%3) for t in rhythm]
  track={'vibes':3,'lead':4,'piano':1,'intro':1,'interlude':3,'coda':1}.get(role,4 if cfg['title'] in ('Traffic','High Noon') else 1)
  if role=='trading':track=1 if phrase_index%2==0 else 4
  if role in ('head','reprise') and phrase_index%4 in (2,3):track=4
  if role=='intro' and phrase_index%2:track=3
  low,high=(60,82) if track!=3 else (65,86)
  for j,offset in enumerate(rhythm):
   pos=start_bar*4+swing_time(offset)+rng.uniform(-jitter,jitter)
   bar=min(cfg['bars']-1,max(0,int(pos//4)));chord=harmony[bar]['chord'];root,pcs=CHORDS[chord]
   degree=degrees[j%len(degrees)]
   desired=60+cfg['scale'][degree]
   if desired<low:desired+=12
   if role not in ('head','reprise'):
    direction=1 if rank[(j+3)%7]%2 else -1
    desired=last_pitch+direction*(1+degree%4)
    desired=closest(set(cfg['scale']),desired,low,high)
   else:
    desired=min([desired+k for k in [-12,0,12] if low<=desired+k<=high],key=lambda n:abs(n-last_pitch))
   if offset%4<.1 or j==len(rhythm)-1:
    desired=closest(set(pcs+[root]),desired,low,high)
   pitch=closest(set(cfg['scale']+pcs),desired,max(low,last_pitch-7),min(high,last_pitch+7))
   if j==len(rhythm)-1 and role in ('head','reprise'):pitch=closest(set(pcs+[root]),pitch,low,high)
   gap=(rhythm[j+1]-offset) if j+1<len(rhythm) else 2.2
   length=min(max(.22,gap*.8),1.2+1.6*persistence)
   if j==len(rhythm)-1:length=1.4+1.2*persistence
   velocity=48+22*energy+8*activity+3*math.sin(j)- (5 if track==3 else 0)
   note(track,pitch,pos,length,velocity);last_pitch=pitch
  # Quiet responses between the main phrases, rather than continuous doubling.
  if role in ('lead','head','reprise') and phrase_index%3==1:
   chord=harmony[min(start_bar+3,len(harmony)-1)]['chord']
   note(3,closest(CHORDS[chord][1],76,67,84),start_bar*4+14.1,1.3,38+8*energy)

 for entry in harmony:
  bar=entry['bar'];beat=bar*4;role=entry['role'];local=entry['local'];energy=entry['energy'];section=entry['section'];chord=entry['chord'];profile=profiles[entry['profile']]
  section_length=cfg['sections'][section][1]
  # Each image also controls local dynamics, including the resolving coda.
  spike_counts=[x['spikes'] for x in profiles]
  image_strength=(profile['spikes']-min(spike_counts))/max(max(spike_counts)-min(spike_counts),1)
  energy*=.85+.3*image_strength
  # Phrase-scale dynamics plus a broad taper through the coda.
  energy*=.92+.08*math.sin(math.pi*(local%16)/16)
  if role=='coda':energy*=max(.3,1-local/max(section_length,1))
  voices=voicing(chord,previous,norm['active_neurons']);previous=voices
  pan_delta=round(10*raw['left_right_balance'])
  if local==0:
   for t,(_,channel,_,_,pan) in enumerate(SPECS):
    controls[t].append((beat*TICKS,mido.Message('control_change',channel=channel,control=10,value=int(np.clip(pan+pan_delta,28,98)))))
    controls[t].append((beat*TICKS,mido.Message('control_change',channel=channel,control=11,value=int(83+12*energy))))
  closing=bar>=cfg['bars']-4
  if role in ('intro','interlude','coda'):
   comp=[0.] if local%2==0 else ([] if energy<.4 else [1.5])
  elif role=='bass':comp=[.5] if local%2==0 else []
  else:
   patterns=[[0.,2.5],[.5],[1.,3.5],[0.],[1.5,3.]]
   select=(local+int(profile['population_activity'][local%7])//101)%len(patterns)
   comp=patterns[select]
   if role=='piano' and local%3==1:comp=comp[:1]
  if closing:comp=[0.] if bar==cfg['bars']-4 else []
  for onset in comp:
   sustain=(2.3+2*persistence) if role in ('intro','interlude','coda') else (.7+1.0*persistence)
   if closing:sustain=13.3
   for j,pitch in enumerate(voices):
    note(0,pitch,beat+swing_time(onset)+j*.016+rng.uniform(-jitter,jitter),sustain,39+17*energy+6*activity-j*.9)
  if bar>=4 and not (role=='interlude' and local%3==2):
   root=36+CHORDS[chord][0]
   if root>48:root-=12
   walking=(cfg['title'] in ('Traffic','High Noon','Bombolini') and role in ('piano','lead','trading','reprise') and local%16>=4) or (role=='bass')
   if closing:
    if bar==cfg['bars']-4:note(2,root,beat+.015,10.,48)
   elif walking:
    nextroot=36+CHORDS[harmony[min(bar+1,len(harmony)-1)]['chord']][0]
    pitches=[root,closest(CHORDS[chord][1],root+4,36,55),closest(CHORDS[chord][1]+[CHORDS[chord][0]],root+7,36,55),int(np.clip(nextroot-1,36,55))]
    for j,pitch in enumerate(pitches):note(2,pitch,beat+j+.008+rng.uniform(-.008,.008),.84,52+8*energy+(3 if j==0 else 0))
   else:
    note(2,root,beat+.015,2.9 if role in ('intro','interlude','coda') else 1.8,48+13*energy)
    if role not in ('intro','interlude','coda') and (local%3!=2 or activity>.65):note(2,min(55,root+7),beat+2.01,1.65,47+9*energy)
  if local%4==0 and not closing:
   if role!='intro' or local>=4:melody_phrase(bar,role,energy,section,local//4)
  # Occasional electric-piano washes, deliberately absent during piano solos.
  if role in ('interlude','coda') and local%8==0 and not closing:
   for pitch in voices[1:]:note(5,pitch,beat+.05,6.+2*persistence,32+7*energy)
  if role not in ('vibes','interlude') and local%8==6 and not closing:
   note(3,closest(CHORDS[chord][1],78,67,84),beat+2.5,2.1+1.5*persistence,35+7*energy)
  drums=bar>=8 and role!='interlude' and not closing and not(role=='intro' and local<8)
  if drums:
   gentle=role in ('coda','bass') or cfg['title'] in ('Sunset','Rainy Evening Drizzle','Crescent Moon')
   pattern=[0.,2.] if gentle else [0.,1.,1.5,2.,3.,3.5]
   if local%4==3:pattern=pattern[:-1]
   for off in pattern:note(6,51,beat+swing_time(off)+rng.uniform(-.008,.008),.11,23+10*energy+5*synchrony+(4 if off in [0,2] else -4))
   for off in [1.,3.]:note(6,44,beat+off,.08,21+9*energy)
   if local%2==0:note(6,36,beat,.1,19+9*energy)
   if not gentle and local%4==1:note(6,38,beat+2.5,.09,19+7*energy)
   if local%16==15 and burst>.35 and role in ('piano','lead','trading','reprise'):
    for j,pitch in enumerate([38,48,45]):note(6,pitch,beat+3+j/3,.1,23+3*j+7*energy)
 # End with the home sonority, voiced above the bass, leaving room for decay.
 finalpc=5 if cfg['tonic']==5 else 2
 note(1,72+finalpc,(cfg['bars']-4)*4+.08,10.,44)
 mid=mido.MidiFile(ticks_per_beat=TICKS,charset='utf-8');conductor=mido.MidiTrack();mid.tracks.append(conductor)
 meta=[(0,mido.MetaMessage('track_name',name=cfg['title'])),(0,mido.MetaMessage('text',text=f'Title: {cfg["title"]}')),(0,mido.MetaMessage('text',text='Artist: Fly on a Wall')),(0,mido.MetaMessage('text',text='Album: Days')),(0,mido.MetaMessage('set_tempo',tempo=tempo)),(0,mido.MetaMessage('time_signature',numerator=4,denominator=4))]
 for section in sections:meta.append(((section['start_bar']-1)*4*TICKS,mido.MetaMessage('marker',text=section['title'])))
 def emit(track,events):
  last=0
  for tick,msg in sorted(events,key=lambda item:(item[0],0 if item[1].type=='note_off' else 1)):
   assert tick>=last;track.append(msg.copy(time=tick-last));last=tick
  track.append(mido.MetaMessage('end_of_track',time=end-last))
 emit(conductor,meta);total_notes=0
 for i,(name,ch,program,volume,pan) in enumerate(SPECS):
  if i in (0,1) and cfg['scene'] in ('rainy_evening_drizzle','crescent_moon'):program=4;name='Electric '+name.lower()
  if i==4:program=cfg['lead'];name={59:'Muted trumpet',64:'Soprano saxophone',65:'Alto saxophone',66:'Tenor saxophone',11:'Vibraphone',71:'Clarinet'}[program]
  track=mido.MidiTrack();mid.tracks.append(track);track.append(mido.MetaMessage('track_name',name=name));track.append(mido.Message('program_change',channel=ch,program=program));track.append(mido.Message('control_change',channel=ch,control=7,value=volume))
  groups=defaultdict(list)
  for n in notes[i]:groups[n[2]].append(n)
  events=list(controls[i])
  for pitch,group in groups.items():
   group.sort(key=lambda n:n[0])
   for a,b in zip(group,group[1:]):a[1]=min(a[1],b[0])
   for start,stop,pitch,velocity in group:
    if stop<=start:continue
    events.extend([(start,mido.Message('note_on',channel=ch,note=pitch,velocity=velocity)),(stop,mido.Message('note_off',channel=ch,note=pitch,velocity=0))]);total_notes+=1
  emit(track,events)
 path=destination or ALBUM/'midi'/f'{cfg["slug"]}.mid';mid.save(path)
 return dict(title=cfg['title'],midi=str(path.relative_to(ALBUM)) if destination is None else path.name,tempo_bpm=cfg['tempo'],transpose_semitones=cfg['transpose'],bars=cfg['bars'],duration_seconds=mid.length,note_count=total_notes,sections=sections,harmony=[x['chord'] for x in harmony],image_profile_by_bar=[x['profile']+1 for x in harmony],swing=swing,motif_degrees=motif,seed=SEED+index*100)

def validate_midi(path,cfg):
 mid=mido.MidiFile(path,charset='utf-8');count=0;markers=[];pertrack=[]
 for track in mid.tracks:
  active=set();n=0;ticks=0
  for msg in track:
   assert msg.time>=0;ticks+=msg.time
   if msg.type=='marker':markers.append(msg.text)
   if msg.type=='note_on' and msg.velocity>0:
    key=(msg.channel,msg.note);assert key not in active;active.add(key);count+=1;n+=1
   elif msg.type=='note_off' or (msg.type=='note_on' and msg.velocity==0):
    key=(msg.channel,msg.note);assert key in active;active.remove(key)
  assert not active and ticks==cfg['bars']*4*TICKS
  pertrack.append(n)
 assert abs(mid.length-cfg['duration'])<.001 and len(markers)==len(cfg['sections']) and count>300
 return dict(duration_seconds=mid.length,notes=count,notes_per_track=pertrack,markers=markers,balanced_notes=True)
def run(cmd,log=None):
 if log:
  with Path(log).open('w') as f:subprocess.run([str(x) for x in cmd],stdout=f,stderr=subprocess.STDOUT,check=True)
 else:subprocess.run([str(x) for x in cmd],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
def measure(path):
 ff=ROOT/'.venv/native/bin/ffmpeg'
 r=subprocess.run([str(ff),'-hide_banner','-i',str(path),'-af','loudnorm=I=-17:TP=-1:LRA=12:print_format=json','-f','null','-'],capture_output=True,text=True,check=True)
 return json.loads(r.stderr[r.stderr.rfind('{'):r.stderr.rfind('}')+1])
def audio_stats(path,expected):
 peak=0.;sum2=0.;silent=0;frames=0;stereo=0.;window=[]
 with sf.SoundFile(path) as f:
  assert f.samplerate==RATE and f.channels==2 and f.frames==expected*RATE
  for block in f.blocks(blocksize=RATE*10,dtype='float64'):
   assert np.isfinite(block).all();peak=max(peak,float(abs(block).max()));sum2+=float(np.sum(block*block));frames+=len(block);silent+=int((np.max(abs(block),axis=1)<1e-5).sum());stereo+=float(np.sum(abs(block[:,0]-block[:,1])));window.append(float(np.sqrt(np.mean(block*block))))
  f.seek(max(0,f.frames-1));last=f.read(1)
 rms=math.sqrt(sum2/(frames*2))
 assert 1e-5<rms<.5 and 0<peak<1 and stereo/frames>1e-6 and abs(last).max()<1e-5
 return dict(duration_seconds=frames/RATE,sample_rate=RATE,channels=2,peak_dbfs=20*math.log10(peak),rms_dbfs=20*math.log10(rms),silent_frame_percent=100*silent/frames,ten_second_rms=window,true_stereo=True)
def render_track(cfg):
 slug=cfg['slug'];work=ALBUM/'work';raw=work/f'{slug}_raw.wav';midi=ALBUM/'midi'/f'{slug}.mid';wav=ALBUM/'audio'/f'{slug}.wav';mp3=wav.with_suffix('.mp3')
 fluid=ROOT/'.venv/native/bin/fluidsynth';ff=ROOT/'.venv/native/bin/ffmpeg';font=ROOT/'soundfonts/MuseScore_General.sf2'
 print('Rendering',cfg['title'],flush=True)
 run([fluid,'-ni','-g','.38','-r',str(RATE),'-o','synth.reverb.active=1','-o','synth.reverb.room-size=.38','-o','synth.reverb.level=.17','-o','synth.chorus.active=0','-T','wav','-O','float','-F',raw,font,midi],work/f'{slug}_render.log')
 loud=measure(raw);gain_db=min(-17-float(loud['input_i']),-1.2-float(loud['input_tp']));gain=10**(gain_db/20)
 # Stream the master so long tracks do not require gigabytes of RAM.
 target_frames=cfg['duration']*RATE;position=0
 with sf.SoundFile(raw) as source,sf.SoundFile(wav,'w',samplerate=RATE,channels=2,subtype='PCM_24') as out:
  while position<target_frames:
   length=min(RATE*10,target_frames-position);block=source.read(length,dtype='float64',always_2d=True)
   if len(block)<length:block=np.pad(block,((0,length-len(block)),(0,0)))
   ix=np.arange(position,position+length);fade=np.minimum(np.clip(ix/(RATE*.08),0,1),np.clip((target_frames-1-ix)/(RATE*5),0,1))
   out.write(block*gain*fade[:,None]);position+=length
 stats=audio_stats(wav,cfg['duration']);master_loudness=measure(wav)
 run([ff,'-y','-hide_banner','-loglevel','error','-i',wav,'-i',ALBUM/'artwork/cover.png','-map','0:a','-map','1:v','-c:a','libmp3lame','-b:a','256k','-c:v','png','-id3v2_version','3','-metadata','album=Days','-metadata','artist=Fly on a Wall','-metadata',f'title={cfg["title"]}','-metadata',f'track={CONFIGS.index(cfg)+1}/{len(CONFIGS)}','-metadata:s:v','title=Album cover','-metadata:s:v','comment=Cover (front)',mp3])
 return dict(wav=str(wav.relative_to(ALBUM)),mp3=str(mp3.relative_to(ALBUM)),validation=stats,master_loudness=master_loudness,static_gain_db=gain_db,sha256={wav.name:digest(wav),mp3.name:digest(mp3)})
def package_album():
 audio=ALBUM/'audio';combined=audio/'Days_full_album.wav'
 with sf.SoundFile(combined,'w',samplerate=RATE,channels=2,subtype='PCM_24') as out:
  for cfg in CONFIGS:
   with sf.SoundFile(audio/f'{cfg["slug"]}.wav') as source:
    for block in source.blocks(blocksize=RATE*10,dtype='int32'):out.write(block)
 ff=ROOT/'.venv/native/bin/ffmpeg';meta=ALBUM/'work/chapters.ffmeta'
 lines=[';FFMETADATA1','album=Days','title=Days','artist=Fly on a Wall']
 position=0
 for cfg in CONFIGS:
  lines+=['[CHAPTER]','TIMEBASE=1/1000',f'START={position*1000}',f'END={(position+cfg["duration"])*1000}',f'title={cfg["title"]}'];position+=cfg['duration']
 meta.write_text('\n'.join(lines)+'\n')
 run([ff,'-y','-hide_banner','-loglevel','error','-i',combined,'-i',ALBUM/'artwork/cover.png','-i',meta,'-map','0:a','-map','1:v','-map_metadata','2','-map_chapters','2','-c:a','libmp3lame','-b:a','256k','-c:v','png','-id3v2_version','3',audio/'Days_full_album.mp3'])
 (ALBUM/'Days.m3u').write_text('#EXTM3U\n'+''.join(f'#EXTINF:{c["duration"]},{c["title"]}\naudio/{c["slug"]}.mp3\n' for c in CONFIGS))
 return dict(duration_seconds=2040,track_starts_seconds=np.cumsum([0]+[c['duration'] for c in CONFIGS[:-1]]).tolist(),wav='audio/Days_full_album.wav',mp3='audio/Days_full_album.mp3',validation=audio_stats(combined,2040),sha256={'Days_full_album.wav':digest(combined),'Days_full_album.mp3':digest(audio/'Days_full_album.mp3')})
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--compose-only',action='store_true');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
 for name in ['midi','audio','work']:(ALBUM/name).mkdir(parents=True,exist_ok=True)
 neural=json.loads((ALBUM/'neural/provenance.json').read_text());tracks=[]
 for i,cfg in enumerate(CONFIGS):
  print('Composing',cfg['title'],flush=True);entry=compose(i,neural);entry['midi_validation']=validate_midi(ALBUM/entry['midi'],cfg);entry['midi_sha256']=digest(ALBUM/entry['midi']);tracks.append(entry)
 write_json(ALBUM/'composition.json',dict(album='Days',tracks=tracks,composer_sha256=digest(Path(__file__))))
 if args.compose_only:return
 for i,cfg in enumerate(CONFIGS):
  cache=ALBUM/'work'/f'{cfg["slug"]}_audio.json'
  cache_key=dict(midi=tracks[i]['midi_sha256'],composer=digest(Path(__file__)),soundfont=digest(ROOT/'soundfonts/MuseScore_General.sf2'),cover=digest(ALBUM/'artwork/cover.png'))
  if args.resume and cache.exists():
   cached=json.loads(cache.read_text())
   if cached.get('cache_key')==cache_key and all((ALBUM/'audio'/name).exists() and digest(ALBUM/'audio'/name)==sha for name,sha in cached['sha256'].items()):tracks[i]['audio']=cached;continue
  result=render_track(cfg);result['midi_sha256']=tracks[i]['midi_sha256'];result['cache_key']=cache_key;tracks[i]['audio']=result;write_json(cache,result)
 album=package_album()
 provenance=dict(album='Days',track_order=[c['title'] for c in CONFIGS],duration_minutes=34,tracks=tracks,album_audio=album,neural_provenance='neural/provenance.json',neural_provenance_sha256=digest(ALBUM/'neural/provenance.json'),composer_sha256=digest(Path(__file__)),artwork_prompts='artwork/prompts.json',soundfont=dict(file='MuseScore_General.sf2',sha256=digest(ROOT/'soundfonts/MuseScore_General.sf2'),source='https://ftp.osuosl.org/pub/musescore/soundfont/MuseScore_General/MuseScore_General.sf2',license='../../soundfonts/MuseScore_General_License.md'),mapping=dict(total_spikes='Density, bass activity and melodic velocity',active_neurons='Chord extensions',synchrony='Rhythmic accents',burstiness='Swing, fills and phrase displacement',persistence='Note sustain',trial_variance='Bounded timing humanization',population_rank='Shared album motif and solo contours',temporal_population_activity='Harmonic choices across developing sections',left_right_balance='Bounded stereo pan'),form_note='Each track is newly composed across 82-136 bars using all 15 scene image responses with theme, solo, interlude, reprise and coda sections. No audio looping or reuse of the earlier two-minute song.',scientific_limitations=neural['scientific_limitations'])
 write_json(ALBUM/'Days_provenance.json',provenance);print('Days complete: 34:00, seven tracks, stereo 48 kHz / 24-bit masters and MP3 listening copies.',flush=True)
if __name__=='__main__':main()
