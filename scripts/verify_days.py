"""Verify all 105 inputs, seven scores, masters, MP3s and neural causality."""
from pathlib import Path
import json,sys,tempfile,subprocess,math
import numpy as np
import pandas as pd
import soundfile as sf
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import days_album as music
R=music.ROOT;A=music.ALBUM
p=json.loads((A/'Days_provenance.json').read_text());neural=json.loads((A/'neural/provenance.json').read_text())
assert p['track_order']==[c['title'] for c in music.CONFIGS] and p['duration_minutes']==34
assert len(neural['source_images'])==105 and neural['images_per_scene']==15
assert len({x['sha256'] for x in neural['source_images']})==105
assert len({c['duration'] for c in music.CONFIGS})==7
assert all(150<=c['duration']<=330 for c in music.CONFIGS)
results=[]
for i,cfg in enumerate(music.CONFIGS):
 track=p['tracks'][i];midi=A/track['midi'];sources=[x for x in neural['source_images'] if x['scene']==cfg['scene']]
 assert len(sources)==15 and set(track['image_profile_by_bar'])==set(range(1,16))
 for source in sources:
  path=A/source['path'];assert music.digest(path)==source['sha256']
  with Image.open(path) as im:assert min(im.size)>=1024;im.verify()
 exp_path=A/'neural'/cfg['scene']/'provenance.json';exp=json.loads(exp_path.read_text())
 assert music.digest(exp_path)==neural['experiments'][i]['sha256']
 assert exp['total_network_neurons']==138639 and exp['trials']==2 and exp['duration']==.5
 image_spikes=[]
 for j in range(15):
  spikes=pd.read_parquet(A/'neural'/cfg['scene']/f'image_{j+1:02d}_spikes.parquet',columns=['trial','time_ms','neuron_index','flywire_id'])
  assert set(spikes.trial)=={0,1} and len(spikes)==exp['image_features'][j]['total_spikes']
  downstream=int((~spikes.flywire_id.isin(exp['visual_input_ids'])).sum());assert downstream>0
  assert spikes.time_ms.between(0,500,inclusive='left').all()
  image_spikes.append(dict(image=j+1,spikes=len(spikes),downstream_spikes=downstream))
 assert music.digest(midi)==track['midi_sha256']
 mid=music.validate_midi(midi,cfg);stats=music.audio_stats(A/track['audio']['wav'],cfg['duration'])
 for name,expected in track['audio']['sha256'].items():assert music.digest(A/'audio'/name)==expected
 minute_rms=[]
 with sf.SoundFile(A/track['audio']['wav']) as f:
  for block in f.blocks(blocksize=60*music.RATE):minute_rms.append(float(np.sqrt(np.mean(block*block))))
 assert len(minute_rms)==math.ceil(cfg['duration']/60) and min(minute_rms)>1e-5
 results.append(dict(title=cfg['title'],midi=mid,audio=stats,images=image_spikes,minute_rms=minute_rms))
with sf.SoundFile(A/'audio/Days_full_album.wav') as full:
 assert full.frames==2040*music.RATE and full.channels==2
 for cfg in music.CONFIGS:
  with sf.SoundFile(A/'audio'/f'{cfg["slug"]}.wav') as part:
   for block in part.blocks(blocksize=music.RATE*10,dtype='int32'):assert np.array_equal(block,full.read(len(block),dtype='int32'))
 assert len(full.read(1))==0
with tempfile.TemporaryDirectory(prefix='validation-',dir=A/'work') as temp:
 for i,cfg in enumerate(music.CONFIGS):
  path=Path(temp)/f'{i}.mid';music.compose(i,neural,path);assert music.digest(path)==p['tracks'][i]['midi_sha256']
  altered=json.loads(json.dumps(neural))
  for field in ['raw_features','normalized_features','temporal_profiles']:altered[field][i]=neural[field][(i+1)%7]
  music.compose(i,altered,path);assert music.digest(path)!=p['tracks'][i]['midi_sha256']
  # Change one image's measured spike count at a time; every image must matter.
  for image_index in range(15):
   altered=json.loads(json.dumps(neural))
   altered['temporal_profiles'][i][image_index]['spikes']=10*max(x['spikes'] for x in neural['temporal_profiles'][i])+1
   music.compose(i,altered,path);assert music.digest(path)!=p['tracks'][i]['midi_sha256']
ffprobe=R/'.venv/native/bin/ffprobe'
for name,seconds in [(c['slug'],c['duration']) for c in music.CONFIGS]+[('Days_full_album',2040)]:
 j=json.loads(subprocess.check_output([str(ffprobe),'-v','error','-show_format','-show_streams','-of','json',str(A/'audio'/f'{name}.mp3')],text=True))
 assert abs(float(j['format']['duration'])-seconds)<.15 and j['format']['tags'].get('album')=='Days'
 assert j['format']['tags'].get('artist')=='Fly on a Wall'
 assert j['format']['tags'].get('title')==next((c['title'] for c in music.CONFIGS if c['slug']==name),'Days')
 assert any(s.get('codec_type')=='audio' and s.get('channels')==2 for s in j['streams'])
 assert any(s.get('codec_type')=='video' for s in j['streams'])
report=dict(passed=True,duration_seconds=2040,duration_minutes=34,total_images=105,tracks=results,continuous_master_bit_exact_concatenation=True,deterministic_recomposition=True,neural_feature_sensitivity=True,individual_image_sensitivity_checks=105,mp3s_valid_with_cover_and_album_tags=True)
music.write_json(A/'validation.json',report)
print(json.dumps({k:v for k,v in report.items() if k!='tracks'},indent=2))
for r in results:print(r['title'],r['midi']['notes'],'notes;',sum(x['spikes'] for x in r['images']),'spikes;',r['audio']['duration_seconds'],'seconds')
