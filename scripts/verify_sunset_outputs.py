"""Independent validation of final MIDI, WAV, provenance and actual neural data."""
from pathlib import Path
from collections import Counter
import hashlib,json
import mido,numpy as np,pandas as pd,soundfile as sf
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=json.loads((R/'sunset_fly_jazz_provenance.json').read_text())
 m=mido.MidiFile(R/'sunset_fly_jazz.mid');markers=[];notes=0;tracks=[]
 assert abs(m.length-120)<.001
 for tr in m.tracks:
  active=Counter();tick=0;tracknotes=0
  for ev in tr:
   assert ev.time>=0;tick+=ev.time
   if ev.type=='marker':markers.append((tick,ev.text))
   if ev.type=='set_tempo':assert ev.tempo==750000
   if ev.type=='time_signature':assert (ev.numerator,ev.denominator)==(4,4)
   if ev.type in ('note_on','note_off'):
    key=(ev.channel,ev.note)
    if ev.type=='note_on' and ev.velocity:
     assert not active[key],f'Overlapping same-pitch notes: {key}'
     assert 1<=ev.velocity<=127 and 0<=ev.note<=127
     active[key]+=1;notes+=1;tracknotes+=1
    else:
     assert active[key]==1,f'Unmatched note off: {key}'
     active[key]-=1
  assert not any(active.values()) and tick==160*m.ticks_per_beat
  tracks.append(tracknotes)
 assert len(markers)==4 and [x[0] for x in markers]==[i*40*m.ticks_per_beat for i in range(4)]
 assert sum(v>0 for v in tracks)>=4 and notes==p['midi_note_count']
 a,sr=sf.read(R/'sunset_fly_jazz.wav');assert a.shape==(5760000,2) and sr==48000
 assert np.isfinite(a).all() and 1e-5<np.sqrt(np.mean(a*a))<.5 and abs(a).max()<1
 assert np.max(abs(a[-1]))<1e-6 and np.mean(abs(a[:,0]-a[:,1]))>1e-6
 section_rms=[float(np.sqrt(np.mean(c*c))) for c in np.array_split(a,4)]
 assert min(section_rms)>1e-5
 spikes=[];active=[];recurrent=[]
 network_ids=pd.read_csv(R/'data/2025_Completeness_783.csv',index_col=0).index.to_numpy()
 ids=set(p['visual_input_ids']);assert ids.issubset(set(network_ids.tolist()))
 n=p['total_network_neurons'];trials=p['trials_per_image'];duration=p['simulation_duration_per_image']
 for i,source in enumerate(p['source_images']):
  assert sha(R/source['filename'])==source['sha256']
  df=pd.read_parquet(R/f'data/sunset_jazz/image_{i+1:02d}_spikes.parquet')
  assert len(df)>0 and df.neuron_index.between(0,n-1).all()
  assert set(df.trial)==set(range(trials)) and df.time_ms.between(0,duration*1000,inclusive='left').all()
  assert set(df.image_filename)=={source['filename']} and set(df.image_index)=={i+1}
  assert len(df)==p['neural_features'][i]['total_spikes']
  assert np.array_equal(network_ids[df.neuron_index.to_numpy()],df.flywire_id.to_numpy())
  assert np.allclose(df.time_ms.to_numpy()/.1,np.round(df.time_ms.to_numpy()/.1))
  downstream=int((~df.flywire_id.isin(ids)).sum());assert downstream>0
  spikes.append(len(df));active.append(int(df.neuron_index.nunique()));recurrent.append(downstream)
 f=pd.read_csv(R/'data/sunset_jazz/features.csv');normalized=f.filter(regex='^normalized_')
 assert len(f)==4 and np.isfinite(normalized.to_numpy()).all()
 assert (normalized.to_numpy()>=0).all() and (normalized.to_numpy()<=1).all()
 assert normalized.drop_duplicates().shape[0]==4
 for filename,expected in p['output_sha256'].items():assert sha(R/filename)==expected
 result={'passed':True,'midi_duration':m.length,'wav_duration':len(a)/sr,'sample_rate':sr,'channels':2,
 'notes':notes,'notes_per_track':tracks,'spikes_per_image':spikes,'active_neurons_per_image':active,
 'downstream_spikes_per_image':recurrent,'section_rms':section_rms,
 'peak_dbfs':float(20*np.log10(abs(a).max())),'rms_dbfs':float(20*np.log10(np.sqrt(np.mean(a*a)))),
 'true_stereo':True,'four_distinct_neural_feature_vectors':True,'no_overlapping_or_stuck_notes':True}
 (R/'data/sunset_jazz/final_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
