"""Run every Days scene image through the existing whole-brain TorchModel."""
from pathlib import Path
from types import SimpleNamespace
import argparse, json, time
import numpy as np
import pandas as pd
import sunset_jazz as brain
ROOT=Path(__file__).resolve().parent
ALBUM=ROOT/'albums/Days_v2'
SCENES=['sunrise','traffic','high_noon','rainy_evening_drizzle','sunset','crescent_moon','bombolini']
TITLES=['Sunrise','Traffic','High Noon','Rainy Evening Drizzle','Sunset','Crescent Moon','Bombolini']

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--wait-for-images',action='store_true');args=parser.parse_args()
 rows=[];segments=[];sources=[];experiments=[]
 for scene in SCENES:
  images=[ALBUM/'artwork'/scene/f'{i:02d}.png' for i in range(1,16)]
  while not all(p.exists() for p in images):
   if not args.wait_for_images:raise RuntimeError(f'Missing artwork for {scene}')
   print(f'Waiting for {scene} artwork ({sum(p.exists() for p in images)}/15)',flush=True);time.sleep(10)
  brain.DATA=ALBUM/'neural'/scene;brain.DATA.mkdir(parents=True,exist_ok=True)
  for name in ['weight_coo.pkl','weight_csr.pkl']:
   target=brain.DATA/name;source=ROOT/'data'/name
   if not source.exists():source=ROOT/'data/sunset_jazz'/name
   if source.exists() and not target.exists():target.symlink_to(source)
  # Two independent trials, 0.5 biological seconds, unchanged full connectome.
  config=SimpleNamespace(threads=2,inputs=1000,trials=2,duration=.5,resume=True)
  raw,norm,provenance=brain.simulate(images,config)
  profiles=[]
  for i,p in enumerate(images):
   df=pd.read_parquet(brain.DATA/f'image_{i+1:02d}_spikes.parquet')
   recurrent=df[~df.flywire_id.isin(provenance['visual_input_ids'])]
   groups=np.bincount(recurrent.neuron_index.to_numpy()%7,minlength=7)
   hist=np.histogram(df.time_ms,bins=np.linspace(0,500,11))[0]
   profiles.append(dict(image_index=i+1,spikes=len(df),population_activity=groups.tolist(),population_rank=np.argsort(-groups).tolist(),burst_profile=(hist/max(int(hist.max()),1)).tolist()))
   sources.append(dict(scene=scene,path=str(p.relative_to(ALBUM)),sha256=brain.digest(p)))
  rows.append({key:float(np.mean([f[key] for f in raw])) for key in raw[0]})
  segments.append(profiles)
  provenance.update(scene=scene,image_features=raw,source_images=sources[-15:])
  (brain.DATA/'provenance.json').write_text(json.dumps(provenance,indent=2,allow_nan=False));experiments.append(dict(scene=scene,provenance=f'{scene}/provenance.json',sha256=brain.digest(brain.DATA/'provenance.json')))
 frame=pd.DataFrame(rows);normalized=((frame-frame.min())/(frame.max()-frame.min()).replace(0,np.nan)).fillna(.5)
 result=dict(album='Days',tracks=TITLES,source_images=sources,raw_features=rows,normalized_features=normalized.to_dict('records'),temporal_profiles=segments,experiments=experiments,images_per_scene=15,total_images=105,simulation_duration_per_trial_seconds=.5,trials_per_image=2,total_network_neurons=provenance['total_network_neurons'],neural_driver_sha256=brain.digest(Path(__file__)),image_generation='Built-in image_gen; prompts in artwork/prompts.json',scientific_limitations='Generated RGB images drive deterministic proxy sensory channels in the full existing fly-brain network. There is no measured UV or anatomical retina reconstruction. Each image uses two independent 0.5-second trials; responses are interpreted by a human-designed jazz composer, not claimed to be visual understanding or autonomous composition.')
 (ALBUM/'neural/provenance.json').write_text(json.dumps(result,indent=2,allow_nan=False));print('All 105 images simulated and aggregated.',flush=True)
if __name__=='__main__':main()
