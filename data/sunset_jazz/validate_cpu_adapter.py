import sys,time,pickle,json
from pathlib import Path
sys.path[:0]=[str(Path.cwd()),str(Path.cwd()/'code')]
from run_pytorch import TorchModel,MODEL_PARAMS,DT
from sunset_cpu_weights import CPUEventWeights
import torch
w=pickle.load(open('data/sunset_jazz/weight_csr.pkl','rb'));torch.set_num_threads(4);n=w.shape[0]
adapted=CPUEventWeights(w);selected=torch.randperm(n,generator=torch.Generator().manual_seed(20260917))[:1000].tolist()
models=[TorchModel(4,n,DT,MODEL_PARAMS,ww,exc_indices=selected) for ww in [w,adapted]]
states=[m.state_init() for m in models];gens=[torch.Generator().manual_seed(20260917) for _ in models];rates=torch.zeros(4,n);rates[:,selected]=120
elapsed=[0.,0.];spikes=0
with torch.no_grad():
 for step in range(160):
  for i in range(2):
   t=time.monotonic();states[i]=models[i](rates,*states[i],generator=gens[i]);elapsed[i]+=time.monotonic()-t
  for a,b in zip(*states):
   if not torch.equal(a,b):raise RuntimeError(f'State mismatch at step {step}: {(a-b).abs().max()}')
  spikes+=int(states[0][2].sum())
report=dict(steps=160,biological_ms=16,trials=4,all_six_state_tensors_bit_identical=True,spikes=spikes,elapsed_seconds=elapsed,speedup=elapsed[0]/elapsed[1])
Path('data/sunset_jazz/cpu_adapter_validation.json').write_text(json.dumps(report,indent=2));print(report)
