"""Check deterministic composition and sensitivity to recorded neural responses."""
import sys,json,tempfile,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import sunset_jazz as pipeline

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 data=ROOT/'data/sunset_jazz';p=json.loads((data/'composition_provenance.json').read_text())
 images=[ROOT/x['filename'] for x in p['source_images']]
 original=sha(ROOT/'sunset_fly_jazz.mid')
 with tempfile.TemporaryDirectory(prefix='causal-check-',dir=data) as temp:
  pipeline.ROOT=Path(temp)
  pipeline.compose(images,p['neural_features'],p['normalized_neural_features'])
  repeated=sha(Path(temp)/'sunset_fly_jazz.mid');assert repeated==original,'Composition not deterministic'
  pipeline.compose(images,list(reversed(p['neural_features'])),list(reversed(p['normalized_neural_features'])))
  changed=sha(Path(temp)/'sunset_fly_jazz.mid');assert changed!=original,'Composition insensitive to neural responses'
 result={'deterministic_recomposition':True,'neural_response_permutation_changes_midi':True,'original_midi_sha256':original,'permuted_neural_response_midi_sha256':changed,
 'method':'Same source filenames, seed, tempo and form; only recorded neural feature vectors were permuted. Temporary comparison MIDIs removed.'}
 (data/'causality_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
