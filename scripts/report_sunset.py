"""Print the completed experiment and output validation report."""
from pathlib import Path
import json
R=Path(__file__).resolve().parents[1]
p=json.loads((R/'sunset_fly_jazz_provenance.json').read_text())
v=json.loads((R/'data/sunset_jazz/final_validation.json').read_text())
lines=['SOURCE IMAGES']+[f"  image {i+1}: {s['filename']}" for i,s in enumerate(p['source_images'])]
lines+=['','NEURAL SIMULATION',f"  backend: {p['brain_backend']}",f"  device: {p['device']}",f"  trials/image: {p['trials_per_image']}",f"  simulation duration/image: {p['simulation_duration_per_image']} biological second",f"  visual input neurons: {p['number_of_visual_input_neurons']} deterministic proxies",f"  total spikes: {sum(v['spikes_per_image']):,}",f"  active neurons by image: {v['active_neurons_per_image']}",'','COMPOSITION','  style: mellow modern jazz','  tempo: 80 BPM','  meter: 4/4','  bars: 40',f"  MIDI duration: {v['midi_duration']:.6f} seconds",f"  MIDI notes: {v['notes']}",f"  tracks: {', '.join(p['midi_tracks'])}",'','AUDIO',f"  SoundFont: {p['soundfont']}",'  renderer: FluidSynth',f"  sample rate: {p['sample_rate']}",f"  channels: {p['channels']}",f"  WAV duration: {p['wav_duration_seconds']:.6f} seconds",f"  peak: {p['audio_peak_dbfs']:.2f} dBFS",f"  RMS: {p['audio_rms_dbfs']:.2f} dBFS",f"  silent frames: {p['silent_frame_percentage']:.3f}%",'','OUTPUTS','  ./sunset_fly_jazz.mid','  ./sunset_fly_jazz.wav','  ./sunset_fly_jazz_provenance.json']
text='\n'.join(lines)+'\n';(R/'data/sunset_jazz/final_report.txt').write_text(text);print(text)
