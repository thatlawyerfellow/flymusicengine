"""Finalize Days tags without re-encoding audio or changing musical events."""
from pathlib import Path
import json, shutil, subprocess, sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mido
import days_album as music
A=music.ALBUM
ff=music.ROOT/'.venv/native/bin/ffmpeg'
probe=music.ROOT/'.venv/native/bin/ffprobe'
def audio_hash(path):
 return subprocess.check_output([str(ff),'-v','error','-i',str(path),'-map','0:a:0','-c','copy','-f','hash','-hash','sha256','-'],text=True).strip()
def events(path):
 return [[m.dict() for m in t if m.type not in ('text','track_name')] for t in mido.MidiFile(path,charset='utf-8').tracks]
report=[]
for i,(slug,title) in enumerate([(c['slug'],c['title']) for c in music.CONFIGS]+[('Days_full_album','Days')]):
 path=A/'audio'/f'{slug}.mp3';temp=path.with_name(path.stem+'.tagged.mp3');before=audio_hash(path)
 cmd=[str(ff),'-v','error','-y','-i',str(path),'-i',str(A/'artwork/cover.png'),'-map','0:a:0','-map','1:v:0','-map_metadata','0','-map_chapters','0','-c','copy','-id3v2_version','3','-metadata',f'title={title}','-metadata','artist=Fly on a Wall','-metadata','album_artist=Fly on a Wall','-metadata','album=Days','-metadata',f'track={i+1}/7' if i<7 else 'track=','-disposition:v:0','attached_pic','-metadata:s:v','title=Album cover','-metadata:s:v','comment=Cover (front)',str(temp)]
 subprocess.run(cmd,check=True)
 assert audio_hash(temp)==before
 info=json.loads(subprocess.check_output([str(probe),'-v','error','-show_format','-show_streams','-show_chapters','-of','json',str(temp)],text=True))
 tags=info['format']['tags'];assert tags['title']==title and tags['artist']=='Fly on a Wall' and tags['album']=='Days'
 assert any(s.get('disposition',{}).get('attached_pic') for s in info['streams'])
 cover=subprocess.check_output([str(ff),'-v','error','-i',str(temp),'-map','0:v:0','-c','copy','-f','image2pipe','-'])
 assert cover==(A/'artwork/cover.png').read_bytes()
 if i==7:assert len(info['chapters'])==7
 temp.replace(path);report.append(dict(file=str(path.relative_to(A)),title=title,artist='Fly on a Wall',album='Days',audio_unchanged=True,cover_verified=True))
neural=json.loads((A/'neural/provenance.json').read_text());p=json.loads((A/'Days_provenance.json').read_text());tracks=[]
for i,cfg in enumerate(music.CONFIGS):
 path=A/'midi'/f'{cfg["slug"]}.mid';before=events(path);entry=music.compose(i,neural);assert events(path)==before
 entry['midi_validation']=music.validate_midi(path,cfg);entry['midi_sha256']=music.digest(path)
 meta=[m for m in mido.MidiFile(path,charset='utf-8').tracks[0]]
 assert meta[0].name==cfg['title']
 assert {'Title: '+cfg['title'],'Artist: Fly on a Wall','Album: Days'}<={m.text for m in meta if m.type=='text'}
 audio=p['tracks'][i]['audio'];audio['midi_sha256']=entry['midi_sha256']
 audio['cache_key'].update(midi=entry['midi_sha256'],composer=music.digest(music.ROOT/'days_album.py'))
 audio['sha256']={name:music.digest(A/'audio'/name) for name in audio['sha256']}
 entry['audio']=audio;tracks.append(entry);music.write_json(A/'work'/f'{cfg["slug"]}_audio.json',audio)
 report.append(dict(file=str(path.relative_to(A)),title=cfg['title'],artist='Fly on a Wall',album='Days',musical_events_unchanged=True))
shutil.copy2(A/'artwork/cover.png',A/'midi/cover.png')
p.update(artist='Fly on a Wall',tracks=tracks,composer_sha256=music.digest(music.ROOT/'days_album.py'))
p['album_audio']['sha256']={name:music.digest(A/'audio'/name) for name in p['album_audio']['sha256']}
music.write_json(A/'Days_provenance.json',p)
music.write_json(A/'composition.json',dict(album='Days',artist='Fly on a Wall',tracks=[{k:v for k,v in t.items() if k!='audio'} for t in tracks],composer_sha256=p['composer_sha256']))
music.write_json(A/'metadata_validation.json',dict(passed=True,files=report,midi_artwork='midi/cover.png'))
print('Updated and verified 8 MP3s and 7 MIDI files; audio and musical events unchanged.')
