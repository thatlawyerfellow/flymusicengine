"""Add RIFF INFO and ID3v2.3 artwork/tags while preserving WAV PCM bytes."""
from pathlib import Path
import hashlib, json, struct, sys, subprocess
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import days_album as music
A = music.ALBUM

def chunks(path):
    with path.open('rb') as f:
        header = f.read(12)
        assert header[:4] == b'RIFF' and header[8:] == b'WAVE'
        assert struct.unpack('<I', header[4:8])[0] + 8 == path.stat().st_size
        while raw := f.read(8):
            assert len(raw) == 8
            kind, size = struct.unpack('<4sI', raw)
            payload = f.read(size)
            assert len(payload) == size
            if size % 2: f.read(1)
            yield kind, payload

def chunk(kind, data):
    return kind + struct.pack('<I', len(data)) + data + (b'\0' if len(data) % 2 else b'')

def frame(kind, data):
    return kind + struct.pack('>I', len(data)) + b'\0\0' + data

def text_frame(kind, text):
    return frame(kind, b'\x01' + text.encode('utf-16'))

def finalize(path, title, number):
    original = list(chunks(path))
    before = hashlib.sha256(b''.join(data for kind, data in original if kind == b'data')).hexdigest()
    rights = 'Fly on a Wall - Days. CC BY-NC 4.0. https://creativecommons.org/licenses/by-nc/4.0/'
    credits = rights + '\nAlbum: https://soundcloud.com/ajay-kumar-819538867/sets/days\nThird-party SoundFont notice (original terms retained):\n' + (music.ROOT / 'soundfonts/MuseScore_General_License.md').read_text()
    info = {b'INAM': title, b'IART': 'Fly on a Wall', b'IPRD': 'Days', b'ICOP': rights, b'ICMT': credits}
    tags = {b'TIT2': title, b'TPE1': 'Fly on a Wall', b'TPE2': 'Fly on a Wall', b'TALB': 'Days', b'TCOP': rights}
    if number:
        info[b'ITRK'] = str(number)
        tags[b'TRCK'] = f'{number}/7'
    cover = (A / 'artwork/cover.png').read_bytes()
    frames = b''.join(text_frame(k, v) for k, v in tags.items())
    frames += frame(b'WCOP', b'https://creativecommons.org/licenses/by-nc/4.0/')
    frames += frame(b'APIC', b'\0image/png\0\x03\0' + cover)
    n = len(frames)
    id3 = b'ID3\x03\0\0' + bytes((n >> shift) & 127 for shift in (21, 14, 7, 0)) + frames
    extras = [(b'LIST', b'INFO' + b''.join(chunk(k, v.encode('utf-8') + b'\0') for k, v in info.items())), (b'id3 ', id3)]
    kept = [(k, v) for k, v in original if k.lower() != b'id3 ' and not (k == b'LIST' and v.startswith(b'INFO'))]
    del original
    temp = path.with_suffix('.tagged.wav')
    with temp.open('wb') as f:
        f.write(b'RIFF\0\0\0\0WAVE')
        for k, v in kept + extras: f.write(chunk(k, v))
        size = f.tell(); f.seek(4); f.write(struct.pack('<I', size - 8))
    del kept
    result = list(chunks(temp))
    after = hashlib.sha256(b''.join(v for k, v in result if k == b'data')).hexdigest()
    assert after == before
    assert next(v for k, v in result if k == b'id3 ') == id3
    # Read every ID3 frame back and validate the decoded text and picture.
    body = next(v for k, v in result if k == b'id3 ')[10:]
    parsed = {}
    while body:
        key = body[:4]; length = struct.unpack('>I', body[4:8])[0]
        parsed[key] = body[10:10+length]; body = body[10+length:]
    assert all(parsed[k][1:].decode('utf-16') == v for k, v in tags.items())
    assert parsed[b'APIC'] == b'\0image/png\0\x03\0' + cover
    probe = music.ROOT / '.venv/native/bin/ffprobe'
    j = json.loads(subprocess.check_output([str(probe), '-v', 'error', '-show_format', '-of', 'json', str(temp)], text=True))
    assert j['format']['tags']['title'] == title
    assert j['format']['tags']['artist'] == 'Fly on a Wall'
    assert j['format']['tags']['album'] == 'Days'
    temp.replace(path)
    print(f'Verified {path.name}: tags, embedded cover, unchanged PCM', flush=True)
    return dict(file=str(path.relative_to(A)), title=title, artist='Fly on a Wall', album='Days', pcm_sha256=before, audio_unchanged=True, embedded_cover_verified=True)

def main():
    results = []
    for i, cfg in enumerate(music.CONFIGS):
        results.append(finalize(A / 'audio' / f'{cfg["slug"]}.wav', cfg['title'], i+1))
    results.append(finalize(A / 'audio/Days_full_album.wav', 'Days', None))
    p = json.loads((A / 'Days_provenance.json').read_text())
    for cfg, track in zip(music.CONFIGS, p['tracks']):
        name = f'{cfg["slug"]}.wav'
        track['audio']['sha256'][name] = music.digest(A / 'audio' / name)
        music.write_json(A / 'work' / f'{cfg["slug"]}_audio.json', track['audio'])
    p['album_audio']['sha256']['Days_full_album.wav'] = music.digest(A / 'audio/Days_full_album.wav')
    music.write_json(A / 'Days_provenance.json', p)
    music.write_json(A / 'wav_metadata_validation.json', dict(passed=True, files=results))

if __name__ == '__main__': main()
