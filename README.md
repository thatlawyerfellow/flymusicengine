# Days — Fly on a Wall

**Seven scenes. 105 generated images. 34 minutes of instrumental jazz.**

[Listen to the complete album on SoundCloud](https://soundcloud.com/ajay-kumar-819538867/sets/days) · [Download the WAV masters](https://github.com/thatlawyerfellow/flymusicengine/releases/tag/days-v1.0)

*Days* follows a day from first light through traffic, high noon, drizzle, sunset,
and a crescent moon, ending with Bombolini: the filled Italian doughnut. We built
an image-to-neural-activity-to-music pipeline, then gave each scene its own tempo,
instrumental lead, harmonic palette, and developing arrangement.

The neural simulation comes from [eonsystemspbc/fly-brain](https://github.com/eonsystemspbc/fly-brain).
This repository focuses on the music experiment; see that project for the model
and its scientific background.

| Track | Length | Musical character |
|---|---:|---|
| Sunrise | 4:40 | Piano and muted trumpet; a gentle opening |
| Traffic | 4:00 | Alto saxophone, walking bass, busy swing |
| High Noon | 4:48 | Tenor saxophone and bright piano |
| Rainy Evening Drizzle | 5:20 | Electric piano, soprano saxophone, restrained drums |
| Sunset | 5:12 | Muted trumpet, acoustic piano, lingering minor harmony |
| Crescent Moon | 5:28 | Electric piano and vibraphone; the slowest, most spacious track |
| Bombolini | 4:32 | Clarinet, piano, vibraphone, and a lively bass line |
| **Complete album** | **34:00** | **48 kHz / 24-bit stereo WAV** |

## Who composed it?

The AI assistant wrote the musical rules and implementation from the user’s scene
and album brief. The simulated fly-brain activity supplied measurements that
influenced choices within those rules; it did not independently compose the album.

[Read the track-by-track account](albums/Days_v2/TRACK_NOTES.md) for each song’s
jazz vocabulary, chord options, instruments, tempo, duration, structure, and the
specific neural controls applied to melody, harmony, rhythm, dynamics, and phrasing.

## How we made the music

### 1. Build a visual sequence for each scene

We generated 15 separate images for each of the seven scenes using built-in AI
image generation: 105 images in total. The sets vary viewpoint, composition,
light, and visual detail while keeping a shared scene. The complete
[prompt set](albums/Days_v2/artwork/prompts.json) records the inputs.

### 2. Turn pixels into stimulation

Each image is resized to a 16 × 12 RGB grid. The encoder calculates nine channels:
luminance, warm colour contrast, blue, green, edge/local contrast, and weighted
left, right, upper, and lower fields. A fixed seed assigns these signals to 1,000
proxy input neurons. Channel values determine bounded Poisson stimulation rates.
The assignment is reproducible, but it is not a reconstruction of a fly retina.

### 3. Record the network response

Each image drives two independent 0.5-second trials of the existing
138,639-neuron leaky integrate-and-fire network, using its full connectivity and
a 0.1 ms timestep. External stimulation stops during the final 75 ms so we can
measure activity that persists through the network. We record spike times and
neuron identities instead of inventing musical controls directly from the pixels.

### 4. Translate measurements into musical decisions

All 15 image responses contribute to a scene's average feature values, normalized
across the seven scenes. Individual responses also control successive passages:
each track progresses through all 15 image profiles in order.

| Measured response | Role in the composition |
|---|---|
| Total spikes | Note/bass activity and melodic dynamics; local image counts also shape section energy |
| Number of active neurons | Chord richness and extensions |
| Population-bin variability (a synchrony proxy) | Rhythmic accents |
| Temporal burstiness and burst profile | Swing, phrase placement, and fills |
| Activity after input stops | Note sustain |
| Variation between trials | Bounded timing variation |
| Relative activity in seven index-based neuron groups | Motif scale degrees, melodic contours, and harmonic choices |
| Assigned left/right channel balance | Small stereo-position changes |

The seven population groups are formed by neuron index modulo seven; they are
musical analysis groups, not anatomical brain regions. The shared album motif
comes from Sunrise's group activity ranking, adapted to each track's mode.

### 5. Give the measurements a musical form

The composer supplies the jazz vocabulary: chord progressions, scales,
voice-leading rules, instrumental ranges, and a sequence of introduction, theme,
solo passages, interlude, reprise, and coda. Neural measurements influence choices
inside that structure. The scene brief determines the tempo, lead instrument,
transposition, and section lengths.

Track lengths come from bar counts and tempos, spanning 82–136 bars at 60–120 BPM.
We synthesize complete MIDI scores; we do not loop an earlier recording or stretch
audio to fill the running time. Repeated themes give the album continuity while
changes in harmony, orchestration, density, and phrasing give each scene its shape.

### 6. Render, master, and verify

FluidSynth renders the scores with MuseScore General. Masters use a static gain
toward −17 LUFS, constrained by a −1.2 dB true-peak ceiling, with short opening and
closing fades. The release contains seven individual WAVs and one continuous
34-minute WAV, with song title, **Fly on a Wall**, **Days**, and embedded cover art.

Checks cover durations, balanced MIDI notes, stereo audio, clipping, file hashes,
deterministic recomposition, and exact concatenation of the seven masters.
Changing each of the 105 individual image-response profiles changes its score.
That demonstrates the inputs affect the music; it does not measure musical quality
or prove visual understanding. See the [validation report](albums/Days_v2/validation.json)
and [WAV metadata checks](albums/Days_v2/wav_metadata_validation.json).

This is a designed sonification of simulated neural activity. The model does not
recognize the scenes or independently compose jazz. The RGB inputs contain no
measured ultraviolet information, and the sensory mapping is an explicit proxy.

## Files and reproduction

- `days_neural.py`: run the images through the existing model and aggregate responses.
- `days_album.py`: translate recorded features into scores and render the album.
- `sunset_jazz.py` and `sunset_cpu_weights.py`: shared visual encoding, simulation, and CPU support.
- `scripts/finalize_days_wav_metadata.py`: add WAV tags and artwork without changing PCM samples.
- `albums/Days_v2/`: prompts, composition records, measured features, validation, and release checksums.

WAV audio is distributed through the GitHub release. MP3s, MIDI files, standalone
image binaries, SoundFont binaries, simulation caches, and environments are not
part of this publication. The 105 original image files remain local; their hashes
and prompts are recorded. Exact reruns require those original images, the model
data from the linked fly repository, and the licensed SoundFont. Generating new
images from the same prompts will produce a different experiment.

With those inputs restored to the paths recorded by the scripts:

```bash
sh scripts/setup_sunset_audio.sh
.venv/bin/python days_neural.py
.venv/bin/python days_album.py --resume
.venv/bin/python scripts/finalize_days_metadata.py
.venv/bin/python scripts/finalize_days_wav_metadata.py
.venv/bin/python scripts/verify_days.py
```

## License and credits

**The Days album is licensed under [Creative Commons Attribution–NonCommercial
4.0 International](https://creativecommons.org/licenses/by-nc/4.0/).** Credit
**Fly on a Wall — Days**, link to the album and license, and indicate modifications.
See [LICENSE-ALBUM.md](LICENSE-ALBUM.md) for scope and attribution.

The software retains the existing [GNU GPL v2 license](LICENSE); the album license
does not replace the upstream software license or third-party rights. The
[MuseScore General notice](soundfonts/MuseScore_General_License.md) retains its
contributors' acknowledgements and MIT terms. The album uses MuseScore General;
GeneralUser GS is not used in these recordings. Neither SoundFont binary is uploaded.
