# Who composed Days? A track-by-track account

**The AI assistant wrote the composition rules in response to the user’s scene and album brief. The simulated fly-brain network supplied numerical responses that influenced the scores.** It did not choose a jazz style, recognize the pictures, select instruments, or independently decide to compose these songs. No live fly was involved.

The accounts below distinguish preset musical choices from the measured inputs used by the program. They are derived from [`days_album.py`](../../days_album.py), the saved [composition records](composition.json), and the [neural measurements](neural/provenance.json). They describe the released album, not proposed changes to it.

## How to read the musical details

- All tracks use 4/4. Track durations were fixed through bar count and tempo: seconds = bars × 240 / BPM.
- Chord names below show sounding roots after transposition. The saved JSON stores the composer’s pre-transposition labels. These labels describe harmonic options; the actual rootless piano voicings select three or four tones, while bass supplies roots. They are not promises that every labelled extension sounds in every voicing.
- Each song traverses all 15 image profiles. For zero-based bar `b`, the selected profile is `min(14, floor(b × 15 / total_bars))`. Neural responses are interpreted through explicit rules, not played back as raw spikes.
- Feature controls are normalized between the seven scene averages: 0 means the lowest scene average here, not no neural activity. The shared motif uses Sunrise’s neural group ranking; local group rankings guide solo contours and some reprise variations.
- Instrumentation, scales, section roles, base energy, base swing, and allowed chord progressions were preset. Neural group counts and entropy select among progression options outside the fixed theme/reprise pattern and final cadence. Voice leading and pitch-range rules constrain the result.
- Swing percentages below are the realized long-eighth fraction of a beat. The neural adjustment is bounded around a preset base swing. Timing bounds refer to the randomized melody/comping onset adjustment in beats; bass and drum timing have separate bounds.

## Track index

- [Sunrise](#1-sunrise)
- [Traffic](#2-traffic)
- [High Noon](#3-high-noon)
- [Rainy Evening Drizzle](#4-rainy-evening-drizzle)
- [Sunset](#5-sunset)
- [Crescent Moon](#6-crescent-moon)
- [Bombolini](#7-bombolini)

## 1. Sunrise

**4:40 · 84 BPM · 98 bars · 4/4**

### Choices set in the composition code

**Jazz vocabulary.** Major-colour jazz with an F Lydian melodic palette: F, G, A, B, C, D, E. Major ninths, ii–V movement, borrowed minor colour, and altered dominants provide contrast around the F-major home sonority.

**Chord options.** Fmaj9, Em7, A7alt, Dm9, G13, Gm9, C13, Am9, Bbmaj9, Bbm6, Am7, D7alt, Cmaj9, Eb13.

**Opening eight theme bars.** Fmaj9 → Fmaj9 → Em7 → A7alt → Dm9 → G13 → Gm9 → C13. This theme progression is prescribed by the form; neural choices affect other passages and its realization.

**Instruments.** Acoustic piano carries the theme and piano solo; muted trumpet takes the lead section and selected theme/reprise phrases. Upright bass, vibraphone responses, quiet electric-piano washes, and jazz drums complete the ensemble.

**Song structure.** Section lengths and order were fixed before applying the neural controls. “Piano solo” uses electric piano where specified above.

| Section | Role | Bars | Length |
|---|---|---:|---:|
| First light | Introduction | 1–8 | 8 bars |
| Open windows | Theme | 9–24 | 16 bars |
| Morning conversation | Piano solo | 25–48 | 24 bars |
| Reflections | Lead solo | 49–64 | 16 bars |
| A breath of air | Interlude | 65–74 | 10 bars |
| Full daylight | Theme return | 75–90 | 16 bars |
| The road ahead | Coda | 91–98 | 8 bars |

### What the neural responses changed

Sunrise has the highest mean spike count in this album, so its density/velocity control is maximal. It nevertheless stays at the deliberately chosen 84 BPM. Its low persistence gives the shortest sustain setting, and its group ranking supplies the shared motif used across all seven songs.

- **Melody:** the shared Sunrise-derived motif is fitted to this track’s scale and chords. Sunrise’s local population rankings guide solo contour and selected reprise notes; pitch range, chord-tone landing, and step-size limits remain programmed constraints.
- **Harmony:** local group activity and scene entropy (0.529 on the album’s 0–1 scale) select among the allowed progressions outside fixed passages. Active-neuron control 0.571 selects **4-note** rootless comping voicings.
- **Rhythm:** burstiness 0.000 adjusts the preset 62.0% swing to **61.40%**. Local group counts choose accompaniment patterns; burst-profile peaks shift eligible phrases by 0, ¼, or ½ beat. Drum fills also require an eligible section and position.
- **Dynamics:** overall activity control 1.000 contributes to melody/accompaniment velocity and note activity. Each image’s spike count multiplies the preset section energy by **0.85–1.15**; a programmed phrase arc and coda taper are then applied. Synchrony control 0.000 influences drum accents.
- **Phrasing:** persistence 0.000 changes note and chord sustain. Trial variance 0.062 sets melody/comping timing variation to **±0.0081 beats**. Rests, four-bar phrase starts, melodic landing rules, and the broad form remain programmed.

The resulting score contains **1,562 note-on events** across its instruments. This is an output of both the fixed arrangement and the neural controls, not a direct translation of one spike into one note.

## 2. Traffic

**4:00 · 120 BPM · 120 bars · 4/4**

### Choices set in the composition code

**Jazz vocabulary.** Minor/modal jazz with a C Dorian melodic palette: C, D, Eb, F, G, A, Bb. Minor ninths, dominant thirteenths, suspended and altered dominants, and ii–V-style movement support the busy arrangement.

**Chord options.** Cm9, F13, Dm7b5, G7alt, G7sus, Fm9, Bb13, Ebmaj9, Abmaj9, Bbmaj9, Dm7, Gm9, C13, Fmaj9, Cm6/9.

**Opening eight theme bars.** Cm9 → F13 → Cm9 → F13 → Dm7b5 → G7alt → Cm9 → G7sus. This theme progression is prescribed by the form; neural choices affect other passages and its realization.

**Instruments.** Alto saxophone leads the theme, lead section, and reprise, with acoustic-piano solo passages. Upright bass switches to walking patterns in designated active passages; vibraphone, occasional electric-piano washes, and jazz drums add support.

**Song structure.** Section lengths and order were fixed before applying the neural controls. “Piano solo” uses electric piano where specified above.

| Section | Role | Bars | Length |
|---|---|---:|---:|
| Ignition | Introduction | 1–8 | 8 bars |
| Crosswalk theme | Theme | 9–32 | 24 bars |
| Lane changes | Piano solo | 33–56 | 24 bars |
| Signals | Lead solo | 57–72 | 16 bars |
| Under the bridge | Interlude | 73–88 | 16 bars |
| Green light | Theme return | 89–112 | 24 bars |
| Turn the corner | Coda | 113–120 | 8 bars |

### What the neural responses changed

Traffic’s busy feel starts with the chosen 120 BPM, saxophone role, walking-bass rules, and active drum pattern. Its neural activity is mid-range rather than the album maximum; the response refines that preset character instead of creating the traffic metaphor.

- **Melody:** the shared Sunrise-derived motif is fitted to this track’s scale and chords. Traffic’s local population rankings guide solo contour and selected reprise notes; pitch range, chord-tone landing, and step-size limits remain programmed constraints.
- **Harmony:** local group activity and scene entropy (0.000 on the album’s 0–1 scale) select among the allowed progressions outside fixed passages. Active-neuron control 0.147 selects **3-note** rootless comping voicings.
- **Rhythm:** burstiness 0.407 adjusts the preset 66.0% swing to **65.89%**. Local group counts choose accompaniment patterns; burst-profile peaks shift eligible phrases by 0, ¼, or ½ beat. Drum fills also require an eligible section and position.
- **Dynamics:** overall activity control 0.456 contributes to melody/accompaniment velocity and note activity. Each image’s spike count multiplies the preset section energy by **0.85–1.15**; a programmed phrase arc and coda taper are then applied. Synchrony control 0.347 influences drum accents.
- **Phrasing:** persistence 0.479 changes note and chord sustain. Trial variance 0.317 sets melody/comping timing variation to **±0.0127 beats**. Rests, four-bar phrase starts, melodic landing rules, and the broad form remain programmed.

The resulting score contains **1,811 note-on events** across its instruments. This is an output of both the fixed arrangement and the neural controls, not a direct translation of one spike into one note.

## 3. High Noon

**4:48 · 100 BPM · 120 bars · 4/4**

### Choices set in the composition code

**Jazz vocabulary.** Minor/modal jazz with a D Dorian melodic palette: D, E, F, G, A, B, C. Minor ninths alternate with major-ninth colours, dominant thirteenths, and altered turnarounds. The scene name does not imply an exclusively major key.

**Chord options.** Dm9, G13, Em7b5, A7alt, A7sus, Gm9, C13, Fmaj9, Bbmaj9, Cmaj9, Em7, Am9, D13, Gmaj9, Dm6/9.

**Opening eight theme bars.** Dm9 → G13 → Dm9 → G13 → Em7b5 → A7alt → Dm9 → A7sus. This theme progression is prescribed by the form; neural choices affect other passages and its realization.

**Instruments.** Tenor saxophone and acoustic piano alternate foreground roles. Upright bass uses walking patterns in designated active passages, with vibraphone responses, occasional electric-piano washes, and jazz drums.

**Song structure.** Section lengths and order were fixed before applying the neural controls. “Piano solo” uses electric piano where specified above.

| Section | Role | Bars | Length |
|---|---|---:|---:|
| Overhead sun | Introduction | 1–8 | 8 bars |
| Short shadows | Theme | 9–32 | 24 bars |
| Hot pavement | Piano solo | 33–56 | 24 bars |
| Bright glass | Lead solo | 57–72 | 16 bars |
| Shade | Interlude | 73–88 | 16 bars |
| The square awakens | Theme return | 89–112 | 24 bars |
| After noon | Coda | 113–120 | 8 bars |

### What the neural responses changed

High Noon has the highest active-neuron control, producing four-note comping voicings. Its relatively high synchrony and burstiness strengthen drum accents and permit fills in eligible passages. The 100 BPM pace and tenor-saxophone identity are preset choices.

- **Melody:** the shared Sunrise-derived motif is fitted to this track’s scale and chords. High Noon’s local population rankings guide solo contour and selected reprise notes; pitch range, chord-tone landing, and step-size limits remain programmed constraints.
- **Harmony:** local group activity and scene entropy (1.000 on the album’s 0–1 scale) select among the allowed progressions outside fixed passages. Active-neuron control 1.000 selects **4-note** rootless comping voicings.
- **Rhythm:** burstiness 0.729 adjusts the preset 64.0% swing to **64.27%**. Local group counts choose accompaniment patterns; burst-profile peaks shift eligible phrases by 0, ¼, or ½ beat. Drum fills also require an eligible section and position.
- **Dynamics:** overall activity control 0.550 contributes to melody/accompaniment velocity and note activity. Each image’s spike count multiplies the preset section energy by **0.85–1.15**; a programmed phrase arc and coda taper are then applied. Synchrony control 0.768 influences drum accents.
- **Phrasing:** persistence 0.477 changes note and chord sustain. Trial variance 0.376 sets melody/comping timing variation to **±0.0138 beats**. Rests, four-bar phrase starts, melodic landing rules, and the broad form remain programmed.

The resulting score contains **1,968 note-on events** across its instruments. This is an output of both the fixed arrangement and the neural controls, not a direct translation of one spike into one note.

## 4. Rainy Evening Drizzle

**5:20 · 72 BPM · 96 bars · 4/4**

### Choices set in the composition code

**Jazz vocabulary.** Restrained minor jazz with an A Dorian melodic palette: A, B, C, D, E, F#, G. Minor sixth/ninth sonorities, half-diminished chords, borrowed colours, and altered dominants supply the harmonic vocabulary.

**Chord options.** Am9, Fmaj9, Dm9, Bm7b5, E7alt, Am6/9, Cmaj9, Em7, Bb13, G13, Fm6, D13.

**Opening eight theme bars.** Am9 → Am9 → Fmaj9 → Dm9 → Bm7b5 → E7alt → Am6/9 → Am6/9. This theme progression is prescribed by the form; neural choices affect other passages and its realization.

**Instruments.** Electric piano replaces the main acoustic-piano parts; soprano saxophone supplies the lead section and selected theme/reprise phrases. Upright bass, vibraphone, additional electric-piano washes, and a deliberately sparse drum pattern support them.

**Song structure.** Section lengths and order were fixed before applying the neural controls. “Piano solo” uses electric piano where specified above.

| Section | Role | Bars | Length |
|---|---|---:|---:|
| First drops | Introduction | 1–8 | 8 bars |
| Window theme | Theme | 9–24 | 16 bars |
| Pavement reflections | Piano solo | 25–48 | 24 bars |
| Drifting umbrellas | Lead solo | 49–64 | 16 bars |
| Shelter | Interlude | 65–72 | 8 bars |
| The rain returns | Theme return | 73–88 | 16 bars |
| Last drops | Coda | 89–96 | 8 bars |

### What the neural responses changed

Rainy Evening Drizzle has the lowest active-neuron control, producing three-note comping voicings. Moderate persistence extends notes within the restrained arrangement. Electric piano, sparse drums, the 72 BPM pace, and the rain association were chosen in the rules.

- **Melody:** the shared Sunrise-derived motif is fitted to this track’s scale and chords. Rainy Evening Drizzle’s local population rankings guide solo contour and selected reprise notes; pitch range, chord-tone landing, and step-size limits remain programmed constraints.
- **Harmony:** local group activity and scene entropy (0.040 on the album’s 0–1 scale) select among the allowed progressions outside fixed passages. Active-neuron control 0.000 selects **3-note** rootless comping voicings.
- **Rhythm:** burstiness 0.424 adjusts the preset 61.0% swing to **61.00%**. Local group counts choose accompaniment patterns; burst-profile peaks shift eligible phrases by 0, ¼, or ½ beat. Drum fills also require an eligible section and position.
- **Dynamics:** overall activity control 0.461 contributes to melody/accompaniment velocity and note activity. Each image’s spike count multiplies the preset section energy by **0.85–1.15**; a programmed phrase arc and coda taper are then applied. Synchrony control 0.366 influences drum accents.
- **Phrasing:** persistence 0.458 changes note and chord sustain. Trial variance 0.271 sets melody/comping timing variation to **±0.0119 beats**. Rests, four-bar phrase starts, melodic landing rules, and the broad form remain programmed.

The resulting score contains **1,054 note-on events** across its instruments. This is an output of both the fixed arrangement and the neural controls, not a direct translation of one spike into one note.

## 5. Sunset

**5:12 · 80 BPM · 104 bars · 4/4**

### Choices set in the composition code

**Jazz vocabulary.** Minor jazz with a D Dorian melodic palette: D, E, F, G, A, B, C. Minor sixth/ninth sonorities, borrowed major/minor colours, and an altered dominant create departures from and returns to D minor.

**Chord options.** Dm9, Bbmaj9, Gm9, Em7b5, A7alt, Dm6/9, Fmaj9, Am7, Eb13, C13, Bbm6, G13.

**Opening eight theme bars.** Dm9 → Dm9 → Bbmaj9 → Gm9 → Em7b5 → A7alt → Dm6/9 → Dm6/9. This theme progression is prescribed by the form; neural choices affect other passages and its realization.

**Instruments.** Acoustic piano and muted trumpet share the foreground. Upright bass, vibraphone, electric-piano washes in quieter sections, and deliberately sparse drums complete the arrangement.

**Song structure.** Section lengths and order were fixed before applying the neural controls. “Piano solo” uses electric piano where specified above.

| Section | Role | Bars | Length |
|---|---|---:|---:|
| Low sun | Introduction | 1–8 | 8 bars |
| Amber theme | Theme | 9–32 | 24 bars |
| Long shadows | Piano solo | 33–56 | 24 bars |
| Copper water | Lead solo | 57–72 | 16 bars |
| The horizon | Interlude | 73–80 | 8 bars |
| Afterglow | Theme return | 81–96 | 16 bars |
| Violet air | Coda | 97–104 | 8 bars |

### What the neural responses changed

Sunset has high overall activity, but its fixed section energies and sparse drum rule keep it restrained. Low persistence gives relatively short neural sustain settings; the longer track length comes from the chosen 104 bars at 80 BPM. It has the smallest trial-variance timing allowance.

- **Melody:** the shared Sunrise-derived motif is fitted to this track’s scale and chords. Sunset’s local population rankings guide solo contour and selected reprise notes; pitch range, chord-tone landing, and step-size limits remain programmed constraints.
- **Harmony:** local group activity and scene entropy (0.511 on the album’s 0–1 scale) select among the allowed progressions outside fixed passages. Active-neuron control 0.536 selects **4-note** rootless comping voicings.
- **Rhythm:** burstiness 0.144 adjusts the preset 62.0% swing to **61.57%**. Local group counts choose accompaniment patterns; burst-profile peaks shift eligible phrases by 0, ¼, or ½ beat. Drum fills also require an eligible section and position.
- **Dynamics:** overall activity control 0.921 contributes to melody/accompaniment velocity and note activity. Each image’s spike count multiplies the preset section energy by **0.85–1.15**; a programmed phrase arc and coda taper are then applied. Synchrony control 0.155 influences drum accents.
- **Phrasing:** persistence 0.080 changes note and chord sustain. Trial variance 0.000 sets melody/comping timing variation to **±0.0070 beats**. Rests, four-bar phrase starts, melodic landing rules, and the broad form remain programmed.

The resulting score contains **1,311 note-on events** across its instruments. This is an output of both the fixed arrangement and the neural controls, not a direct translation of one spike into one note.

## 6. Crescent Moon

**5:28 · 60 BPM · 82 bars · 4/4**

### Choices set in the composition code

**Jazz vocabulary.** Spacious minor jazz with an E Dorian melodic palette: E, F#, G, A, B, C#, D. Minor sixth/ninth chords, half-diminished chords, and altered dominants provide gentle tension and resolution.

**Chord options.** Em9, Cmaj9, Am9, F#m7b5, B7alt, Em6/9, Gmaj9, Bm7, F13, D13, Cm6, A13.

**Opening eight theme bars.** Em9 → Em9 → Cmaj9 → Am9 → F#m7b5 → B7alt → Em6/9 → Em6/9. This theme progression is prescribed by the form; neural choices affect other passages and its realization.

**Instruments.** Electric piano carries the piano parts and vibraphone is the selected lead instrument. Upright bass, separate vibraphone responses, electric-piano washes, and deliberately sparse drums keep the arrangement spacious.

**Song structure.** Section lengths and order were fixed before applying the neural controls. “Piano solo” uses electric piano where specified above.

| Section | Role | Bars | Length |
|---|---|---:|---:|
| Blue hour | Introduction | 1–8 | 8 bars |
| Silver crescent | Theme | 9–24 | 16 bars |
| Night windows | Piano solo | 25–40 | 16 bars |
| Still water | Lead solo | 41–50 | 10 bars |
| Dark sky | Interlude | 51–58 | 8 bars |
| Moonlight returns | Theme return | 59–74 | 16 bars |
| Goodnight | Coda | 75–82 | 8 bars |

### What the neural responses changed

Crescent Moon combines the lowest mean spike count with the highest persistence and trial-variance controls. That reduces activity-dependent note density, increases sustain, and widens bounded timing variation. Its 60 BPM tempo, vibraphone lead, and long form were chosen independently of those measurements.

- **Melody:** the shared Sunrise-derived motif is fitted to this track’s scale and chords. Crescent Moon’s local population rankings guide solo contour and selected reprise notes; pitch range, chord-tone landing, and step-size limits remain programmed constraints.
- **Harmony:** local group activity and scene entropy (0.145 on the album’s 0–1 scale) select among the allowed progressions outside fixed passages. Active-neuron control 0.101 selects **3-note** rootless comping voicings.
- **Rhythm:** burstiness 0.863 adjusts the preset 61.0% swing to **61.44%**. Local group counts choose accompaniment patterns; burst-profile peaks shift eligible phrases by 0, ¼, or ½ beat. Drum fills also require an eligible section and position.
- **Dynamics:** overall activity control 0.000 contributes to melody/accompaniment velocity and note activity. Each image’s spike count multiplies the preset section energy by **0.85–1.15**; a programmed phrase arc and coda taper are then applied. Synchrony control 0.758 influences drum accents.
- **Phrasing:** persistence 1.000 changes note and chord sustain. Trial variance 1.000 sets melody/comping timing variation to **±0.0250 beats**. Rests, four-bar phrase starts, melodic landing rules, and the broad form remain programmed.

The resulting score contains **873 note-on events** across its instruments. This is an output of both the fixed arrangement and the neural controls, not a direct translation of one spike into one note.

## 7. Bombolini

**4:32 · 120 BPM · 136 bars · 4/4**

### Choices set in the composition code

**Jazz vocabulary.** Major-colour jazz with a Bb Lydian melodic palette: Bb, C, D, E, F, G, A. Major ninths, ii–V movement, borrowed minor chords, and altered dominants give the closing piece its harmonic options.

**Chord options.** Bbmaj9, Am7, D7alt, Gm9, C13, Cm9, F13, Dm9, Ebmaj9, Ebm6, Dm7, G7alt, Fmaj9, Ab13.

**Opening eight theme bars.** Bbmaj9 → Bbmaj9 → Am7 → D7alt → Gm9 → C13 → Cm9 → F13. This theme progression is prescribed by the form; neural choices affect other passages and its realization.

**Instruments.** Acoustic piano carries the opening theme and a long solo; vibraphone takes the dedicated second solo section. Clarinet appears in selected theme/reprise phrases. Upright bass walks in designated active passages, supported by jazz drums and occasional electric-piano washes.

**Song structure.** Section lengths and order were fixed before applying the neural controls. “Piano solo” uses electric piano where specified above.

| Section | Role | Bars | Length |
|---|---|---:|---:|
| Bakery door | Introduction | 1–8 | 8 bars |
| Sugar theme | Theme | 9–32 | 24 bars |
| Custard conversation | Piano solo | 33–64 | 32 bars |
| Espresso break | Vibraphone solo | 65–88 | 24 bars |
| Crumbs | Interlude | 89–104 | 16 bars |
| One more bite | Theme return | 105–128 | 24 bars |
| Sweet goodbye | Coda | 129–136 | 8 bars |

### What the neural responses changed

Bombolini has relatively low mean activity but the highest burstiness and synchrony controls. The latter increase swing and drum accents/fill eligibility. Its lively character also depends on the deliberately chosen 120 BPM, walking-bass rules, bright harmonic palette, and long piano/vibraphone passages.

- **Melody:** the shared Sunrise-derived motif is fitted to this track’s scale and chords. Bombolini’s local population rankings guide solo contour and selected reprise notes; pitch range, chord-tone landing, and step-size limits remain programmed constraints.
- **Harmony:** local group activity and scene entropy (0.438 on the album’s 0–1 scale) select among the allowed progressions outside fixed passages. Active-neuron control 0.256 selects **3-note** rootless comping voicings.
- **Rhythm:** burstiness 1.000 adjusts the preset 66.0% swing to **66.60%**. Local group counts choose accompaniment patterns; burst-profile peaks shift eligible phrases by 0, ¼, or ½ beat. Drum fills also require an eligible section and position.
- **Dynamics:** overall activity control 0.135 contributes to melody/accompaniment velocity and note activity. Each image’s spike count multiplies the preset section energy by **0.85–1.15**; a programmed phrase arc and coda taper are then applied. Synchrony control 1.000 influences drum accents.
- **Phrasing:** persistence 0.877 changes note and chord sustain. Trial variance 0.540 sets melody/comping timing variation to **±0.0167 beats**. Rests, four-bar phrase starts, melodic landing rules, and the broad form remain programmed.

The resulting score contains **2,047 note-on events** across its instruments. This is an output of both the fixed arrangement and the neural controls, not a direct translation of one spike into one note.

## What the validation establishes

The saved checks confirm that changing each of the 105 individual image-response profiles changes its song’s MIDI score, and that the same recorded inputs reproduce the same scores. The individual-image check perturbs a recorded spike-count input; it does not test whether the network understands the image or whether every feature has an independently audible effect.

The honest attribution is **AI-written algorithmic composition shaped by simulated fly-brain activity**. The user supplied the scenes, duration constraints, artist name, and album direction. The assistant supplied the visual prompts, musical framework, mapping rules, and implementation; the simulation supplied the responses used inside that framework.

[Listen on SoundCloud](https://soundcloud.com/ajay-kumar-819538867/sets/days) · [WAV downloads](https://github.com/thatlawyerfellow/flymusicengine/releases/tag/days-v1.0) · [Return to the project](../../README.md)
