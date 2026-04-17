# sample_forge

> Offline BPM and key retargeting for audio samples — no API keys, no subscriptions.

`sample_forge` is a lightweight command-line tool for music producers that lets you **time-stretch** and **pitch-shift** any audio sample to a target BPM and/or musical key. It uses [librosa](https://librosa.org/) for automatic beat detection and key estimation, and the [Rubber Band Library](https://breakfastquay.com/rubberband/) (via [pyrubberband](https://github.com/bmcfee/pyrubberband)) for high-quality, artefact-free processing.

The result is a free, fully-offline alternative to subscription tools like Splice Variations — export-ready WAV or FLAC files that drop straight into your DAW.

---

## Table of Contents

1. [Features](#features)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Usage Reference](#usage-reference)
6. [Examples](#examples)
7. [Output Formats](#output-formats)
8. [Key Name Reference](#key-name-reference)
9. [How It Works](#how-it-works)
10. [Python API](#python-api)
11. [Development & Testing](#development--testing)
12. [Troubleshooting](#troubleshooting)
13. [License](#license)

---

## Features

- **Automatic BPM detection** using librosa's beat tracker — or supply a known BPM with `--source-bpm` to skip analysis.
- **Musical key estimation** from chroma (CENS) features with the Krumhansl–Schmuckler algorithm — or override with `--source-key`.
- **High-quality time-stretching** via the Rubber Band Library — preserves timbre and transients, no robotic artefacts.
- **Semitone-accurate pitch shifting** to any of the 12 chromatic keys, major or minor.
- **Flexible output** — WAV (PCM 16-bit default) or FLAC (PCM 24-bit default), auto-detected from the output file extension.
- **Fully offline** — no network calls, no accounts, runs on any machine with Python ≥ 3.9.

---

## Requirements

| Dependency | Version | Purpose |
|---|---|---|
| Python | ≥ 3.9 | Runtime |
| [librosa](https://librosa.org/) | ≥ 0.10 | BPM detection, key estimation |
| [pyrubberband](https://github.com/bmcfee/pyrubberband) | ≥ 0.3 | Time-stretching & pitch-shifting |
| [soundfile](https://python-soundfile.readthedocs.io/) | ≥ 0.12 | Audio file I/O |
| [numpy](https://numpy.org/) | ≥ 1.24 | Numerical arrays |
| [click](https://click.palletsprojects.com/) | ≥ 8.1 | CLI framework |

### System dependency: Rubber Band

`pyrubberband` requires the **Rubber Band** command-line binary (`rubberband`) to be installed and available on your `PATH`.

```bash
# macOS (Homebrew)
brew install rubberband

# Ubuntu / Debian
sudo apt-get install rubberband-cli

# Arch Linux
sudo pacman -S rubberband

# Windows
# Download the binary from https://breakfastquay.com/rubberband/
# and add it to your PATH.
```

---

## Installation

### From source (recommended while in alpha)

```bash
# Clone the repository
git clone https://github.com/yourname/sample_forge.git
cd sample_forge

# Create and activate a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
.venv\Scripts\activate.bat     # Windows

# Install in editable mode with all dependencies
pip install -e .
```

### From PyPI (once published)

```bash
pip install sample_forge
```

After installation the `sample-forge` command will be available in your shell:

```bash
sample-forge --version
# sample-forge, version 0.1.0
```

---

## Quick Start

```bash
# Stretch a drum loop to 140 BPM (auto-detects the source BPM)
sample-forge -i break.wav -o break_140bpm.wav --bpm 140

# Transpose a synth riff from C to A minor
sample-forge -i riff_C.wav -o riff_Am.wav --key Am --source-key C

# Do both in one pass and export as FLAC
sample-forge -i loop.wav -o loop_140_Am.flac --bpm 140 --key Am
```

---

## Usage Reference

```
Usage: sample-forge [OPTIONS]

  sample-forge — offline BPM and key retargeting for audio samples.

Options:
  -i, --input PATH          Path to the source audio file (WAV, FLAC, MP3,
                            OGG, …).  [required]
  -o, --output PATH         Destination file path for the processed audio.
                            [required]
  --bpm FLOAT               Target BPM. If omitted, BPM is not changed.
  --key TEXT                Target musical key (e.g. 'C', 'F#m', 'Bbmaj').
                            If omitted, key is not changed.
  --format [WAV|FLAC|wav|flac]
                            Output format. Defaults to the extension of
                            --output, or WAV if the extension is not
                            recognised.
  --sr INTEGER              Resample the source audio to this sample rate
                            before processing (e.g. 44100, 48000). Uses the
                            native sample rate if omitted.
  --source-bpm FLOAT        Manually specify the source BPM instead of
                            auto-detecting it. Only used when --bpm is also
                            set.
  --source-key TEXT         Manually specify the source key instead of
                            auto-detecting it. Only used when --key is also
                            set.
  -V, --version             Show the version and exit.
  -h, --help                Show this message and exit.
```

### Option details

| Option | Default | Notes |
|---|---|---|
| `-i` / `--input` | *(required)* | Any format supported by librosa / soundfile: WAV, FLAC, MP3, OGG, AIFF, … |
| `-o` / `--output` | *(required)* | Parent directories are created automatically if they do not exist. |
| `--bpm` | *(none — no stretch)* | Target tempo in beats per minute. Must be > 0. |
| `--key` | *(none — no shift)* | Target key. See [Key Name Reference](#key-name-reference). |
| `--format` | Auto from extension | Explicit format override. `WAV` or `FLAC` (case-insensitive). |
| `--sr` | Native file SR | Resample before processing. Useful for normalising sample rates across a project. |
| `--source-bpm` | Auto-detected | Supply when you already know the source BPM to skip the beat tracker. |
| `--source-key` | Auto-detected | Supply when you already know the source key to skip the key estimator. |

---

## Examples

### 1 — Time-stretch only (auto-detect source BPM)

```bash
sample-forge -i amen_break.wav -o amen_170bpm.wav --bpm 170
```

Output:
```
Loading amen_break.wav …
  Loaded 2.91 s  |  44100 Hz  |  mono
  Detecting source BPM …
  Detected BPM: 136.17
Processing …
  Output duration: 2.32 s
  BPM: 136.17 → 170.00  (ratio 0.8010)
Writing amen_170bpm.wav (WAV) …
Done.
```

### 2 — Pitch-shift only (provide known source key)

```bash
sample-forge -i bass_C.wav -o bass_Gm.wav --key Gm --source-key C
```

### 3 — Both transforms at once, FLAC output

```bash
sample-forge \
  -i vocal_chop.wav \
  -o vocal_chop_140_Dm.flac \
  --bpm 140 \
  --key Dm \
  --source-bpm 128 \
  --source-key Am
```

### 4 — Resample to 44.1 kHz before processing

```bash
sample-forge -i old_sample_22khz.wav -o out_44k.wav --bpm 120 --sr 44100
```

### 5 — Batch processing with a shell loop

```bash
for f in loops/*.wav; do
  out="output/$(basename "${f%.wav}")_140bpm.wav"
  sample-forge -i "$f" -o "$out" --bpm 140
done
```

### 6 — Python API usage

```python
from sample_forge import audio_io, analyzer, processor

# Load audio
audio, sr = audio_io.load_audio("break.wav", mono=True)

# Analyse
detected_bpm = analyzer.detect_bpm(audio, sr)
pitch_class, mode = analyzer.estimate_key(audio, sr)
print(f"Detected: {detected_bpm:.1f} BPM, key pc={pitch_class} ({mode})")

# Process
processed = processor.process(
    audio, sr,
    source_bpm=detected_bpm, target_bpm=140.0,
    source_key="Am", target_key="Cm",
)

# Save
audio_io.write_audio("output.flac", processed, sr, fmt="FLAC")
```

---

## Output Formats

| Format | Default subtype | Notes |
|---|---|---|
| `WAV` | PCM 16-bit | Best compatibility with older DAWs and hardware samplers. |
| `FLAC` | PCM 24-bit | Lossless compression, ~50–60 % smaller than equivalent WAV. Preferred for archiving. |

The format is determined in the following priority order:

1. The `--format` flag (explicit override).
2. The file extension of `--output` (`.wav` → WAV, `.flac` → FLAC).
3. `WAV` as the default fallback for all other or missing extensions.

You can also request a specific bit-depth subtype via the Python API (`subtype='PCM_24'` for WAV, `subtype='PCM_16'` for FLAC, etc.).

---

## Key Name Reference

`sample_forge` accepts a flexible range of key name notations:

| Notation | Meaning | Examples |
|---|---|---|
| `<Note>` | Major key (bare note) | `C`, `F#`, `Bb` |
| `<Note>m` | Minor key | `Am`, `C#m`, `Ebm` |
| `<Note>M` | Major key (uppercase M) | `DM`, `F#M` |
| `<Note>maj` | Major key | `Cmaj`, `Gbmaj` |
| `<Note>major` | Major key | `Cmajor`, `Abmajor` |
| `<Note>min` | Minor key | `Amin`, `F#min` |
| `<Note>minor` | Minor key | `Aminor`, `Ebminor` |

Note names support both sharps (`#`) and flats (`b`):

```
C  C#/Db  D  D#/Eb  E  F  F#/Gb  G  G#/Ab  A  A#/Bb  B
```

Enharmonic equivalents are treated identically (`C#` = `Db`, `F#` = `Gb`, etc.).

The pitch-shift amount is calculated as the **shortest path** around the chromatic circle, so the semitone offset is always in **[-6, +6]**. For example, `C → G` is `-5` (down a perfect fifth) rather than `+7` (up a perfect fifth), because -5 is a smaller absolute shift.

---

## How It Works

### Pipeline overview

```
Input file
    │
    ▼
[audio_io.load_audio]
    │  float32 mono array at native (or --sr) sample rate
    ▼
[analyzer.detect_bpm]     (only if --bpm is set and --source-bpm not given)
[analyzer.estimate_key]   (only if --key is set and --source-key not given)
    │
    ▼
[processor.time_stretch]  (only if --bpm is set)
    │  ratio = source_bpm / target_bpm
    ▼
[processor.pitch_shift]   (only if --key is set)
    │  semitones = shortest_path(source_key → target_key)
    ▼
[audio_io.write_audio]
    │
    ▼
Output file (WAV / FLAC)
```

### BPM detection

Librosa's `beat_track` function computes an onset-strength envelope from the audio's spectral flux and then applies a dynamic-programming beat tracker. The returned tempo is used as `source_bpm`; the stretch ratio is simply `source_bpm / target_bpm`.

- A ratio **> 1** means the audio is stretched (slowed down) — more samples per beat.
- A ratio **< 1** means the audio is compressed (sped up) — fewer samples per beat.

### Key estimation

Chroma Energy Normalised Statistics (CENS) features are extracted and averaged over time to produce a 12-element chroma vector. The Krumhansl–Schmuckler algorithm correlates this vector against major and minor key profiles rotated through all 12 pitch classes, selecting the best-matching key.

### Pitch shifting

The semitone offset between source and target root is computed as the shortest chromatic distance (range [-6, +6]). Rubber Band's phase-vocoder pitch shifter is then applied. Duration is preserved exactly.

### Time stretching

Rubber Band uses a phase-vocoder approach with transient detection and phase-locking. It produces natural-sounding results on melodic and percussive material at ratios from ~0.5× to ~2×. Extreme ratios (outside ~[0.25, 4.0]) may introduce artefacts.

---

## Python API

All core functionality is accessible as a Python library.

### `sample_forge.audio_io`

```python
from sample_forge.audio_io import load_audio, write_audio, get_audio_info

# Load (mono, native sample rate)
audio, sr = load_audio("sample.wav")

# Load (stereo, resample to 44100 Hz)
audio, sr = load_audio("sample.wav", target_sr=44100, mono=False)

# Inspect without decoding
info = get_audio_info("sample.wav")
print(info)  # {'sample_rate': 44100, 'channels': 2, 'frames': 88200,
             #  'duration': 2.0, 'format': 'WAV', 'subtype': 'PCM_16'}

# Write
write_audio("out.flac", audio, sr, fmt="FLAC", subtype="PCM_24")
```

### `sample_forge.analyzer`

```python
from sample_forge.analyzer import detect_bpm, estimate_key, detect_bpm_and_key

bpm = detect_bpm(audio, sr)               # float, e.g. 136.17
pc, mode = estimate_key(audio, sr)        # (9, 'minor') = A minor
bpm, pc, mode = detect_bpm_and_key(audio, sr)  # convenience wrapper
```

### `sample_forge.processor`

```python
from sample_forge.processor import (
    compute_time_stretch_ratio,
    compute_pitch_shift_semitones,
    time_stretch,
    pitch_shift,
    process,
)

ratio = compute_time_stretch_ratio(120.0, 140.0)   # 0.857...
semitones = compute_pitch_shift_semitones("Am", "Cm")  # 3

stretched = time_stretch(audio, sr, ratio=0.857)
shifted   = pitch_shift(audio, sr, semitones=3)

# All in one
out = process(audio, sr,
              source_bpm=120.0, target_bpm=140.0,
              source_key="Am",  target_key="Cm")
```

### `sample_forge.key_utils`

```python
from sample_forge.key_utils import (
    parse_key, validate_key, semitone_offset,
    key_to_pitch_class, pitch_class_to_name, all_key_names,
)

parse_key("F#m")           # (6, 'minor')
validate_key("Bbmaj")      # True
validate_key("ZZZ")        # False
semitone_offset("C", "E")  # 4
pitch_class_to_name(9)     # 'A'
all_key_names("minor")     # ['Cm', 'C#m', 'Dm', ...]
```

---

## Development & Testing

### Set up the development environment

```bash
git clone https://github.com/yourname/sample_forge.git
cd sample_forge
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"   # or: pip install -e . pytest
```

### Run the test suite

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=sample_forge --cov-report=term-missing

# Run a specific test file
pytest tests/test_key_utils.py -v

# Run only fast (non-I/O) tests
pytest tests/test_key_utils.py tests/test_processor.py -v
```

### Test structure

| File | Coverage |
|---|---|
| `tests/test_key_utils.py` | Key parsing, semitone math, validation — no audio I/O |
| `tests/test_audio_io.py` | Load/write WAV & FLAC, format validation, metadata |
| `tests/test_analyzer.py` | BPM detection, key estimation, Krumhansl–Schmuckler |
| `tests/test_processor.py` | Stretch ratio, semitone calculation, time-stretch, pitch-shift, `process()` |
| `tests/test_cli.py` | CLI integration via Click's `CliRunner` |
| `tests/test_smoke.py` | Import sanity, public API surface |

### Code style

The codebase follows PEP 8 with full type hints. Run a formatter and linter:

```bash
pip install ruff black
black sample_forge tests
ruff check sample_forge tests
```

---

## Troubleshooting

### `rubberband` not found

```
RuntimeError: Time-stretching failed …: [Errno 2] No such file or directory: 'rubberband'
```

Install the Rubber Band binary for your platform (see [Requirements](#requirements)).
Verify it is on your PATH:

```bash
rubberband --version
```

### `soundfile` / `libsndfile` error loading MP3

soundfile does not support MP3 natively. librosa uses `audioread` as a fallback for MP3 and other lossy formats. If loading fails:

```bash
pip install audioread
# On macOS, also: brew install ffmpeg
# On Linux:       sudo apt-get install ffmpeg
```

### BPM detection gives wrong result

Librosa's beat tracker may detect half-time or double-time tempos for complex rhythms. Use `--source-bpm` to supply the correct BPM manually:

```bash
sample-forge -i loop.wav -o out.wav --bpm 140 --source-bpm 70
```

### Key estimation sounds wrong

Automatic key detection from short samples is inherently uncertain. Use `--source-key` to override:

```bash
sample-forge -i chord.wav -o out.wav --key Am --source-key F#m
```

### Output sounds artefacted at extreme ratios

Rubber Band works best for time-stretch ratios between approximately 0.5× and 2.0× (one octave of tempo range). Very extreme ratios (e.g. stretching a 70 BPM loop to 200 BPM) will produce audible processing artefacts — this is a fundamental limitation of current phase-vocoder technology.

---

## License

MIT License — see [LICENSE](LICENSE) for full text.

This project uses:
- [librosa](https://github.com/librosa/librosa) — ISC License
- [pyrubberband](https://github.com/bmcfee/pyrubberband) — ISC License
- [Rubber Band Library](https://breakfastquay.com/rubberband/) — GPL v2 (CLI binary) / commercial licence
- [soundfile](https://github.com/bastibe/python-soundfile) — BSD 3-Clause
- [numpy](https://numpy.org/) — BSD 3-Clause
- [click](https://palletsprojects.com/p/click/) — BSD 3-Clause

> **Note on Rubber Band licensing:** The `rubberband-cli` binary used by pyrubberband is GPL v2. If you are distributing a commercial product, you may need to purchase a commercial licence from Breakfast Quay.
