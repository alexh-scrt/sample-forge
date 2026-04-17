# sample_forge

> Offline BPM and key retargeting for audio samples — no API keys, no subscriptions.

`sample_forge` is a lightweight command-line tool for music producers that lets you **time-stretch** and **pitch-shift** any audio sample to a target BPM and/or musical key. It uses [librosa](https://librosa.org/) for automatic beat detection and key estimation, and the [Rubber Band Library](https://breakfastquay.com/rubberband/) (via [pyrubberband](https://github.com/bmcfee/pyrubberband)) for high-quality, artefact-free processing — giving you a free, fully-offline alternative to subscription tools like Splice Variations.

---

## Quick Start

**Install:**

```bash
pip install sample_forge
```

> **Note:** `pyrubberband` requires the native Rubber Band Library. Install it first:
> - **macOS:** `brew install rubberband`
> - **Ubuntu/Debian:** `sudo apt install librubberband-dev`
> - **Windows:** Download from [breakfastquay.com/rubberband](https://breakfastquay.com/rubberband/)

**Basic usage:**

```bash
# Retarget a sample to 140 BPM
sample-forge -i break.wav -o break_140.wav --bpm 140

# Shift a sample to A minor
sample-forge -i riff.wav -o riff_Am.wav --key Am

# Both at once, export as FLAC
sample-forge -i loop.wav -o loop_out.flac --bpm 128 --key C --format FLAC
```

That's it — drop the output file straight into your DAW.

---

## Features

- **Automatic BPM detection** — librosa beat tracking analyses your sample's tempo, with an optional `--source-bpm` override if you already know it
- **Musical key estimation & pitch shifting** — chroma-based key detection plus semitone-precise transposition to any target key (e.g. `C`, `F#m`, `Bbmaj`)
- **High-quality time-stretching** — powered by the Rubber Band Library via `pyrubberband`, preserving timbre and transients without the warbling artefacts of simple resampling
- **Flexible export** — WAV or FLAC output with configurable sample rate; files are immediately DAW-ready
- **Fully offline and free** — no internet connection, no API keys, no subscriptions; runs on any machine with Python 3.9+

---

## Usage Examples

### Retarget BPM only

```bash
# Auto-detect source BPM, stretch to 170 BPM
sample-forge -i amen_break.wav -o amen_170.wav --bpm 170
```

### Shift key only

```bash
# Auto-detect source key, shift to F# minor
sample-forge -i chord_stab.wav -o chord_stab_Fsm.wav --key F#m
```

### Full retarget: BPM + key

```bash
sample-forge -i loop.wav -o loop_128_Cm.wav --bpm 128 --key Cm
```

### Override auto-detection

```bash
# Skip analysis, specify source values manually (faster)
sample-forge -i break.wav -o break_160.wav \
  --bpm 160 \
  --source-bpm 132 \
  --source-key Am \
  --key Dm
```

### Export as FLAC at a specific sample rate

```bash
sample-forge -i synth_loop.wav -o synth_loop.flac \
  --bpm 140 --key G --format FLAC --sr 48000
```

### Python API

```python
import librosa
from sample_forge.analyzer import detect_bpm_and_key
from sample_forge.processor import process
from sample_forge.audio_io import load_audio, write_audio

audio, sr = load_audio("break.wav")
source_bpm, source_key = detect_bpm_and_key(audio, sr)

processed = process(
    audio, sr,
    source_bpm=source_bpm, target_bpm=140.0,
    source_key=source_key, target_key="Am",
)

write_audio("break_140_Am.wav", processed, sr)
```

---

## CLI Reference

```
Usage: sample-forge [OPTIONS]

Options:
  -i, --input PATH         Input audio file (WAV, FLAC, MP3, etc.)  [required]
  -o, --output PATH        Output file path  [required]
  --bpm FLOAT              Target BPM
  --key TEXT               Target key (e.g. C, F#m, Bbmaj)
  --source-bpm FLOAT       Override auto-detected source BPM
  --source-key TEXT        Override auto-detected source key
  --format [WAV|FLAC]      Output format (default: inferred from extension)
  --sr INTEGER             Output sample rate (default: preserve source)
  --version                Show version and exit
  --help                   Show this message and exit
```

---

## Key Name Reference

`sample_forge` accepts standard key names in the format `<note>[m|maj|min]`:

| Format | Examples | Meaning |
|--------|----------|---------|
| `<Note>` | `C`, `F#`, `Bb` | Major key |
| `<Note>m` or `<Note>min` | `Am`, `F#m`, `Bbmin` | Minor key |
| `<Note>maj` | `Cmaj`, `Gmaj` | Major key (explicit) |

Supported accidentals: `#` (sharp) and `b` (flat). Enharmonic equivalents (`C#` / `Db`) are treated as identical.

---

## Output Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| WAV | `.wav` | Default; 32-bit float or 16-bit PCM |
| FLAC | `.flac` | Lossless compression; use `--format FLAC` or `.flac` extension |

The output format is inferred from the output file extension. Use `--format` to override.

---

## Project Structure

```
sample_forge/
├── pyproject.toml          # Project metadata, dependencies, CLI entry point
├── README.md
├── sample_forge/
│   ├── __init__.py         # Package init, version string
│   ├── cli.py              # Click CLI entry point (--bpm, --key, --input, --output)
│   ├── analyzer.py         # BPM detection and key estimation (librosa)
│   ├── processor.py        # Time-stretching and pitch-shifting (pyrubberband)
│   ├── audio_io.py         # Audio loading and WAV/FLAC writing
│   └── key_utils.py        # Key name parsing and semitone offset math
└── tests/
    ├── __init__.py
    ├── test_analyzer.py     # BPM/key analysis unit tests
    ├── test_processor.py    # Stretch ratio and pitch-shift math tests
    ├── test_key_utils.py    # Key parsing and validation tests
    ├── test_audio_io.py     # Audio I/O tests (temp files)
    └── test_cli.py          # CLI integration tests (CliRunner)
```

---

## Configuration

`sample_forge` has no config file — all options are passed as CLI flags or API arguments. The table below summarises the tuneable parameters:

| Parameter | CLI Flag | Default | Description |
|-----------|----------|---------|-------------|
| Target BPM | `--bpm` | *(none)* | Desired output tempo; skipped if omitted |
| Target key | `--key` | *(none)* | Desired output key; skipped if omitted |
| Source BPM | `--source-bpm` | Auto-detected | Override librosa beat tracking |
| Source key | `--source-key` | Auto-detected | Override librosa key estimation |
| Sample rate | `--sr` | Preserve source | Resample output to this rate (Hz) |
| Output format | `--format` | Inferred from extension | `WAV` or `FLAC` |

---

## Requirements

- Python ≥ 3.9
- [Rubber Band Library](https://breakfastquay.com/rubberband/) (native, required by `pyrubberband`)
- Python packages (installed automatically via pip):
  - `librosa >= 0.10`
  - `pyrubberband >= 0.3`
  - `soundfile >= 0.12`
  - `numpy >= 1.24`
  - `click >= 8.1`

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/
```

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

*Built with [Jitter](https://github.com/jitter-ai) — an AI agent that ships code daily.*
