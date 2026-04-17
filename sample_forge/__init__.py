"""sample_forge: Offline time-stretching and pitch-shifting for audio samples.

This package provides a command-line interface and reusable Python API for
retargeting the BPM and musical key of audio samples without quality-destroying
simple resampling. It leverages librosa for audio analysis and pyrubberband
for high-quality time-stretching and pitch-shifting.

Typical usage::

    $ sample-forge --input break.wav --bpm 140 --key Am --output break_140_Am.wav
"""

__version__ = "0.1.0"
__author__ = "sample_forge contributors"
__license__ = "MIT"

__all__ = ["__version__"]
