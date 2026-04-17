"""audio_io: Audio file loading and writing utilities.

Provides thin wrappers around librosa and soundfile for loading audio samples
into numpy arrays and writing processed arrays back to WAV or FLAC files.

All audio is handled as mono or stereo float32 numpy arrays internally. The
sample rate is always preserved or resampled explicitly — no silent resampling
happens inside this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_audio(
    path: str | Path,
    target_sr: int | None = None,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """Load an audio file into a numpy float32 array.

    Uses librosa under the hood, which supports a wide range of formats
    (WAV, FLAC, MP3, OGG, AIFF, etc.) via soundfile and audioread.

    Args:
        path: Path to the source audio file.
        target_sr: If provided, resample the audio to this sample rate. If
            ``None``, the native sample rate of the file is used.
        mono: If ``True`` (default), the audio is mixed down to mono. If
            ``False``, the original channel layout is preserved.

    Returns:
        A tuple ``(audio, sample_rate)`` where *audio* is a float32 numpy
        array of shape ``(samples,)`` for mono or ``(channels, samples)`` for
        multi-channel audio, and *sample_rate* is the integer sample rate.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file cannot be decoded as audio.
        RuntimeError: For unexpected librosa/soundfile errors.
    """
    import librosa  # deferred so stubs import without heavy deps

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    try:
        audio, sr = librosa.load(
            str(path),
            sr=target_sr,
            mono=mono,
            dtype=np.float32,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load audio from {path}: {exc}"
        ) from exc

    return audio, int(sr)


def write_audio(
    path: str | Path,
    audio: np.ndarray,
    sample_rate: int,
    fmt: str = "WAV",
    subtype: str | None = None,
) -> None:
    """Write a numpy audio array to a file.

    Supports WAV and FLAC output via soundfile.

    Args:
        path: Destination file path. The parent directory must already exist.
        audio: Float32 numpy array of shape ``(samples,)`` (mono) or
            ``(channels, samples)`` (multi-channel).  The array will be
            transposed if necessary so soundfile receives ``(samples,)`` or
            ``(samples, channels)``.
        sample_rate: Sample rate in Hz.
        fmt: Output format string, either ``'WAV'`` or ``'FLAC'``. Case
            insensitive.
        subtype: soundfile subtype string (e.g. ``'PCM_16'``, ``'PCM_24'``).
            If ``None``, a sensible default is chosen per format:
            ``'PCM_16'`` for WAV and ``'PCM_24'`` for FLAC.

    Raises:
        ValueError: If *fmt* is not ``'WAV'`` or ``'FLAC'``, or if
            *sample_rate* is not positive.
        OSError: If the destination file cannot be written.
        RuntimeError: For unexpected soundfile errors.
    """
    import soundfile as sf  # deferred import

    path = Path(path)
    fmt_upper = fmt.upper()
    if fmt_upper not in ("WAV", "FLAC"):
        raise ValueError(
            f"Unsupported output format {fmt!r}. Choose 'WAV' or 'FLAC'."
        )
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be a positive integer, got {sample_rate}.")

    # Determine default subtype
    if subtype is None:
        subtype = "PCM_16" if fmt_upper == "WAV" else "PCM_24"

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # soundfile expects shape (samples,) or (samples, channels)
    audio_out = _prepare_array_for_write(audio)

    try:
        sf.write(
            str(path),
            audio_out,
            sample_rate,
            subtype=subtype,
            format=fmt_upper,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to write audio to {path}: {exc}"
        ) from exc


def get_supported_formats() -> list[str]:
    """Return a list of output format strings supported by this module.

    Returns:
        A list of uppercase format strings, currently ``['WAV', 'FLAC']``.
    """
    return ["WAV", "FLAC"]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _prepare_array_for_write(audio: np.ndarray) -> np.ndarray:
    """Reshape / transpose *audio* so soundfile can write it.

    soundfile expects:
      - 1-D array for mono: shape ``(samples,)``
      - 2-D array for multi-channel: shape ``(samples, channels)``

    librosa returns multi-channel audio as ``(channels, samples)``, so we
    need to transpose.

    Args:
        audio: Input numpy array.

    Returns:
        Array ready for ``soundfile.write``.

    Raises:
        ValueError: If *audio* has more than 2 dimensions.
    """
    if audio.ndim == 1:
        return audio.astype(np.float32)
    if audio.ndim == 2:
        # Decide orientation: if shape is (channels, samples) librosa-style,
        # transpose to (samples, channels). We heuristically assume that the
        # larger dimension is 'samples'.
        rows, cols = audio.shape
        if rows < cols:
            # (channels, samples) -> (samples, channels)
            return audio.T.astype(np.float32)
        # Already (samples, channels)
        return audio.astype(np.float32)
    raise ValueError(
        f"audio array must be 1-D or 2-D, got shape {audio.shape}."
    )
