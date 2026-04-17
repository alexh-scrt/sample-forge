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
        ValueError: If the file cannot be decoded as audio or if *target_sr*
            is not a positive integer when provided.
        RuntimeError: For unexpected librosa/soundfile errors.
    """
    import librosa  # deferred so stubs import without heavy deps

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    if target_sr is not None and target_sr <= 0:
        raise ValueError(
            f"target_sr must be a positive integer, got {target_sr}."
        )

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
        path: Destination file path. Parent directories are created
            automatically if they do not exist.
        audio: Float32 numpy array of shape ``(samples,)`` (mono) or
            ``(channels, samples)`` (multi-channel, librosa convention).
            The array will be transposed if necessary so soundfile receives
            ``(samples,)`` or ``(samples, channels)``.
        sample_rate: Sample rate in Hz.
        fmt: Output format string, either ``'WAV'`` or ``'FLAC'``. Case
            insensitive.
        subtype: soundfile subtype string (e.g. ``'PCM_16'``, ``'PCM_24'``).
            If ``None``, a sensible default is chosen per format:
            ``'PCM_16'`` for WAV and ``'PCM_24'`` for FLAC.

    Raises:
        ValueError: If *fmt* is not ``'WAV'`` or ``'FLAC'``, if
            *sample_rate* is not positive, or if *audio* has an invalid shape.
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
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError(
            f"sample_rate must be a positive integer, got {sample_rate!r}."
        )

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


def get_audio_info(path: str | Path) -> dict[str, object]:
    """Return basic metadata about an audio file without fully decoding it.

    Uses soundfile for efficient header-only inspection.

    Args:
        path: Path to the audio file.

    Returns:
        A dictionary with keys:

        - ``'sample_rate'`` (int): Sample rate in Hz.
        - ``'channels'`` (int): Number of audio channels.
        - ``'frames'`` (int): Total number of sample frames.
        - ``'duration'`` (float): Duration in seconds.
        - ``'format'`` (str): File format string (e.g. ``'WAV'``).
        - ``'subtype'`` (str): PCM subtype string (e.g. ``'PCM_16'``).

    Raises:
        FileNotFoundError: If *path* does not exist.
        RuntimeError: If soundfile cannot read the file headers.
    """
    import soundfile as sf

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        info = sf.info(str(path))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read audio info from {path}: {exc}"
        ) from exc

    return {
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "frames": info.frames,
        "duration": info.duration,
        "format": info.format,
        "subtype": info.subtype,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _prepare_array_for_write(audio: np.ndarray) -> np.ndarray:
    """Reshape / transpose *audio* so soundfile can write it.

    soundfile expects:
      - 1-D array for mono: shape ``(samples,)``
      - 2-D array for multi-channel: shape ``(samples, channels)``

    librosa returns multi-channel audio as ``(channels, samples)``, so we
    need to transpose when the first dimension is smaller than the second
    (heuristic for channels < samples).

    Args:
        audio: Input numpy array.

    Returns:
        Array ready for ``soundfile.write``, dtype float32.

    Raises:
        ValueError: If *audio* has more than 2 dimensions or is empty.
    """
    if not isinstance(audio, np.ndarray):
        raise ValueError(
            f"audio must be a numpy ndarray, got {type(audio).__name__}."
        )
    if audio.size == 0:
        raise ValueError("audio array must not be empty.")
    if audio.ndim == 1:
        return audio.astype(np.float32, copy=False)
    if audio.ndim == 2:
        rows, cols = audio.shape
        if rows < cols:
            # Assume (channels, samples) librosa layout -> transpose
            return audio.T.astype(np.float32, copy=False)
        # Already (samples, channels) or square — pass through
        return audio.astype(np.float32, copy=False)
    raise ValueError(
        f"audio array must be 1-D or 2-D, got shape {audio.shape}."
    )
