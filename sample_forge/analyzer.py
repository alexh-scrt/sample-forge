"""analyzer: Audio analysis using librosa.

Provides functions to estimate the BPM (tempo) and musical key of a loaded
audio sample represented as a numpy float32 array.  All heavy lifting is
delegated to librosa's beat-tracking and chroma analysis algorithms.

The functions in this module are intentionally stateless and side-effect-free
so they are easy to unit-test and compose.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_bpm(audio: np.ndarray, sample_rate: int) -> float:
    """Estimate the tempo (BPM) of an audio signal.

    Uses librosa's beat tracker (``librosa.beat.beat_track``) which is based
    on dynamic programming over an onset strength envelope.

    Args:
        audio: 1-D float32 numpy array of audio samples (mono).
        sample_rate: Sample rate of *audio* in Hz.

    Returns:
        Estimated tempo in beats per minute as a float.

    Raises:
        ValueError: If *audio* is not a 1-D array or is too short to analyse.
        RuntimeError: If librosa raises an unexpected error.

    Examples::

        >>> import numpy as np
        >>> sr = 22050
        >>> audio = np.random.randn(sr * 5).astype(np.float32)
        >>> bpm = detect_bpm(audio, sr)
        >>> isinstance(bpm, float)
        True
        >>> bpm > 0
        True
    """
    import librosa  # deferred import

    _validate_audio_array(audio, sample_rate, min_duration_seconds=0.5)

    try:
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sample_rate)
    except Exception as exc:
        raise RuntimeError(f"BPM detection failed: {exc}") from exc

    # librosa >=0.10 may return a numpy scalar or 1-element array
    bpm = float(np.atleast_1d(tempo)[0])
    return bpm


def estimate_key(audio: np.ndarray, sample_rate: int) -> Tuple[int, str]:
    """Estimate the musical key of an audio signal.

    Computes chroma energy normalised statistics (CENS) features and builds
    a chroma histogram. The Krumhansl-Schmuckler key-finding algorithm is
    then applied to determine the most likely key and mode.

    The algorithm rotates both major and minor key profiles across all 12
    pitch classes and selects the (pitch_class, mode) pair that yields the
    highest Pearson-like correlation with the observed chroma energy.

    Args:
        audio: 1-D float32 numpy array of audio samples (mono).
        sample_rate: Sample rate of *audio* in Hz.

    Returns:
        A tuple ``(pitch_class, mode)`` where *pitch_class* is an integer in
        ``[0, 11]`` (C=0 … B=11) and *mode* is either ``'major'`` or
        ``'minor'``.

    Raises:
        ValueError: If *audio* is not suitable for analysis.
        RuntimeError: If librosa raises an unexpected error.

    Examples::

        >>> import numpy as np
        >>> sr = 22050
        >>> audio = np.random.randn(sr * 3).astype(np.float32)
        >>> pc, mode = estimate_key(audio, sr)
        >>> 0 <= pc <= 11
        True
        >>> mode in ('major', 'minor')
        True
    """
    import librosa  # deferred import

    _validate_audio_array(audio, sample_rate, min_duration_seconds=0.5)

    try:
        # Compute chroma energy normalised statistics (CENS) — more robust
        # against timbre variation than plain chroma.
        chroma = librosa.feature.chroma_cens(y=audio, sr=sample_rate)
    except Exception as exc:
        raise RuntimeError(f"Chroma feature extraction failed: {exc}") from exc

    # Build a 12-element chroma vector by averaging over time frames.
    chroma_mean = np.mean(chroma, axis=1)  # shape (12,)

    pitch_class, mode = _krumhansl_schmuckler(chroma_mean)
    return pitch_class, mode


def detect_bpm_and_key(
    audio: np.ndarray,
    sample_rate: int,
) -> Tuple[float, int, str]:
    """Convenience wrapper that runs BPM detection and key estimation together.

    Args:
        audio: 1-D float32 numpy array of audio samples (mono).
        sample_rate: Sample rate of *audio* in Hz.

    Returns:
        A tuple ``(bpm, pitch_class, mode)``.

    Raises:
        ValueError: If *audio* is not suitable for analysis.
        RuntimeError: If librosa raises an unexpected error.
    """
    bpm = detect_bpm(audio, sample_rate)
    pitch_class, mode = estimate_key(audio, sample_rate)
    return bpm, pitch_class, mode


# ---------------------------------------------------------------------------
# Krumhansl-Schmuckler key profiles
# ---------------------------------------------------------------------------

# Reference: Krumhansl, C. L. (1990). Cognitive foundations of musical pitch.
# Oxford University Press.
#
# These are the original Krumhansl-Kessler probe-tone ratings for major and
# minor keys, starting on pitch class C (index 0).
_KS_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=np.float64,
)
_KS_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=np.float64,
)


def _normalise_profile(profile: np.ndarray) -> np.ndarray:
    """Return a zero-mean, unit-variance version of *profile*.

    Args:
        profile: A 1-D numpy array.

    Returns:
        Normalised array of the same shape.
    """
    p = profile - profile.mean()
    std = p.std()
    if std == 0.0:
        return p
    return p / std


# Pre-normalise the profiles once at module import time.
_KS_MAJOR_NORM: np.ndarray = _normalise_profile(_KS_MAJOR)
_KS_MINOR_NORM: np.ndarray = _normalise_profile(_KS_MINOR)


def _krumhansl_schmuckler(chroma_vector: np.ndarray) -> Tuple[int, str]:
    """Apply the Krumhansl-Schmuckler algorithm to a chroma vector.

    Rotates the major and minor profiles through all 12 pitch classes and
    picks the (key, mode) pair with the highest dot-product correlation
    against the normalised chroma vector.

    Args:
        chroma_vector: A 12-element array of chroma energies indexed from
            pitch class C (index 0) to B (index 11).

    Returns:
        A tuple ``(pitch_class, mode)`` where *mode* is ``'major'`` or
        ``'minor'``.
    """
    chroma_norm = _normalise_profile(chroma_vector.astype(np.float64))

    best_correlation = -np.inf
    best_pc = 0
    best_mode = "major"

    for pc in range(12):
        # Rotate the profile so that it aligns with pitch class *pc*.
        # np.roll shifts elements to the right, meaning index 0 of the
        # rotated profile corresponds to pitch class *pc* of C-based profile.
        major_rotated = np.roll(_KS_MAJOR_NORM, pc)
        minor_rotated = np.roll(_KS_MINOR_NORM, pc)

        corr_major = float(np.dot(chroma_norm, major_rotated))
        corr_minor = float(np.dot(chroma_norm, minor_rotated))

        if corr_major > best_correlation:
            best_correlation = corr_major
            best_pc = pc
            best_mode = "major"

        if corr_minor > best_correlation:
            best_correlation = corr_minor
            best_pc = pc
            best_mode = "minor"

    return best_pc, best_mode


# ---------------------------------------------------------------------------
# Private validation helpers
# ---------------------------------------------------------------------------


def _validate_audio_array(
    audio: np.ndarray,
    sample_rate: int,
    min_duration_seconds: float = 0.5,
) -> None:
    """Raise ``ValueError`` if *audio* is not suitable for analysis.

    Args:
        audio: The audio array to validate.
        sample_rate: The sample rate in Hz.
        min_duration_seconds: Minimum required duration in seconds.

    Raises:
        ValueError: Describing the specific problem found.
    """
    if not isinstance(audio, np.ndarray):
        raise ValueError(
            f"audio must be a numpy ndarray, got {type(audio).__name__}."
        )
    if audio.ndim != 1:
        raise ValueError(
            f"audio must be a 1-D mono array, got shape {audio.shape}. "
            "Use audio_io.load_audio(..., mono=True) to obtain a mono signal."
        )
    if audio.size == 0:
        raise ValueError("audio array is empty.")
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError(
            f"sample_rate must be a positive integer, got {sample_rate!r}."
        )
    min_samples = int(min_duration_seconds * sample_rate)
    if len(audio) < min_samples:
        raise ValueError(
            f"audio is too short for analysis: {len(audio)} samples "
            f"({len(audio) / sample_rate:.3f} s). Need at least "
            f"{min_duration_seconds} s ({min_samples} samples at {sample_rate} Hz)."
        )
