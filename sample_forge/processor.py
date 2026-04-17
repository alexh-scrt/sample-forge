"""processor: Time-stretching and pitch-shifting audio processor.

Applies high-quality time-stretching (to retarget BPM) and pitch-shifting
(to retarget musical key) using the Rubber Band library via the pyrubberband
binding. The module also provides pure-Python helpers for computing the
stretch ratio and semitone offset so they can be tested independently of any
audio I/O.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Ratio / semitone calculation helpers
# ---------------------------------------------------------------------------


def compute_time_stretch_ratio(source_bpm: float, target_bpm: float) -> float:
    """Compute the time-stretch ratio needed to retarget *source_bpm* to *target_bpm*.

    A ratio > 1 means the audio is slowed down (fewer BPM); a ratio < 1 means
    it is sped up (more BPM). This is the value passed directly to
    ``pyrubberband.time_stretch``.

    Args:
        source_bpm: The detected (or assumed) BPM of the source material.
        target_bpm: The desired BPM after processing.

    Returns:
        A positive float stretch ratio.

    Raises:
        ValueError: If either BPM value is not a positive number.

    Examples::

        >>> compute_time_stretch_ratio(120.0, 140.0)  # speed up
        0.857...
        >>> compute_time_stretch_ratio(140.0, 70.0)   # halve speed
        2.0
    """
    _validate_bpm(source_bpm, "source_bpm")
    _validate_bpm(target_bpm, "target_bpm")
    return source_bpm / target_bpm


def compute_pitch_shift_semitones(
    source_key: str,
    target_key: str,
) -> int:
    """Return the number of semitones to shift from *source_key* to *target_key*.

    This is a thin delegation to :func:`sample_forge.key_utils.semitone_offset`
    so that the processor module has a single call-site for pitch shift math.

    Args:
        source_key: The musical key of the source audio (e.g. ``'Am'``).
        target_key:  The desired key after processing (e.g. ``'Cm'``).

    Returns:
        A signed integer number of semitones in the range ``[-6, 6]``.

    Raises:
        ValueError: If either key string is invalid.
    """
    from sample_forge.key_utils import semitone_offset  # local import to avoid cycles

    return semitone_offset(source_key, target_key)


# ---------------------------------------------------------------------------
# Core processing functions
# ---------------------------------------------------------------------------


def time_stretch(
    audio: np.ndarray,
    sample_rate: int,
    ratio: float,
) -> np.ndarray:
    """Time-stretch *audio* by *ratio* using the Rubber Band library.

    Args:
        audio: 1-D float32 numpy array (mono audio).
        sample_rate: Sample rate of *audio* in Hz.
        ratio: Stretch ratio. Values > 1 slow the audio down; < 1 speed it up.
            A ratio of 1.0 returns the input unchanged.

    Returns:
        A new float32 numpy array with the time-stretched audio.

    Raises:
        ValueError: If *audio* is not 1-D, or if *ratio* is not positive.
        RuntimeError: If pyrubberband raises an unexpected error.
    """
    import pyrubberband as pyrb  # deferred import

    _validate_audio_1d(audio)
    if ratio <= 0:
        raise ValueError(f"ratio must be a positive float, got {ratio}.")

    if ratio == 1.0:
        return audio.copy()

    try:
        stretched = pyrb.time_stretch(audio, sample_rate, ratio)
    except Exception as exc:
        raise RuntimeError(f"Time-stretching failed: {exc}") from exc

    return stretched.astype(np.float32)


def pitch_shift(
    audio: np.ndarray,
    sample_rate: int,
    semitones: float,
) -> np.ndarray:
    """Pitch-shift *audio* by *semitones* using the Rubber Band library.

    Args:
        audio: 1-D float32 numpy array (mono audio).
        sample_rate: Sample rate of *audio* in Hz.
        semitones: Number of semitones to shift. Positive values raise the
            pitch; negative values lower it. Fractional values are accepted.

    Returns:
        A new float32 numpy array with the pitch-shifted audio.

    Raises:
        ValueError: If *audio* is not 1-D.
        RuntimeError: If pyrubberband raises an unexpected error.
    """
    import pyrubberband as pyrb  # deferred import

    _validate_audio_1d(audio)

    if semitones == 0.0:
        return audio.copy()

    try:
        shifted = pyrb.pitch_shift(audio, sample_rate, semitones)
    except Exception as exc:
        raise RuntimeError(f"Pitch-shifting failed: {exc}") from exc

    return shifted.astype(np.float32)


def process(
    audio: np.ndarray,
    sample_rate: int,
    source_bpm: float | None = None,
    target_bpm: float | None = None,
    source_key: str | None = None,
    target_key: str | None = None,
) -> np.ndarray:
    """Apply time-stretching and/or pitch-shifting in one call.

    The two transforms are applied in sequence: time-stretch first, then
    pitch-shift. Either transform can be skipped by passing ``None`` for the
    corresponding parameters.

    Args:
        audio: 1-D float32 mono audio array.
        sample_rate: Sample rate in Hz.
        source_bpm: Detected BPM of *audio*. Required if *target_bpm* is set.
        target_bpm: Desired BPM after processing. If ``None``, no time-stretch
            is applied.
        source_key: Detected key of *audio* (e.g. ``'Am'``). Required if
            *target_key* is set.
        target_key: Desired key after processing. If ``None``, no pitch-shift
            is applied.

    Returns:
        A processed float32 numpy array.

    Raises:
        ValueError: If BPM or key parameters are inconsistently specified.
        RuntimeError: For processing failures.
    """
    result = audio

    # --- Time-stretch ---
    if target_bpm is not None:
        if source_bpm is None:
            raise ValueError(
                "source_bpm must be provided when target_bpm is specified."
            )
        ratio = compute_time_stretch_ratio(source_bpm, target_bpm)
        result = time_stretch(result, sample_rate, ratio)

    # --- Pitch-shift ---
    if target_key is not None:
        if source_key is None:
            raise ValueError(
                "source_key must be provided when target_key is specified."
            )
        semitones = compute_pitch_shift_semitones(source_key, target_key)
        result = pitch_shift(result, sample_rate, float(semitones))

    return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _validate_bpm(value: float, name: str) -> None:
    """Raise ``ValueError`` if *value* is not a positive finite float."""
    if not isinstance(value, (int, float)) or np.isnan(value) or np.isinf(value):
        raise ValueError(
            f"{name} must be a finite number, got {value!r}."
        )
    if value <= 0:
        raise ValueError(f"{name} must be a positive number, got {value}.")


def _validate_audio_1d(audio: np.ndarray) -> None:
    """Raise ``ValueError`` if *audio* is not a non-empty 1-D numpy array."""
    if not isinstance(audio, np.ndarray):
        raise ValueError(
            f"audio must be a numpy ndarray, got {type(audio).__name__}."
        )
    if audio.ndim != 1:
        raise ValueError(
            f"audio must be a 1-D array, got shape {audio.shape}."
        )
    if audio.size == 0:
        raise ValueError("audio array must not be empty.")
