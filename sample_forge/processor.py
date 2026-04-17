"""processor: Time-stretching and pitch-shifting audio processor.

Applies high-quality time-stretching (to retarget BPM) and pitch-shifting
(to retarget musical key) using the Rubber Band library via the pyrubberband
binding. The module also provides pure-Python helpers for computing the
stretch ratio and semitone offset so they can be tested independently of any
audio I/O.

Typical usage::

    from sample_forge.processor import process

    processed = process(
        audio, sample_rate,
        source_bpm=120.0, target_bpm=140.0,
        source_key='Am', target_key='Cm',
    )
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Ratio / semitone calculation helpers
# ---------------------------------------------------------------------------


def compute_time_stretch_ratio(source_bpm: float, target_bpm: float) -> float:
    """Compute the time-stretch ratio needed to retarget *source_bpm* to *target_bpm*.

    Rubber Band's ``time_stretch`` function expects a ratio where values > 1
    slow the audio down (stretch it longer) and values < 1 speed it up
    (compress it shorter).  For BPM retargeting the relationship is simply::

        ratio = source_bpm / target_bpm

    Because a higher target BPM means fewer samples per beat, i.e. the audio
    is compressed in time.

    Args:
        source_bpm: The detected (or assumed) BPM of the source material.
            Must be a positive finite number.
        target_bpm: The desired BPM after processing. Must be a positive
            finite number.

    Returns:
        A positive float stretch ratio suitable for passing directly to
        :func:`time_stretch`.

    Raises:
        ValueError: If either BPM value is not a positive finite number.

    Examples::

        >>> compute_time_stretch_ratio(120.0, 140.0)  # speed up
        0.8571428571428571
        >>> compute_time_stretch_ratio(140.0, 70.0)   # halve speed
        2.0
        >>> compute_time_stretch_ratio(100.0, 100.0)  # no change
        1.0
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
    The result is the shortest path around the chromatic circle and is always
    in the range ``[-6, 6]``.

    Args:
        source_key: The musical key of the source audio (e.g. ``'Am'``).
        target_key: The desired key after processing (e.g. ``'Cm'``).

    Returns:
        A signed integer number of semitones in the range ``[-6, 6]``.
        Positive values shift the pitch up; negative values shift it down.

    Raises:
        ValueError: If either key string is invalid.

    Examples::

        >>> compute_pitch_shift_semitones('C', 'E')   # +4
        4
        >>> compute_pitch_shift_semitones('C', 'G')   # -5 (shortest path)
        -5
        >>> compute_pitch_shift_semitones('Am', 'Am') # 0
        0
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

    Rubber Band preserves the pitch (timbre) of the signal while altering
    its duration.  It uses a phase-vocoder-based approach with transient
    detection for natural-sounding results on percussive material.

    Args:
        audio: 1-D float32 numpy array (mono audio samples).
        sample_rate: Sample rate of *audio* in Hz.  Must be a positive
            integer.
        ratio: Time-stretch ratio.

            - ``ratio > 1``: audio is stretched (slowed down, output is longer)
            - ``ratio < 1``: audio is compressed (sped up, output is shorter)
            - ``ratio == 1.0``: no change; returns a copy of the input.

            Must be a positive finite float.

    Returns:
        A new float32 numpy array containing the time-stretched audio.
        Length will be approximately ``len(audio) * ratio``.

    Raises:
        ValueError: If *audio* is not a non-empty 1-D numpy array, if
            *sample_rate* is not a positive integer, or if *ratio* is not
            a positive finite number.
        RuntimeError: If pyrubberband raises an unexpected error during
            processing.
    """
    import pyrubberband as pyrb  # deferred import

    _validate_audio_1d(audio)
    _validate_sample_rate(sample_rate)
    _validate_ratio(ratio)

    if ratio == 1.0:
        return audio.copy()

    try:
        stretched = pyrb.time_stretch(audio, sample_rate, ratio)
    except Exception as exc:
        raise RuntimeError(
            f"Time-stretching failed (ratio={ratio}, sr={sample_rate}): {exc}"
        ) from exc

    return stretched.astype(np.float32)


def pitch_shift(
    audio: np.ndarray,
    sample_rate: int,
    semitones: float,
) -> np.ndarray:
    """Pitch-shift *audio* by *semitones* using the Rubber Band library.

    Rubber Band preserves the timing (duration) of the signal while altering
    its pitch.  Fractional semitone values are accepted for fine-tuning.

    Args:
        audio: 1-D float32 numpy array (mono audio samples).
        sample_rate: Sample rate of *audio* in Hz.  Must be a positive
            integer.
        semitones: Number of semitones to shift.

            - Positive values raise the pitch.
            - Negative values lower the pitch.
            - ``0.0``: no change; returns a copy of the input.

            Fractional values are accepted (e.g. ``0.5`` for a quarter-tone).

    Returns:
        A new float32 numpy array containing the pitch-shifted audio.
        Length will be the same as the input.

    Raises:
        ValueError: If *audio* is not a non-empty 1-D numpy array, or if
            *sample_rate* is not a positive integer.
        RuntimeError: If pyrubberband raises an unexpected error during
            processing.
    """
    import pyrubberband as pyrb  # deferred import

    _validate_audio_1d(audio)
    _validate_sample_rate(sample_rate)

    if semitones == 0.0:
        return audio.copy()

    try:
        shifted = pyrb.pitch_shift(audio, sample_rate, semitones)
    except Exception as exc:
        raise RuntimeError(
            f"Pitch-shifting failed (semitones={semitones}, sr={sample_rate}): {exc}"
        ) from exc

    return shifted.astype(np.float32)


def process(
    audio: np.ndarray,
    sample_rate: int,
    source_bpm: float | None = None,
    target_bpm: float | None = None,
    source_key: str | None = None,
    target_key: str | None = None,
) -> np.ndarray:
    """Apply time-stretching and/or pitch-shifting in one convenient call.

    The two transforms are applied in sequence:

    1. **Time-stretch** (if *target_bpm* is provided): stretches or compresses
       the audio so that its tempo matches *target_bpm*.
    2. **Pitch-shift** (if *target_key* is provided): shifts the pitch of the
       (possibly time-stretched) audio so that its root note matches *target_key*.

    Either transform can be skipped by passing ``None`` for the corresponding
    parameters.  If both transforms are skipped, the input is returned unchanged
    (as a copy).

    Args:
        audio: 1-D float32 mono audio array.
        sample_rate: Sample rate of *audio* in Hz.
        source_bpm: Detected or known BPM of *audio*.  **Required** when
            *target_bpm* is not ``None``.
        target_bpm: Desired BPM after processing.  If ``None``, no
            time-stretching is applied.
        source_key: Detected or known key of *audio* (e.g. ``'Am'``).
            **Required** when *target_key* is not ``None``.
        target_key: Desired key after processing (e.g. ``'Cm'``).  If
            ``None``, no pitch-shifting is applied.

    Returns:
        A processed float32 numpy array.  If no transforms are requested,
        this is a copy of *audio*.

    Raises:
        ValueError: If *target_bpm* is provided without *source_bpm*, if
            *target_key* is provided without *source_key*, if BPM values are
            invalid, or if key strings cannot be parsed.
        RuntimeError: For processing failures in the Rubber Band library.

    Examples::

        # Time-stretch only
        out = process(audio, sr, source_bpm=120.0, target_bpm=140.0)

        # Pitch-shift only
        out = process(audio, sr, source_key='Am', target_key='Cm')

        # Both transforms
        out = process(audio, sr,
                      source_bpm=120.0, target_bpm=140.0,
                      source_key='C', target_key='G')
    """
    _validate_audio_1d(audio)
    _validate_sample_rate(sample_rate)

    result = audio

    # ------------------------------------------------------------------
    # 1. Time-stretch
    # ------------------------------------------------------------------
    if target_bpm is not None:
        if source_bpm is None:
            raise ValueError(
                "source_bpm must be provided when target_bpm is specified."
            )
        ratio = compute_time_stretch_ratio(source_bpm, target_bpm)
        result = time_stretch(result, sample_rate, ratio)

    # ------------------------------------------------------------------
    # 2. Pitch-shift
    # ------------------------------------------------------------------
    if target_key is not None:
        if source_key is None:
            raise ValueError(
                "source_key must be provided when target_key is specified."
            )
        semitones = compute_pitch_shift_semitones(source_key, target_key)
        result = pitch_shift(result, sample_rate, float(semitones))

    # If no transforms were requested, return a copy (not the original reference)
    if target_bpm is None and target_key is None:
        return audio.copy()

    return result


# ---------------------------------------------------------------------------
# Private validation helpers
# ---------------------------------------------------------------------------


def _validate_bpm(value: float, name: str) -> None:
    """Raise ``ValueError`` if *value* is not a positive finite float.

    Args:
        value: The value to validate.
        name: The parameter name for use in the error message.

    Raises:
        ValueError: With a descriptive message if *value* is invalid.
    """
    if not isinstance(value, (int, float)):
        raise ValueError(
            f"{name} must be a positive finite number, got {value!r} "
            f"(type {type(value).__name__})."
        )
    if isinstance(value, bool):  # bool is a subclass of int — reject it
        raise ValueError(
            f"{name} must be a numeric type, not bool."
        )
    fval = float(value)
    if np.isnan(fval) or np.isinf(fval):
        raise ValueError(
            f"{name} must be a finite number, got {value!r}."
        )
    if fval <= 0:
        raise ValueError(
            f"{name} must be a positive number, got {value}."
        )


def _validate_ratio(ratio: float) -> None:
    """Raise ``ValueError`` if *ratio* is not a positive finite float.

    Args:
        ratio: The time-stretch ratio to validate.

    Raises:
        ValueError: With a descriptive message if *ratio* is invalid.
    """
    if not isinstance(ratio, (int, float)) or isinstance(ratio, bool):
        raise ValueError(
            f"ratio must be a positive finite float, got {ratio!r} "
            f"(type {type(ratio).__name__})."
        )
    fval = float(ratio)
    if np.isnan(fval) or np.isinf(fval):
        raise ValueError(f"ratio must be a finite number, got {ratio!r}.")
    if fval <= 0:
        raise ValueError(f"ratio must be a positive float, got {ratio}.")


def _validate_audio_1d(audio: np.ndarray) -> None:
    """Raise ``ValueError`` if *audio* is not a non-empty 1-D numpy array.

    Args:
        audio: The audio array to validate.

    Raises:
        ValueError: With a descriptive message indicating the specific problem.
    """
    if not isinstance(audio, np.ndarray):
        raise ValueError(
            f"audio must be a numpy ndarray, got {type(audio).__name__}."
        )
    if audio.ndim != 1:
        raise ValueError(
            f"audio must be a 1-D array, got shape {audio.shape}. "
            "Use audio_io.load_audio(..., mono=True) to obtain a mono signal."
        )
    if audio.size == 0:
        raise ValueError("audio array must not be empty.")


def _validate_sample_rate(sample_rate: int) -> None:
    """Raise ``ValueError`` if *sample_rate* is not a positive integer.

    Args:
        sample_rate: The sample rate to validate.

    Raises:
        ValueError: With a descriptive message if *sample_rate* is invalid.
    """
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool):
        raise ValueError(
            f"sample_rate must be a positive integer, got {sample_rate!r} "
            f"(type {type(sample_rate).__name__})."
        )
    if sample_rate <= 0:
        raise ValueError(
            f"sample_rate must be a positive integer, got {sample_rate}."
        )
