"""key_utils: Musical key name parsing and semitone offset utilities.

Provides functions to validate user-supplied key strings (e.g. 'C', 'F#m',
'Bbmaj'), look up their semitone positions, and compute the signed semitone
difference required to transpose from one key to another.

This module is intentionally free of audio I/O or heavy dependencies — it
operates purely on string and integer data so it can be unit-tested quickly.
"""

from __future__ import annotations

from typing import Tuple

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

# Mapping of note names (including enharmonic equivalents) to MIDI pitch class
# integers where C = 0, C#/Db = 1, ..., B = 11.
_NOTE_TO_PC: dict[str, int] = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "Fb": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
    "Cb": 11,
}

# Aliases accepted for the mode suffix (stored as lowercase for comparison)
_MAJOR_SUFFIXES: frozenset[str] = frozenset({"", "maj", "major", "m"})
# Note: bare "M" is major; "m" alone is minor — handled in _parse_mode by
# case-sensitive check before lowercasing.
_MINOR_SUFFIXES: frozenset[str] = frozenset({"m", "min", "minor"})

# Relative minor offset: the minor scale root sits 3 semitones below the
# parallel major root. Informational only.
_RELATIVE_MINOR_OFFSET: int = 3


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_key(key_string: str) -> Tuple[int, str]:
    """Parse a key string into a (pitch_class, mode) tuple.

    Accepts a wide variety of notations, for example::

        'C'       -> (0, 'major')
        'F#m'     -> (6, 'minor')
        'Bbmaj'   -> (10, 'major')
        'Abminor' -> (8, 'minor')
        'D#M'     -> (3, 'major')
        'Amin'    -> (9, 'minor')
        'Gmajor'  -> (7, 'major')

    Args:
        key_string: A human-readable musical key name.

    Returns:
        A tuple ``(pitch_class, mode)`` where *pitch_class* is an integer
        in ``[0, 11]`` and *mode* is either ``'major'`` or ``'minor'``.

    Raises:
        ValueError: If *key_string* cannot be parsed as a valid key.
    """
    if not isinstance(key_string, str) or not key_string.strip():
        raise ValueError(
            f"Key string must be a non-empty string, got: {key_string!r}"
        )

    key_string = key_string.strip()

    note_part, suffix_part = _split_note_and_suffix(key_string)

    note_part_normalised = _normalise_note_name(note_part)
    if note_part_normalised not in _NOTE_TO_PC:
        raise ValueError(
            f"Unknown note name {note_part!r} in key string {key_string!r}. "
            f"Valid notes: {sorted(_NOTE_TO_PC.keys())}"
        )

    pitch_class = _NOTE_TO_PC[note_part_normalised]
    mode = _parse_mode(suffix_part, key_string)

    return pitch_class, mode


def key_to_pitch_class(key_string: str) -> int:
    """Return the pitch class integer (0-11) for a key string.

    This is a convenience wrapper around :func:`parse_key` that discards the
    mode information.

    Args:
        key_string: A human-readable musical key name.

    Returns:
        An integer in ``[0, 11]`` representing the root pitch class.

    Raises:
        ValueError: If *key_string* is not a valid key name.
    """
    pitch_class, _ = parse_key(key_string)
    return pitch_class


def semitone_offset(source_key: str, target_key: str) -> int:
    """Compute the signed semitone shift needed to move from *source_key* to *target_key*.

    The function returns the shortest path around the chromatic circle, so the
    result is always in the range ``[-6, 6]``. Specifically it returns a value
    in ``[-6, 6]`` (with +6 preferred over -6 for tritones).

    The mode of the keys is intentionally ignored when computing the offset;
    only the root pitch classes matter for determining how many semitones to
    shift.

    Args:
        source_key: The key of the original audio.
        target_key: The desired key after transposition.

    Returns:
        A signed integer number of semitones (negative = shift down,
        positive = shift up).

    Raises:
        ValueError: If either key string is invalid.

    Examples::

        >>> semitone_offset('C', 'G')    # +7 naive, but shortest = -5
        -5
        >>> semitone_offset('C', 'E')    # +4
        4
        >>> semitone_offset('A', 'Am')   # same root, 0
        0
    """
    src_pc, _ = parse_key(source_key)
    tgt_pc, _ = parse_key(target_key)

    raw_diff = (tgt_pc - src_pc) % 12  # always 0-11
    # Prefer the shorter path: if going up is more than 6 semitones, go down.
    if raw_diff > 6:
        return raw_diff - 12
    return raw_diff


def validate_key(key_string: str) -> bool:
    """Return ``True`` if *key_string* is a valid key name, ``False`` otherwise.

    This function does **not** raise; it is intended for cheap validation in
    CLI option callbacks.

    Args:
        key_string: A string to test.

    Returns:
        ``True`` when :func:`parse_key` would succeed, ``False`` otherwise.
    """
    try:
        parse_key(key_string)
        return True
    except ValueError:
        return False


def pitch_class_to_name(pitch_class: int, prefer_sharps: bool = True) -> str:
    """Convert a pitch class integer back to a canonical note name.

    Args:
        pitch_class: An integer in ``[0, 11]``.
        prefer_sharps: When ``True`` (default) use sharps for accidentals
            (e.g. ``'C#'``); when ``False`` use flats (e.g. ``'Db'``).

    Returns:
        A string note name.

    Raises:
        ValueError: If *pitch_class* is not in ``[0, 11]``.
    """
    if not isinstance(pitch_class, int) or not (0 <= pitch_class <= 11):
        raise ValueError(
            f"pitch_class must be an integer in [0, 11], got: {pitch_class!r}"
        )

    _SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    _FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

    if prefer_sharps:
        return _SHARP_NAMES[pitch_class]
    return _FLAT_NAMES[pitch_class]


def all_key_names(mode: str = "major", prefer_sharps: bool = True) -> list[str]:
    """Return all 12 canonical key names for a given mode.

    Args:
        mode: Either ``'major'`` or ``'minor'``.
        prefer_sharps: If ``True``, use sharps for accidentals; otherwise use
            flats.

    Returns:
        A list of 12 key name strings ordered by pitch class (C=0 first).

    Raises:
        ValueError: If *mode* is not ``'major'`` or ``'minor'``.
    """
    if mode not in ("major", "minor"):
        raise ValueError(f"mode must be 'major' or 'minor', got {mode!r}.")

    suffix = "m" if mode == "minor" else ""
    return [
        pitch_class_to_name(pc, prefer_sharps=prefer_sharps) + suffix
        for pc in range(12)
    ]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _normalise_note_name(note: str) -> str:
    """Capitalise the first letter and lower-case the accidental.

    E.g. ``'f#'`` -> ``'F#'``, ``'BB'`` -> ``'Bb'``, ``'eb'`` -> ``'Eb'``.
    """
    if not note:
        return note
    return note[0].upper() + note[1:].lower()


def _split_note_and_suffix(key_string: str) -> Tuple[str, str]:
    """Split *key_string* into a note name and a mode suffix.

    The note name is the leading letter plus an optional accidental (``'#'``
    or ``'b'``). The remainder is the mode suffix.

    Strategy: try 2-character note names first (e.g. ``'C#'``, ``'Db'``,
    ``'Bb'``); fall back to 1-character.

    Args:
        key_string: The full key string to split.

    Returns:
        A tuple ``(note_part, suffix_part)``.

    Raises:
        ValueError: If the string is empty or does not start with a letter.
    """
    if not key_string:
        raise ValueError("Key string is empty.")

    first_char = key_string[0]
    if not first_char.isalpha():
        raise ValueError(
            f"Key string must start with a letter, got: {key_string!r}"
        )

    # Attempt 2-character note names (e.g. 'C#', 'Db', 'Bb')
    if len(key_string) >= 2 and key_string[1] in ("#", "b"):
        candidate_note = key_string[:2]
        candidate_note_norm = _normalise_note_name(candidate_note)
        if candidate_note_norm in _NOTE_TO_PC:
            return candidate_note, key_string[2:]
        # If not a known 2-char note, fall through to 1-char.

    # 1-character note name
    return key_string[:1], key_string[1:]


def _parse_mode(suffix: str, original: str) -> str:
    """Convert a mode suffix string to ``'major'`` or ``'minor'``.

    The suffix ``'M'`` (uppercase) is treated as major, while ``'m'``
    (lowercase) is treated as minor. All other suffixes are compared
    case-insensitively against the known major/minor alias sets.

    Args:
        suffix: The trailing part of the key string after the note name.
        original: The full original key string (used in error messages only).

    Returns:
        ``'major'`` or ``'minor'``.

    Raises:
        ValueError: If the suffix is not recognised.
    """
    # Empty suffix means major (e.g. plain "C")
    if suffix == "":
        return "major"

    # Exact-case checks for single-letter suffixes: 'M' = major, 'm' = minor
    if suffix == "M":
        return "major"
    if suffix == "m":
        return "minor"

    # Case-insensitive checks for multi-character suffixes
    suffix_lower = suffix.lower()
    if suffix_lower in {"maj", "major"}:
        return "major"
    if suffix_lower in {"min", "minor"}:
        return "minor"

    raise ValueError(
        f"Unrecognised mode suffix {suffix!r} in key string {original!r}. "
        f"Accepted major suffixes: ['', 'M', 'maj', 'major'], "
        f"accepted minor suffixes: ['m', 'min', 'minor']."
    )
