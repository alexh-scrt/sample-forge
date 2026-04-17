"""test_key_utils: Unit tests for sample_forge.key_utils.

Covers key-name parsing, semitone offset lookup, pitch_class_to_name,
all_key_names, validate_key, and error handling for invalid inputs.
"""

from __future__ import annotations

import pytest

from sample_forge.key_utils import (
    _normalise_note_name,
    _parse_mode,
    _split_note_and_suffix,
    all_key_names,
    key_to_pitch_class,
    parse_key,
    pitch_class_to_name,
    semitone_offset,
    validate_key,
)


# ---------------------------------------------------------------------------
# parse_key — valid inputs
# ---------------------------------------------------------------------------


class TestParseKeyValid:
    """Tests for parse_key with well-formed input strings."""

    def test_plain_c_major(self) -> None:
        """Plain 'C' should map to pitch class 0, major."""
        pc, mode = parse_key("C")
        assert pc == 0
        assert mode == "major"

    def test_plain_a_major(self) -> None:
        """Plain 'A' should map to pitch class 9, major."""
        pc, mode = parse_key("A")
        assert pc == 9
        assert mode == "major"

    def test_a_minor_lowercase_m(self) -> None:
        """'Am' (lowercase m) should be A minor."""
        pc, mode = parse_key("Am")
        assert pc == 9
        assert mode == "minor"

    def test_a_minor_min_suffix(self) -> None:
        """'Amin' should be A minor."""
        pc, mode = parse_key("Amin")
        assert pc == 9
        assert mode == "minor"

    def test_a_minor_full_suffix(self) -> None:
        """'Aminor' should be A minor."""
        pc, mode = parse_key("Aminor")
        assert pc == 9
        assert mode == "minor"

    def test_c_sharp_major(self) -> None:
        """'C#' should be pitch class 1, major."""
        pc, mode = parse_key("C#")
        assert pc == 1
        assert mode == "major"

    def test_c_sharp_minor(self) -> None:
        """'C#m' should be pitch class 1, minor."""
        pc, mode = parse_key("C#m")
        assert pc == 1
        assert mode == "minor"

    def test_db_major(self) -> None:
        """'Db' is enharmonic to C# — pitch class 1."""
        pc, mode = parse_key("Db")
        assert pc == 1
        assert mode == "major"

    def test_f_sharp_minor(self) -> None:
        """'F#m' should be pitch class 6, minor."""
        pc, mode = parse_key("F#m")
        assert pc == 6
        assert mode == "minor"

    def test_bb_major(self) -> None:
        """'Bb' should be pitch class 10, major."""
        pc, mode = parse_key("Bb")
        assert pc == 10
        assert mode == "major"

    def test_bbmaj_suffix(self) -> None:
        """'Bbmaj' should be pitch class 10, major."""
        pc, mode = parse_key("Bbmaj")
        assert pc == 10
        assert mode == "major"

    def test_bb_minor(self) -> None:
        """'Bbm' should be pitch class 10, minor."""
        pc, mode = parse_key("Bbm")
        assert pc == 10
        assert mode == "minor"

    def test_eb_minor(self) -> None:
        """'Ebminor' should be pitch class 3, minor."""
        pc, mode = parse_key("Ebminor")
        assert pc == 3
        assert mode == "minor"

    def test_ab_major(self) -> None:
        """'Abmaj' should be pitch class 8, major."""
        pc, mode = parse_key("Abmaj")
        assert pc == 8
        assert mode == "major"

    def test_uppercase_M_means_major(self) -> None:
        """Uppercase 'M' suffix should mean major."""
        pc, mode = parse_key("DM")
        assert pc == 2
        assert mode == "major"

    def test_d_sharp_major(self) -> None:
        """'D#' should be pitch class 3, major."""
        pc, mode = parse_key("D#")
        assert pc == 3
        assert mode == "major"

    def test_g_sharp_minor(self) -> None:
        """'G#m' should be pitch class 8, minor."""
        pc, mode = parse_key("G#m")
        assert pc == 8
        assert mode == "minor"

    def test_b_major(self) -> None:
        """'B' should be pitch class 11, major."""
        pc, mode = parse_key("B")
        assert pc == 11
        assert mode == "major"

    def test_cb_major(self) -> None:
        """'Cb' is enharmonic to B — pitch class 11."""
        pc, mode = parse_key("Cb")
        assert pc == 11
        assert mode == "major"

    def test_fb_major(self) -> None:
        """'Fb' is enharmonic to E — pitch class 4."""
        pc, mode = parse_key("Fb")
        assert pc == 4
        assert mode == "major"

    def test_e_major(self) -> None:
        """'Emajor' should be pitch class 4, major."""
        pc, mode = parse_key("Emajor")
        assert pc == 4
        assert mode == "major"

    def test_g_minor_full(self) -> None:
        """'Gminor' should be pitch class 7, minor."""
        pc, mode = parse_key("Gminor")
        assert pc == 7
        assert mode == "minor"

    def test_leading_trailing_whitespace(self) -> None:
        """Leading/trailing whitespace should be stripped."""
        pc, mode = parse_key("  F#m  ")
        assert pc == 6
        assert mode == "minor"

    @pytest.mark.parametrize(
        "key_str, expected_pc, expected_mode",
        [
            ("C", 0, "major"),
            ("C#", 1, "major"),
            ("Db", 1, "major"),
            ("D", 2, "major"),
            ("D#", 3, "major"),
            ("Eb", 3, "major"),
            ("E", 4, "major"),
            ("F", 5, "major"),
            ("F#", 6, "major"),
            ("Gb", 6, "major"),
            ("G", 7, "major"),
            ("G#", 8, "major"),
            ("Ab", 8, "major"),
            ("A", 9, "major"),
            ("A#", 10, "major"),
            ("Bb", 10, "major"),
            ("B", 11, "major"),
        ],
    )
    def test_all_natural_and_accidental_notes(self, key_str, expected_pc, expected_mode) -> None:
        """All 12 pitch classes (with sharps and flats) should parse correctly."""
        pc, mode = parse_key(key_str)
        assert pc == expected_pc
        assert mode == expected_mode

    @pytest.mark.parametrize(
        "key_str, expected_pc",
        [
            ("Cm", 0),
            ("C#m", 1),
            ("Dm", 2),
            ("Ebm", 3),
            ("Em", 4),
            ("Fm", 5),
            ("F#m", 6),
            ("Gm", 7),
            ("Abm", 8),
            ("Am", 9),
            ("Bbm", 10),
            ("Bm", 11),
        ],
    )
    def test_all_minor_keys(self, key_str, expected_pc) -> None:
        """All 12 minor keys should parse as minor."""
        pc, mode = parse_key(key_str)
        assert pc == expected_pc
        assert mode == "minor"


# ---------------------------------------------------------------------------
# parse_key — invalid inputs
# ---------------------------------------------------------------------------


class TestParseKeyInvalid:
    """Tests for parse_key with malformed input strings."""

    def test_empty_string_raises(self) -> None:
        """Empty string should raise ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            parse_key("")

    def test_whitespace_only_raises(self) -> None:
        """Whitespace-only string should raise ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            parse_key("   ")

    def test_non_string_raises(self) -> None:
        """Non-string input should raise ValueError."""
        with pytest.raises(ValueError):
            parse_key(None)  # type: ignore[arg-type]

    def test_integer_raises(self) -> None:
        """Integer input should raise ValueError."""
        with pytest.raises(ValueError):
            parse_key(42)  # type: ignore[arg-type]

    def test_digit_start_raises(self) -> None:
        """String starting with a digit should raise ValueError."""
        with pytest.raises(ValueError):
            parse_key("4m")

    def test_unknown_note_raises(self) -> None:
        """Unknown note name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown note name"):
            parse_key("H")

    def test_unknown_mode_suffix_raises(self) -> None:
        """Unrecognised mode suffix should raise ValueError."""
        with pytest.raises(ValueError, match="Unrecognised mode suffix"):
            parse_key("Cxyz")

    def test_numeric_suffix_raises(self) -> None:
        """Numeric suffix should raise ValueError."""
        with pytest.raises(ValueError):
            parse_key("C7")

    def test_double_sharp_raises(self) -> None:
        """Double sharp (##) is not a valid note accidental."""
        with pytest.raises(ValueError):
            parse_key("C##")


# ---------------------------------------------------------------------------
# validate_key
# ---------------------------------------------------------------------------


class TestValidateKey:
    """Tests for the non-raising validate_key function."""

    @pytest.mark.parametrize(
        "key_str",
        ["C", "Am", "F#m", "Bbmaj", "Ebminor", "DM", "Gb", "Abmin"],
    )
    def test_valid_keys_return_true(self, key_str: str) -> None:
        """Valid key strings should return True."""
        assert validate_key(key_str) is True

    @pytest.mark.parametrize(
        "key_str",
        ["", "  ", "H", "Cxyz", "4m", "123"],
    )
    def test_invalid_keys_return_false(self, key_str: str) -> None:
        """Invalid key strings should return False."""
        assert validate_key(key_str) is False

    def test_none_returns_false(self) -> None:
        """None input should return False without raising."""
        assert validate_key(None) is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# key_to_pitch_class
# ---------------------------------------------------------------------------


class TestKeyToPitchClass:
    """Tests for key_to_pitch_class convenience wrapper."""

    def test_c_major(self) -> None:
        """C should give pitch class 0."""
        assert key_to_pitch_class("C") == 0

    def test_a_minor_same_as_a_major_root(self) -> None:
        """Am and A share root pitch class 9."""
        assert key_to_pitch_class("Am") == key_to_pitch_class("A")

    def test_f_sharp(self) -> None:
        """F# should give pitch class 6."""
        assert key_to_pitch_class("F#") == 6

    def test_bb(self) -> None:
        """Bb should give pitch class 10."""
        assert key_to_pitch_class("Bb") == 10

    def test_invalid_raises(self) -> None:
        """Invalid key string should raise ValueError."""
        with pytest.raises(ValueError):
            key_to_pitch_class("Z")


# ---------------------------------------------------------------------------
# semitone_offset
# ---------------------------------------------------------------------------


class TestSemitoneOffset:
    """Tests for the semitone_offset function."""

    def test_same_key_returns_zero(self) -> None:
        """Identical source and target keys should return 0."""
        assert semitone_offset("C", "C") == 0

    def test_same_root_different_mode_returns_zero(self) -> None:
        """Mode is ignored; same root = 0 semitones."""
        assert semitone_offset("A", "Am") == 0
        assert semitone_offset("Am", "A") == 0

    def test_c_to_e_is_plus_four(self) -> None:
        """C to E is 4 semitones up."""
        assert semitone_offset("C", "E") == 4

    def test_c_to_g_is_minus_five(self) -> None:
        """C to G: up 7 or down 5; shortest is -5."""
        assert semitone_offset("C", "G") == -5

    def test_c_to_f_is_plus_five(self) -> None:
        """C to F: up 5 or down 7; shortest is +5."""
        assert semitone_offset("C", "F") == 5

    def test_c_to_gb_is_plus_six(self) -> None:
        """Tritone (6 semitones) should be +6 (not -6)."""
        assert semitone_offset("C", "Gb") == 6

    def test_g_to_c_is_plus_five(self) -> None:
        """G to C: up 5 semitones."""
        assert semitone_offset("G", "C") == 5

    def test_a_to_c_is_plus_three(self) -> None:
        """A to C: up 3 semitones."""
        assert semitone_offset("A", "C") == 3

    def test_e_to_c_is_minus_four(self) -> None:
        """E to C: down 4 semitones."""
        assert semitone_offset("E", "C") == -4

    def test_bb_to_f_sharp_is_minus_four(self) -> None:
        """Bb to F#: Bb=10, F#=6, (6-10)%12 = 8 -> 8-12 = -4."""
        assert semitone_offset("Bb", "F#") == -4

    def test_f_sharp_to_bb_is_plus_four(self) -> None:
        """F# to Bb: (10-6)%12 = 4."""
        assert semitone_offset("F#", "Bb") == 4

    def test_invalid_source_raises(self) -> None:
        """Invalid source key should raise ValueError."""
        with pytest.raises(ValueError):
            semitone_offset("Z", "C")

    def test_invalid_target_raises(self) -> None:
        """Invalid target key should raise ValueError."""
        with pytest.raises(ValueError):
            semitone_offset("C", "Z")

    @pytest.mark.parametrize(
        "src, tgt, expected",
        [
            ("C", "C", 0),
            ("C", "C#", 1),
            ("C", "D", 2),
            ("C", "D#", 3),
            ("C", "E", 4),
            ("C", "F", 5),
            ("C", "F#", 6),
            ("C", "G", -5),
            ("C", "G#", -4),
            ("C", "A", -3),
            ("C", "A#", -2),
            ("C", "B", -1),
        ],
    )
    def test_all_offsets_from_c(self, src, tgt, expected) -> None:
        """Verify all 12 offsets from C."""
        assert semitone_offset(src, tgt) == expected

    def test_result_always_in_range(self) -> None:
        """semitone_offset must always return a value in [-6, 6]."""
        all_notes = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
        for src in all_notes:
            for tgt in all_notes:
                offset = semitone_offset(src, tgt)
                assert -6 <= offset <= 6, (
                    f"Offset from {src} to {tgt} = {offset} is out of range [-6, 6]"
                )


# ---------------------------------------------------------------------------
# pitch_class_to_name
# ---------------------------------------------------------------------------


class TestPitchClassToName:
    """Tests for pitch_class_to_name."""

    def test_c_is_zero(self) -> None:
        """Pitch class 0 should be 'C'."""
        assert pitch_class_to_name(0) == "C"

    def test_b_is_eleven(self) -> None:
        """Pitch class 11 should be 'B'."""
        assert pitch_class_to_name(11) == "B"

    def test_prefer_sharps_default(self) -> None:
        """Default should prefer sharps."""
        assert pitch_class_to_name(1) == "C#"
        assert pitch_class_to_name(6) == "F#"

    def test_prefer_flats(self) -> None:
        """When prefer_sharps=False, accidentals should be flat."""
        assert pitch_class_to_name(1, prefer_sharps=False) == "Db"
        assert pitch_class_to_name(10, prefer_sharps=False) == "Bb"

    def test_round_trip_sharps(self) -> None:
        """key_to_pitch_class(pitch_class_to_name(pc)) should recover pc."""
        for pc in range(12):
            name = pitch_class_to_name(pc, prefer_sharps=True)
            recovered = key_to_pitch_class(name)
            assert recovered == pc

    def test_round_trip_flats(self) -> None:
        """Round-trip with flat names should also work."""
        for pc in range(12):
            name = pitch_class_to_name(pc, prefer_sharps=False)
            recovered = key_to_pitch_class(name)
            assert recovered == pc

    def test_negative_pitch_class_raises(self) -> None:
        """Negative pitch class should raise ValueError."""
        with pytest.raises(ValueError):
            pitch_class_to_name(-1)

    def test_pitch_class_12_raises(self) -> None:
        """Pitch class 12 is out of range."""
        with pytest.raises(ValueError):
            pitch_class_to_name(12)

    def test_float_raises(self) -> None:
        """Float pitch class should raise ValueError."""
        with pytest.raises(ValueError):
            pitch_class_to_name(1.5)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# all_key_names
# ---------------------------------------------------------------------------


class TestAllKeyNames:
    """Tests for all_key_names helper."""

    def test_returns_12_major_keys(self) -> None:
        """Should return exactly 12 major key names."""
        keys = all_key_names(mode="major")
        assert len(keys) == 12

    def test_returns_12_minor_keys(self) -> None:
        """Should return exactly 12 minor key names."""
        keys = all_key_names(mode="minor")
        assert len(keys) == 12

    def test_major_keys_start_with_c(self) -> None:
        """First major key should be 'C'."""
        keys = all_key_names(mode="major")
        assert keys[0] == "C"

    def test_minor_keys_start_with_cm(self) -> None:
        """First minor key should be 'Cm'."""
        keys = all_key_names(mode="minor")
        assert keys[0] == "Cm"

    def test_all_major_keys_valid(self) -> None:
        """Every returned major key name should parse successfully."""
        for name in all_key_names(mode="major"):
            assert validate_key(name), f"{name!r} should be a valid key"

    def test_all_minor_keys_valid(self) -> None:
        """Every returned minor key name should parse successfully."""
        for name in all_key_names(mode="minor"):
            assert validate_key(name), f"{name!r} should be a valid key"

    def test_flat_major_keys(self) -> None:
        """Flat naming should produce 'Bb' instead of 'A#'."""
        keys = all_key_names(mode="major", prefer_sharps=False)
        assert "Bb" in keys
        assert "A#" not in keys

    def test_invalid_mode_raises(self) -> None:
        """Unsupported mode should raise ValueError."""
        with pytest.raises(ValueError, match="mode"):
            all_key_names(mode="chromatic")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Private helper tests
# ---------------------------------------------------------------------------


class TestNormaliseNoteName:
    """Tests for _normalise_note_name."""

    def test_lowercase_note(self) -> None:
        """'c' should become 'C'."""
        assert _normalise_note_name("c") == "C"

    def test_lowercase_accidental(self) -> None:
        """'F#' should remain 'F#'."""
        assert _normalise_note_name("f#") == "F#"

    def test_double_upper_flat(self) -> None:
        """'BB' should become 'Bb'."""
        assert _normalise_note_name("BB") == "Bb"

    def test_empty_string(self) -> None:
        """Empty string should return empty string."""
        assert _normalise_note_name("") == ""


class TestSplitNoteAndSuffix:
    """Tests for _split_note_and_suffix."""

    def test_plain_c(self) -> None:
        """'C' -> note='C', suffix=''."""
        note, suffix = _split_note_and_suffix("C")
        assert note == "C"
        assert suffix == ""

    def test_c_sharp_minor(self) -> None:
        """'C#m' -> note='C#', suffix='m'."""
        note, suffix = _split_note_and_suffix("C#m")
        assert note == "C#"
        assert suffix == "m"

    def test_bb_major(self) -> None:
        """'Bbmaj' -> note='Bb', suffix='maj'."""
        note, suffix = _split_note_and_suffix("Bbmaj")
        assert note == "Bb"
        assert suffix == "maj"

    def test_b_minor(self) -> None:
        """'Bm' -> note='B', suffix='m' (not 'Bm' as a flat note)."""
        note, suffix = _split_note_and_suffix("Bm")
        # 'Bm' is NOT a known 2-char note, so it splits as B + m
        assert note == "B"
        assert suffix == "m"

    def test_empty_string_raises(self) -> None:
        """Empty string should raise ValueError."""
        with pytest.raises(ValueError):
            _split_note_and_suffix("")

    def test_digit_start_raises(self) -> None:
        """String starting with digit should raise ValueError."""
        with pytest.raises(ValueError):
            _split_note_and_suffix("4G")


class TestParseMode:
    """Tests for _parse_mode."""

    def test_empty_suffix_is_major(self) -> None:
        """Empty suffix should return 'major'."""
        assert _parse_mode("", "C") == "major"

    def test_uppercase_M_is_major(self) -> None:
        """'M' should return 'major'."""
        assert _parse_mode("M", "CM") == "major"

    def test_lowercase_m_is_minor(self) -> None:
        """'m' should return 'minor'."""
        assert _parse_mode("m", "Am") == "minor"

    def test_maj_is_major(self) -> None:
        """'maj' should return 'major'."""
        assert _parse_mode("maj", "Cmaj") == "major"

    def test_major_is_major(self) -> None:
        """'major' should return 'major'."""
        assert _parse_mode("major", "Cmajor") == "major"

    def test_min_is_minor(self) -> None:
        """'min' should return 'minor'."""
        assert _parse_mode("min", "Amin") == "minor"

    def test_minor_is_minor(self) -> None:
        """'minor' should return 'minor'."""
        assert _parse_mode("minor", "Aminor") == "minor"

    def test_case_insensitive_major(self) -> None:
        """'MAJ' should be recognised as major."""
        assert _parse_mode("MAJ", "CMAJ") == "major"

    def test_case_insensitive_minor(self) -> None:
        """'MIN' should be recognised as minor."""
        assert _parse_mode("MIN", "AMIN") == "minor"

    def test_invalid_suffix_raises(self) -> None:
        """Unrecognised suffix should raise ValueError."""
        with pytest.raises(ValueError, match="Unrecognised mode suffix"):
            _parse_mode("xyz", "Cxyz")
