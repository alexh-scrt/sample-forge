"""test_processor: Unit tests for sample_forge.processor.

Covers:
- compute_time_stretch_ratio: correct math, edge cases, error paths.
- compute_pitch_shift_semitones: delegation to key_utils, error paths.
- time_stretch: correct output shape/dtype, no-op ratio, error paths.
  Uses a small synthetic sine wave; actual Rubber Band processing is tested
  structurally (duration approximately matches ratio * input length).
- pitch_shift: correct output length/dtype, no-op semitones, error paths.
- process: all combinations of BPM/key transforms, missing param errors.
- _validate_bpm: all invalid input types.
- _validate_audio_1d: shape and type checks.
- _validate_sample_rate: type and range checks.
- _validate_ratio: invalid ratio values.
"""

from __future__ import annotations

import numpy as np
import pytest

from sample_forge.processor import (
    _validate_audio_1d,
    _validate_bpm,
    _validate_ratio,
    _validate_sample_rate,
    compute_pitch_shift_semitones,
    compute_time_stretch_ratio,
    pitch_shift,
    process,
    time_stretch,
)


# ---------------------------------------------------------------------------
# Synthetic signal helpers
# ---------------------------------------------------------------------------


def _make_sine(
    frequency: float = 440.0,
    duration: float = 1.0,
    sr: int = 22050,
) -> np.ndarray:
    """Return a mono sine wave as a float32 array."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2.0 * np.pi * frequency * t).astype(np.float32)


def _make_noise(
    duration: float = 1.0,
    sr: int = 22050,
    seed: int = 0,
) -> np.ndarray:
    """Return white noise as a float32 array."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(int(sr * duration)).astype(np.float32) * 0.1


# ---------------------------------------------------------------------------
# compute_time_stretch_ratio
# ---------------------------------------------------------------------------


class TestComputeTimeStretchRatio:
    """Tests for compute_time_stretch_ratio."""

    def test_equal_bpm_returns_one(self) -> None:
        """Same source and target BPM should give ratio 1.0."""
        assert compute_time_stretch_ratio(120.0, 120.0) == pytest.approx(1.0)

    def test_double_target_bpm_halves_ratio(self) -> None:
        """Doubling the target BPM should halve the ratio (speed up 2x)."""
        assert compute_time_stretch_ratio(100.0, 200.0) == pytest.approx(0.5)

    def test_half_target_bpm_doubles_ratio(self) -> None:
        """Halving the target BPM should double the ratio (slow down 2x)."""
        assert compute_time_stretch_ratio(140.0, 70.0) == pytest.approx(2.0)

    def test_120_to_140(self) -> None:
        """120 -> 140 BPM should give ratio 120/140."""
        assert compute_time_stretch_ratio(120.0, 140.0) == pytest.approx(120.0 / 140.0)

    def test_90_to_120(self) -> None:
        """90 -> 120 BPM should give ratio 90/120 = 0.75."""
        assert compute_time_stretch_ratio(90.0, 120.0) == pytest.approx(0.75)

    def test_accepts_integer_bpm(self) -> None:
        """Integer BPM values should be accepted and produce a float ratio."""
        ratio = compute_time_stretch_ratio(120, 140)
        assert isinstance(ratio, float)
        assert ratio == pytest.approx(120.0 / 140.0)

    def test_ratio_is_always_positive(self) -> None:
        """Ratio must always be positive for any positive BPM pair."""
        for src in [60.0, 90.0, 120.0, 140.0, 174.0]:
            for tgt in [60.0, 90.0, 120.0, 140.0, 174.0]:
                ratio = compute_time_stretch_ratio(src, tgt)
                assert ratio > 0.0, f"Negative ratio for {src} -> {tgt}: {ratio}"

    def test_ratio_formula(self) -> None:
        """Ratio must equal source_bpm / target_bpm."""
        src, tgt = 173.0, 88.0
        expected = src / tgt
        assert compute_time_stretch_ratio(src, tgt) == pytest.approx(expected)

    # --- Error paths ---

    def test_zero_source_bpm_raises(self) -> None:
        """source_bpm=0 should raise ValueError."""
        with pytest.raises(ValueError, match="source_bpm"):
            compute_time_stretch_ratio(0.0, 120.0)

    def test_negative_source_bpm_raises(self) -> None:
        """Negative source_bpm should raise ValueError."""
        with pytest.raises(ValueError, match="source_bpm"):
            compute_time_stretch_ratio(-120.0, 120.0)

    def test_zero_target_bpm_raises(self) -> None:
        """target_bpm=0 should raise ValueError."""
        with pytest.raises(ValueError, match="target_bpm"):
            compute_time_stretch_ratio(120.0, 0.0)

    def test_negative_target_bpm_raises(self) -> None:
        """Negative target_bpm should raise ValueError."""
        with pytest.raises(ValueError, match="target_bpm"):
            compute_time_stretch_ratio(120.0, -140.0)

    def test_nan_source_bpm_raises(self) -> None:
        """NaN source_bpm should raise ValueError."""
        with pytest.raises(ValueError, match="source_bpm"):
            compute_time_stretch_ratio(float("nan"), 120.0)

    def test_inf_target_bpm_raises(self) -> None:
        """Infinite target_bpm should raise ValueError."""
        with pytest.raises(ValueError, match="target_bpm"):
            compute_time_stretch_ratio(120.0, float("inf"))

    def test_string_source_bpm_raises(self) -> None:
        """String source_bpm should raise ValueError."""
        with pytest.raises(ValueError):
            compute_time_stretch_ratio("120", 140.0)  # type: ignore[arg-type]

    def test_none_target_bpm_raises(self) -> None:
        """None target_bpm should raise ValueError."""
        with pytest.raises(ValueError):
            compute_time_stretch_ratio(120.0, None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# compute_pitch_shift_semitones
# ---------------------------------------------------------------------------


class TestComputePitchShiftSemitones:
    """Tests for compute_pitch_shift_semitones."""

    def test_same_key_returns_zero(self) -> None:
        """Same source and target key should give 0 semitones."""
        assert compute_pitch_shift_semitones("C", "C") == 0

    def test_same_root_different_mode_returns_zero(self) -> None:
        """Am vs A should give 0 (mode is ignored for pitch class comparison)."""
        assert compute_pitch_shift_semitones("A", "Am") == 0
        assert compute_pitch_shift_semitones("Am", "A") == 0

    def test_c_to_e_is_plus_four(self) -> None:
        """C to E is +4 semitones."""
        assert compute_pitch_shift_semitones("C", "E") == 4

    def test_c_to_g_is_minus_five(self) -> None:
        """C to G: shortest path is -5 (down a fifth)."""
        assert compute_pitch_shift_semitones("C", "G") == -5

    def test_a_minor_to_c_minor(self) -> None:
        """Am to Cm: A=9, C=0, (0-9)%12=3 -> +3."""
        assert compute_pitch_shift_semitones("Am", "Cm") == 3

    def test_f_sharp_to_bb(self) -> None:
        """F# to Bb: (10-6)%12=4 -> +4."""
        assert compute_pitch_shift_semitones("F#", "Bb") == 4

    def test_result_type_is_int(self) -> None:
        """Return type must be int."""
        result = compute_pitch_shift_semitones("C", "E")
        assert isinstance(result, int)

    def test_result_in_range(self) -> None:
        """Result must always be in [-6, 6]."""
        all_notes = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
        for src in all_notes:
            for tgt in all_notes:
                result = compute_pitch_shift_semitones(src, tgt)
                assert -6 <= result <= 6, (
                    f"{src} -> {tgt}: expected in [-6,6], got {result}"
                )

    @pytest.mark.parametrize(
        "src, tgt, expected",
        [
            ("C", "C", 0),
            ("C", "C#", 1),
            ("C", "D", 2),
            ("C", "Eb", 3),
            ("C", "E", 4),
            ("C", "F", 5),
            ("C", "F#", 6),
            ("C", "G", -5),
            ("C", "Ab", -4),
            ("C", "A", -3),
            ("C", "Bb", -2),
            ("C", "B", -1),
        ],
    )
    def test_all_offsets_from_c(self, src: str, tgt: str, expected: int) -> None:
        """Verify all 12 semitone offsets from C."""
        assert compute_pitch_shift_semitones(src, tgt) == expected

    # --- Error paths ---

    def test_invalid_source_key_raises(self) -> None:
        """Invalid source key should raise ValueError."""
        with pytest.raises(ValueError):
            compute_pitch_shift_semitones("Z", "C")

    def test_invalid_target_key_raises(self) -> None:
        """Invalid target key should raise ValueError."""
        with pytest.raises(ValueError):
            compute_pitch_shift_semitones("C", "X")

    def test_empty_source_key_raises(self) -> None:
        """Empty source key should raise ValueError."""
        with pytest.raises(ValueError):
            compute_pitch_shift_semitones("", "C")

    def test_empty_target_key_raises(self) -> None:
        """Empty target key should raise ValueError."""
        with pytest.raises(ValueError):
            compute_pitch_shift_semitones("C", "")


# ---------------------------------------------------------------------------
# time_stretch
# ---------------------------------------------------------------------------


class TestTimeStretch:
    """Tests for time_stretch."""

    SR = 22050

    def test_returns_ndarray(self) -> None:
        """Output must be a numpy ndarray."""
        audio = _make_sine(duration=1.0, sr=self.SR)
        result = time_stretch(audio, self.SR, ratio=1.0)
        assert isinstance(result, np.ndarray)

    def test_output_dtype_float32(self) -> None:
        """Output dtype must be float32."""
        audio = _make_sine(duration=1.0, sr=self.SR)
        result = time_stretch(audio, self.SR, ratio=1.0)
        assert result.dtype == np.float32

    def test_output_is_1d(self) -> None:
        """Output must be 1-D."""
        audio = _make_sine(duration=1.0, sr=self.SR)
        result = time_stretch(audio, self.SR, ratio=1.0)
        assert result.ndim == 1

    def test_ratio_one_returns_copy_equal_to_input(self) -> None:
        """ratio=1.0 should return a copy with the same values."""
        audio = _make_sine(duration=1.0, sr=self.SR)
        result = time_stretch(audio, self.SR, ratio=1.0)
        np.testing.assert_array_equal(result, audio)
        # Must be a copy, not the same object
        assert result is not audio

    def test_ratio_two_doubles_length(self) -> None:
        """ratio=2.0 should approximately double the output length."""
        audio = _make_sine(duration=1.0, sr=self.SR)
        result = time_stretch(audio, self.SR, ratio=2.0)
        expected_len = len(audio) * 2
        # Allow 5% tolerance for Rubber Band's frame-based processing
        assert abs(len(result) - expected_len) < expected_len * 0.05, (
            f"Expected ~{expected_len} samples, got {len(result)}"
        )

    def test_ratio_half_halves_length(self) -> None:
        """ratio=0.5 should approximately halve the output length."""
        audio = _make_sine(duration=2.0, sr=self.SR)
        result = time_stretch(audio, self.SR, ratio=0.5)
        expected_len = len(audio) // 2
        assert abs(len(result) - expected_len) < expected_len * 0.05, (
            f"Expected ~{expected_len} samples, got {len(result)}"
        )

    def test_output_length_proportional_to_ratio(self) -> None:
        """Output length should be approximately ratio * input length."""
        audio = _make_noise(duration=2.0, sr=self.SR)
        for ratio in [0.75, 1.25, 1.5]:
            result = time_stretch(audio, self.SR, ratio=ratio)
            expected = len(audio) * ratio
            # 5% tolerance
            assert abs(len(result) - expected) < expected * 0.05, (
                f"ratio={ratio}: expected ~{expected:.0f} samples, got {len(result)}"
            )

    def test_values_are_finite(self) -> None:
        """All output values must be finite (no NaN or Inf)."""
        audio = _make_sine(duration=1.0, sr=self.SR)
        result = time_stretch(audio, self.SR, ratio=1.25)
        assert np.all(np.isfinite(result)), "Output contains non-finite values"

    def test_values_in_audio_range(self) -> None:
        """Output values should be roughly in [-1, 1] for normalised input."""
        audio = _make_sine(duration=1.0, sr=self.SR)  # sine amplitude = 1
        result = time_stretch(audio, self.SR, ratio=1.0)
        assert float(np.max(np.abs(result))) <= 1.5  # allow some headroom

    def test_accepts_44100_sr(self) -> None:
        """Should work with 44100 Hz sample rate."""
        audio = _make_sine(duration=0.5, sr=44100)
        result = time_stretch(audio, 44100, ratio=1.0)
        assert result.ndim == 1

    # --- Error paths ---

    def test_non_array_raises(self) -> None:
        """Non-array audio should raise ValueError."""
        with pytest.raises(ValueError, match="numpy ndarray"):
            time_stretch([0.1, 0.2, 0.3], self.SR, ratio=1.0)  # type: ignore[arg-type]

    def test_2d_array_raises(self) -> None:
        """2-D audio array should raise ValueError."""
        audio = np.zeros((2, self.SR), dtype=np.float32)
        with pytest.raises(ValueError, match="1-D"):
            time_stretch(audio, self.SR, ratio=1.0)

    def test_empty_array_raises(self) -> None:
        """Empty array should raise ValueError."""
        audio = np.array([], dtype=np.float32)
        with pytest.raises(ValueError, match="empty"):
            time_stretch(audio, self.SR, ratio=1.0)

    def test_zero_ratio_raises(self) -> None:
        """ratio=0 should raise ValueError."""
        audio = _make_sine(duration=1.0)
        with pytest.raises(ValueError, match="ratio"):
            time_stretch(audio, self.SR, ratio=0.0)

    def test_negative_ratio_raises(self) -> None:
        """Negative ratio should raise ValueError."""
        audio = _make_sine(duration=1.0)
        with pytest.raises(ValueError, match="ratio"):
            time_stretch(audio, self.SR, ratio=-1.0)

    def test_nan_ratio_raises(self) -> None:
        """NaN ratio should raise ValueError."""
        audio = _make_sine(duration=1.0)
        with pytest.raises(ValueError, match="ratio"):
            time_stretch(audio, self.SR, ratio=float("nan"))

    def test_inf_ratio_raises(self) -> None:
        """Infinite ratio should raise ValueError."""
        audio = _make_sine(duration=1.0)
        with pytest.raises(ValueError, match="ratio"):
            time_stretch(audio, self.SR, ratio=float("inf"))

    def test_zero_sample_rate_raises(self) -> None:
        """sample_rate=0 should raise ValueError."""
        audio = _make_sine(duration=1.0)
        with pytest.raises(ValueError, match="sample_rate"):
            time_stretch(audio, 0, ratio=1.0)

    def test_negative_sample_rate_raises(self) -> None:
        """Negative sample_rate should raise ValueError."""
        audio = _make_sine(duration=1.0)
        with pytest.raises(ValueError, match="sample_rate"):
            time_stretch(audio, -22050, ratio=1.0)

    def test_float_sample_rate_raises(self) -> None:
        """Float sample_rate should raise ValueError."""
        audio = _make_sine(duration=1.0)
        with pytest.raises(ValueError, match="sample_rate"):
            time_stretch(audio, 22050.0, ratio=1.0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# pitch_shift
# ---------------------------------------------------------------------------


class TestPitchShift:
    """Tests for pitch_shift."""

    SR = 22050

    def test_returns_ndarray(self) -> None:
        """Output must be a numpy ndarray."""
        audio = _make_sine(duration=1.0, sr=self.SR)
        result = pitch_shift(audio, self.SR, semitones=0.0)
        assert isinstance(result, np.ndarray)

    def test_output_dtype_float32(self) -> None:
        """Output dtype must be float32."""
        audio = _make_sine(duration=1.0, sr=self.SR)
        result = pitch_shift(audio, self.SR, semitones=0.0)
        assert result.dtype == np.float32

    def test_output_is_1d(self) -> None:
        """Output must be 1-D."""
        audio = _make_sine(duration=1.0, sr=self.SR)
        result = pitch_shift(audio, self.SR, semitones=0.0)
        assert result.ndim == 1

    def test_zero_semitones_returns_copy_equal_to_input(self) -> None:
        """semitones=0.0 should return a copy equal to the input."""
        audio = _make_sine(duration=1.0, sr=self.SR)
        result = pitch_shift(audio, self.SR, semitones=0.0)
        np.testing.assert_array_equal(result, audio)
        assert result is not audio

    def test_output_length_preserved(self) -> None:
        """Pitch shift should preserve the number of samples."""
        audio = _make_sine(duration=1.0, sr=self.SR)
        for semitones in [-7.0, -1.0, 0.0, 1.0, 7.0]:
            result = pitch_shift(audio, self.SR, semitones=semitones)
            assert len(result) == len(audio), (
                f"semitones={semitones}: expected {len(audio)} samples, "
                f"got {len(result)}"
            )

    def test_values_are_finite_positive_shift(self) -> None:
        """All output values must be finite for a positive semitone shift."""
        audio = _make_sine(duration=1.0, sr=self.SR)
        result = pitch_shift(audio, self.SR, semitones=2.0)
        assert np.all(np.isfinite(result)), "Output contains non-finite values"

    def test_values_are_finite_negative_shift(self) -> None:
        """All output values must be finite for a negative semitone shift."""
        audio = _make_sine(duration=1.0, sr=self.SR)
        result = pitch_shift(audio, self.SR, semitones=-3.0)
        assert np.all(np.isfinite(result)), "Output contains non-finite values"

    def test_positive_semitones_raises_pitch(self) -> None:
        """Pitch-shifting up should raise the dominant frequency.

        We verify this by comparing the peak FFT frequency bin of the original
        sine (440 Hz) against the shifted version (+12 semitones = 880 Hz).
        """
        sr = self.SR
        freq = 440.0
        audio = _make_sine(frequency=freq, duration=1.0, sr=sr)
        shifted = pitch_shift(audio, sr, semitones=12.0)  # +1 octave

        # Find dominant frequency via FFT
        fft = np.abs(np.fft.rfft(shifted))
        freqs = np.fft.rfftfreq(len(shifted), d=1.0 / sr)
        dominant_freq = float(freqs[np.argmax(fft)])

        # Expect ~880 Hz (±20% tolerance for Rubber Band artefacts)
        expected = freq * 2.0
        assert expected * 0.8 <= dominant_freq <= expected * 1.2, (
            f"Expected dominant freq ~{expected} Hz, got {dominant_freq:.1f} Hz"
        )

    def test_fractional_semitones_accepted(self) -> None:
        """Fractional semitone values should be accepted without error."""
        audio = _make_sine(duration=1.0, sr=self.SR)
        result = pitch_shift(audio, self.SR, semitones=0.5)
        assert result.ndim == 1
        assert result.dtype == np.float32

    def test_accepts_44100_sr(self) -> None:
        """Should work with 44100 Hz sample rate."""
        audio = _make_sine(duration=0.5, sr=44100)
        result = pitch_shift(audio, 44100, semitones=2.0)
        assert result.ndim == 1
        assert len(result) == len(audio)

    # --- Error paths ---

    def test_non_array_raises(self) -> None:
        """Non-array audio should raise ValueError."""
        with pytest.raises(ValueError, match="numpy ndarray"):
            pitch_shift([0.1, 0.2, 0.3], self.SR, semitones=0.0)  # type: ignore[arg-type]

    def test_2d_array_raises(self) -> None:
        """2-D audio array should raise ValueError."""
        audio = np.zeros((2, self.SR), dtype=np.float32)
        with pytest.raises(ValueError, match="1-D"):
            pitch_shift(audio, self.SR, semitones=0.0)

    def test_empty_array_raises(self) -> None:
        """Empty array should raise ValueError."""
        audio = np.array([], dtype=np.float32)
        with pytest.raises(ValueError, match="empty"):
            pitch_shift(audio, self.SR, semitones=0.0)

    def test_zero_sample_rate_raises(self) -> None:
        """sample_rate=0 should raise ValueError."""
        audio = _make_sine(duration=1.0)
        with pytest.raises(ValueError, match="sample_rate"):
            pitch_shift(audio, 0, semitones=1.0)

    def test_negative_sample_rate_raises(self) -> None:
        """Negative sample_rate should raise ValueError."""
        audio = _make_sine(duration=1.0)
        with pytest.raises(ValueError, match="sample_rate"):
            pitch_shift(audio, -22050, semitones=1.0)

    def test_float_sample_rate_raises(self) -> None:
        """Float sample_rate should raise ValueError."""
        audio = _make_sine(duration=1.0)
        with pytest.raises(ValueError, match="sample_rate"):
            pitch_shift(audio, 22050.5, semitones=1.0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# process — combined transform orchestration
# ---------------------------------------------------------------------------


class TestProcess:
    """Tests for the process() high-level orchestration function."""

    SR = 22050

    def _audio(self, duration: float = 1.0) -> np.ndarray:
        return _make_sine(duration=duration, sr=self.SR)

    # --- No-op ---

    def test_no_transforms_returns_copy(self) -> None:
        """Calling process with no BPM or key args should return a copy of the input."""
        audio = self._audio()
        result = process(audio, self.SR)
        np.testing.assert_array_equal(result, audio)
        assert result is not audio

    def test_no_transforms_copy_is_float32(self) -> None:
        """Copy returned when no transforms are requested should be float32."""
        audio = self._audio()
        result = process(audio, self.SR)
        assert result.dtype == np.float32

    # --- BPM-only transform ---

    def test_bpm_transform_changes_length(self) -> None:
        """Applying a BPM transform should change the output length."""
        audio = self._audio(duration=2.0)
        result = process(audio, self.SR, source_bpm=120.0, target_bpm=60.0)
        # ratio = 120/60 = 2.0 -> output ~2x longer
        assert len(result) > len(audio)

    def test_bpm_transform_output_float32(self) -> None:
        """BPM-transformed output should be float32."""
        audio = self._audio()
        result = process(audio, self.SR, source_bpm=120.0, target_bpm=120.0)
        assert result.dtype == np.float32

    def test_bpm_no_change_preserves_approx_length(self) -> None:
        """Same source and target BPM should keep length approximately the same."""
        audio = self._audio(duration=1.0)
        result = process(audio, self.SR, source_bpm=120.0, target_bpm=120.0)
        # ratio = 1.0, length should be identical (short-circuit path)
        assert len(result) == len(audio)

    def test_bpm_target_without_source_raises(self) -> None:
        """Providing target_bpm without source_bpm should raise ValueError."""
        audio = self._audio()
        with pytest.raises(ValueError, match="source_bpm"):
            process(audio, self.SR, target_bpm=140.0)

    # --- Key-only transform ---

    def test_key_transform_preserves_length(self) -> None:
        """Pitch shift should preserve sample count."""
        audio = self._audio(duration=1.0)
        result = process(audio, self.SR, source_key="C", target_key="E")
        assert len(result) == len(audio)

    def test_key_transform_output_float32(self) -> None:
        """Key-transformed output should be float32."""
        audio = self._audio()
        result = process(audio, self.SR, source_key="Am", target_key="Cm")
        assert result.dtype == np.float32

    def test_key_no_change_returns_equal_audio(self) -> None:
        """Same source and target key should return audio equal to input."""
        audio = self._audio()
        result = process(audio, self.SR, source_key="C", target_key="C")
        np.testing.assert_array_equal(result, audio)

    def test_key_target_without_source_raises(self) -> None:
        """Providing target_key without source_key should raise ValueError."""
        audio = self._audio()
        with pytest.raises(ValueError, match="source_key"):
            process(audio, self.SR, target_key="Am")

    def test_invalid_target_key_raises(self) -> None:
        """Invalid target_key string should raise ValueError."""
        audio = self._audio()
        with pytest.raises(ValueError):
            process(audio, self.SR, source_key="C", target_key="ZZZ")

    def test_invalid_source_key_raises(self) -> None:
        """Invalid source_key string should raise ValueError."""
        audio = self._audio()
        with pytest.raises(ValueError):
            process(audio, self.SR, source_key="ZZZ", target_key="C")

    # --- Combined BPM + Key transforms ---

    def test_combined_transform_returns_ndarray(self) -> None:
        """Both BPM and key transforms should succeed and return ndarray."""
        audio = self._audio(duration=2.0)
        result = process(
            audio, self.SR,
            source_bpm=120.0, target_bpm=140.0,
            source_key="C", target_key="E",
        )
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    def test_combined_transform_length_reflects_bpm(self) -> None:
        """With both transforms, the length should reflect the BPM ratio applied first."""
        audio = self._audio(duration=2.0)
        result = process(
            audio, self.SR,
            source_bpm=120.0, target_bpm=60.0,  # ratio=2 -> 2x longer
            source_key="Am", target_key="Am",   # same key -> pitch shift is no-op
        )
        expected_len = len(audio) * 2
        # 5% tolerance
        assert abs(len(result) - expected_len) < expected_len * 0.05

    def test_combined_transform_values_finite(self) -> None:
        """Combined transform output values should all be finite."""
        audio = self._audio(duration=1.0)
        result = process(
            audio, self.SR,
            source_bpm=120.0, target_bpm=100.0,
            source_key="C", target_key="G",
        )
        assert np.all(np.isfinite(result))

    # --- Input validation ---

    def test_non_array_raises(self) -> None:
        """Non-array audio should raise ValueError."""
        with pytest.raises(ValueError, match="numpy ndarray"):
            process([0.1, 0.2], self.SR)  # type: ignore[arg-type]

    def test_2d_audio_raises(self) -> None:
        """2-D audio array should raise ValueError."""
        audio = np.zeros((2, self.SR), dtype=np.float32)
        with pytest.raises(ValueError, match="1-D"):
            process(audio, self.SR)

    def test_empty_audio_raises(self) -> None:
        """Empty audio array should raise ValueError."""
        audio = np.array([], dtype=np.float32)
        with pytest.raises(ValueError, match="empty"):
            process(audio, self.SR)

    def test_zero_sample_rate_raises(self) -> None:
        """sample_rate=0 should raise ValueError."""
        audio = self._audio()
        with pytest.raises(ValueError, match="sample_rate"):
            process(audio, 0)

    def test_negative_sample_rate_raises(self) -> None:
        """Negative sample_rate should raise ValueError."""
        audio = self._audio()
        with pytest.raises(ValueError, match="sample_rate"):
            process(audio, -22050)


# ---------------------------------------------------------------------------
# _validate_bpm
# ---------------------------------------------------------------------------


class TestValidateBpm:
    """Tests for the _validate_bpm private helper."""

    def test_valid_positive_float(self) -> None:
        """A positive float should pass without raising."""
        _validate_bpm(120.0, "bpm")  # Should not raise

    def test_valid_positive_int(self) -> None:
        """A positive integer should pass without raising."""
        _validate_bpm(140, "bpm")  # Should not raise

    def test_zero_raises(self) -> None:
        """Zero should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            _validate_bpm(0.0, "bpm")

    def test_negative_raises(self) -> None:
        """Negative value should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            _validate_bpm(-1.0, "bpm")

    def test_nan_raises(self) -> None:
        """NaN should raise ValueError."""
        with pytest.raises(ValueError, match="finite"):
            _validate_bpm(float("nan"), "bpm")

    def test_positive_inf_raises(self) -> None:
        """Positive infinity should raise ValueError."""
        with pytest.raises(ValueError, match="finite"):
            _validate_bpm(float("inf"), "bpm")

    def test_negative_inf_raises(self) -> None:
        """Negative infinity should raise ValueError."""
        with pytest.raises(ValueError, match="finite"):
            _validate_bpm(float("-inf"), "bpm")

    def test_string_raises(self) -> None:
        """String input should raise ValueError."""
        with pytest.raises(ValueError):
            _validate_bpm("120", "bpm")  # type: ignore[arg-type]

    def test_none_raises(self) -> None:
        """None should raise ValueError."""
        with pytest.raises(ValueError):
            _validate_bpm(None, "bpm")  # type: ignore[arg-type]

    def test_bool_raises(self) -> None:
        """Bool should raise ValueError (bool is subclass of int)."""
        with pytest.raises(ValueError):
            _validate_bpm(True, "bpm")  # type: ignore[arg-type]

    def test_param_name_in_error_message(self) -> None:
        """The parameter name should appear in the error message."""
        with pytest.raises(ValueError, match="my_param"):
            _validate_bpm(-5.0, "my_param")


# ---------------------------------------------------------------------------
# _validate_audio_1d
# ---------------------------------------------------------------------------


class TestValidateAudio1d:
    """Tests for the _validate_audio_1d private helper."""

    def test_valid_1d_array(self) -> None:
        """A non-empty 1-D float32 array should pass."""
        audio = np.zeros(100, dtype=np.float32)
        _validate_audio_1d(audio)  # Should not raise

    def test_2d_array_raises(self) -> None:
        """2-D array should raise ValueError mentioning '1-D'."""
        audio = np.zeros((2, 100), dtype=np.float32)
        with pytest.raises(ValueError, match="1-D"):
            _validate_audio_1d(audio)

    def test_0d_array_raises(self) -> None:
        """0-D (scalar) array should raise ValueError."""
        audio = np.float32(0.5)
        with pytest.raises(ValueError, match="1-D"):
            _validate_audio_1d(audio)

    def test_3d_array_raises(self) -> None:
        """3-D array should raise ValueError."""
        audio = np.zeros((1, 2, 100), dtype=np.float32)
        with pytest.raises(ValueError, match="1-D"):
            _validate_audio_1d(audio)

    def test_empty_array_raises(self) -> None:
        """Empty 1-D array should raise ValueError mentioning 'empty'."""
        audio = np.array([], dtype=np.float32)
        with pytest.raises(ValueError, match="empty"):
            _validate_audio_1d(audio)

    def test_list_raises(self) -> None:
        """Python list should raise ValueError."""
        with pytest.raises(ValueError, match="numpy ndarray"):
            _validate_audio_1d([0.1, 0.2, 0.3])  # type: ignore[arg-type]

    def test_none_raises(self) -> None:
        """None should raise ValueError."""
        with pytest.raises(ValueError, match="numpy ndarray"):
            _validate_audio_1d(None)  # type: ignore[arg-type]

    def test_int_raises(self) -> None:
        """Scalar int should raise ValueError."""
        with pytest.raises(ValueError, match="numpy ndarray"):
            _validate_audio_1d(42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _validate_sample_rate
# ---------------------------------------------------------------------------


class TestValidateSampleRate:
    """Tests for the _validate_sample_rate private helper."""

    def test_valid_sr(self) -> None:
        """A positive integer should pass."""
        _validate_sample_rate(22050)  # Should not raise
        _validate_sample_rate(44100)
        _validate_sample_rate(48000)

    def test_zero_raises(self) -> None:
        """sample_rate=0 should raise ValueError."""
        with pytest.raises(ValueError, match="sample_rate"):
            _validate_sample_rate(0)

    def test_negative_raises(self) -> None:
        """Negative sample_rate should raise ValueError."""
        with pytest.raises(ValueError, match="sample_rate"):
            _validate_sample_rate(-44100)

    def test_float_raises(self) -> None:
        """Float sample_rate should raise ValueError."""
        with pytest.raises(ValueError, match="sample_rate"):
            _validate_sample_rate(22050.0)  # type: ignore[arg-type]

    def test_string_raises(self) -> None:
        """String sample_rate should raise ValueError."""
        with pytest.raises(ValueError, match="sample_rate"):
            _validate_sample_rate("44100")  # type: ignore[arg-type]

    def test_none_raises(self) -> None:
        """None should raise ValueError."""
        with pytest.raises(ValueError, match="sample_rate"):
            _validate_sample_rate(None)  # type: ignore[arg-type]

    def test_bool_raises(self) -> None:
        """Bool should raise ValueError."""
        with pytest.raises(ValueError, match="sample_rate"):
            _validate_sample_rate(True)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _validate_ratio
# ---------------------------------------------------------------------------


class TestValidateRatio:
    """Tests for the _validate_ratio private helper."""

    def test_valid_ratio_one(self) -> None:
        """ratio=1.0 should pass."""
        _validate_ratio(1.0)  # Should not raise

    def test_valid_ratio_point_five(self) -> None:
        """ratio=0.5 should pass."""
        _validate_ratio(0.5)  # Should not raise

    def test_valid_ratio_two(self) -> None:
        """ratio=2.0 should pass."""
        _validate_ratio(2.0)  # Should not raise

    def test_valid_ratio_small(self) -> None:
        """Very small positive ratio should pass."""
        _validate_ratio(0.001)  # Should not raise

    def test_zero_raises(self) -> None:
        """ratio=0 should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            _validate_ratio(0.0)

    def test_negative_raises(self) -> None:
        """Negative ratio should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            _validate_ratio(-1.0)

    def test_nan_raises(self) -> None:
        """NaN ratio should raise ValueError."""
        with pytest.raises(ValueError, match="finite"):
            _validate_ratio(float("nan"))

    def test_inf_raises(self) -> None:
        """Infinite ratio should raise ValueError."""
        with pytest.raises(ValueError, match="finite"):
            _validate_ratio(float("inf"))

    def test_string_raises(self) -> None:
        """String ratio should raise ValueError."""
        with pytest.raises(ValueError):
            _validate_ratio("1.0")  # type: ignore[arg-type]

    def test_bool_raises(self) -> None:
        """Bool should raise ValueError."""
        with pytest.raises(ValueError):
            _validate_ratio(True)  # type: ignore[arg-type]

    def test_none_raises(self) -> None:
        """None should raise ValueError."""
        with pytest.raises(ValueError):
            _validate_ratio(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Parametric BPM ratio correctness
# ---------------------------------------------------------------------------


class TestBpmRatioParametric:
    """Parametric correctness tests for compute_time_stretch_ratio."""

    @pytest.mark.parametrize(
        "source_bpm, target_bpm, expected_ratio",
        [
            (120.0, 120.0, 1.0),
            (120.0, 60.0, 2.0),
            (60.0, 120.0, 0.5),
            (140.0, 70.0, 2.0),
            (90.0, 180.0, 0.5),
            (100.0, 125.0, 0.8),
            (174.0, 87.0, 2.0),
            (128.0, 96.0, 128.0 / 96.0),
        ],
    )
    def test_ratio_values(self, source_bpm: float, target_bpm: float, expected_ratio: float) -> None:
        """Ratio should equal source_bpm / target_bpm."""
        assert compute_time_stretch_ratio(source_bpm, target_bpm) == pytest.approx(
            expected_ratio, rel=1e-9
        )


# ---------------------------------------------------------------------------
# Parametric pitch shift semitones correctness
# ---------------------------------------------------------------------------


class TestPitchShiftSemitonesParametric:
    """Parametric correctness tests for compute_pitch_shift_semitones."""

    @pytest.mark.parametrize(
        "src, tgt, expected",
        [
            ("C", "C", 0),
            ("C", "C#", 1),
            ("C", "D", 2),
            ("C", "Eb", 3),
            ("C", "E", 4),
            ("C", "F", 5),
            ("C", "F#", 6),
            ("C", "G", -5),
            ("C", "Ab", -4),
            ("C", "A", -3),
            ("C", "Bb", -2),
            ("C", "B", -1),
            ("Am", "Cm", 3),
            ("F#m", "Cm", -6),
            ("Bb", "F", 7 - 12),  # = -5
            ("G", "C", 5),
        ],
    )
    def test_semitone_values(self, src: str, tgt: str, expected: int) -> None:
        """Semitone offset should match expected value."""
        assert compute_pitch_shift_semitones(src, tgt) == expected
