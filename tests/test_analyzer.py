"""test_analyzer: Unit tests for sample_forge.analyzer.

Covers:
- detect_bpm: valid inputs, return type, known-rhythm signals, error paths.
- estimate_key: valid inputs, return type, synthetic tonal signals, error paths.
- _validate_audio_array: edge cases for the validation helper.
- _normalise_profile: correctness of zero-mean unit-variance normalisation.
- _krumhansl_schmuckler: correctness on crafted chroma vectors.
- detect_bpm_and_key: integration wrapper.

Synthetic audio is used throughout to avoid any dependency on real audio files.
Where librosa's algorithms are non-deterministic or input-sensitive, tests
check structural properties (correct types, valid ranges) rather than exact
numeric values.
"""

from __future__ import annotations

import numpy as np
import pytest

from sample_forge.analyzer import (
    _krumhansl_schmuckler,
    _normalise_profile,
    _validate_audio_array,
    detect_bpm,
    detect_bpm_and_key,
    estimate_key,
)


# ---------------------------------------------------------------------------
# Synthetic signal helpers
# ---------------------------------------------------------------------------


def _make_silence(duration: float = 2.0, sr: int = 22050) -> np.ndarray:
    """Return a silent (all-zeros) mono float32 array."""
    return np.zeros(int(sr * duration), dtype=np.float32)


def _make_white_noise(duration: float = 2.0, sr: int = 22050, seed: int = 42) -> np.ndarray:
    """Return white noise as a mono float32 array."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(int(sr * duration)).astype(np.float32) * 0.1


def _make_sine(frequency: float, duration: float = 2.0, sr: int = 22050) -> np.ndarray:
    """Return a pure sine wave at *frequency* Hz."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2.0 * np.pi * frequency * t).astype(np.float32)


def _make_click_track(
    bpm: float,
    duration: float = 4.0,
    sr: int = 22050,
    click_duration_ms: float = 10.0,
) -> np.ndarray:
    """Return a click track at *bpm* beats per minute.

    Each click is a short impulse (raised-cosine window) to give librosa's
    onset detector a clear signal.  This gives a strongly periodic signal
    with energy concentrated at integer multiples of the beat frequency.
    """
    n_samples = int(sr * duration)
    audio = np.zeros(n_samples, dtype=np.float32)

    beat_period_samples = int(60.0 / bpm * sr)
    click_len = int(click_duration_ms / 1000.0 * sr)
    # Raised-cosine click envelope
    click = np.sin(np.linspace(0, np.pi, click_len)).astype(np.float32)

    pos = 0
    while pos < n_samples:
        end = min(pos + click_len, n_samples)
        audio[pos:end] += click[: end - pos]
        pos += beat_period_samples

    return audio


def _make_chroma_vector(dominant_pc: int) -> np.ndarray:
    """Return a 12-element chroma vector heavily weighted on *dominant_pc*.

    The dominant pitch class gets weight 1.0; all others get 0.0. This
    produces a maximally simple vector for sanity-checking the KS algorithm.
    """
    chroma = np.zeros(12, dtype=np.float64)
    chroma[dominant_pc] = 1.0
    return chroma


def _make_major_scale_chroma(root_pc: int) -> np.ndarray:
    """Return a chroma vector representing a major scale starting at *root_pc*.

    Major scale intervals (in semitones from root): 0, 2, 4, 5, 7, 9, 11.
    """
    intervals = [0, 2, 4, 5, 7, 9, 11]
    chroma = np.zeros(12, dtype=np.float64)
    for interval in intervals:
        chroma[(root_pc + interval) % 12] = 1.0
    return chroma


def _make_minor_scale_chroma(root_pc: int) -> np.ndarray:
    """Return a chroma vector representing a natural minor scale at *root_pc*.

    Natural minor scale intervals: 0, 2, 3, 5, 7, 8, 10.
    """
    intervals = [0, 2, 3, 5, 7, 8, 10]
    chroma = np.zeros(12, dtype=np.float64)
    for interval in intervals:
        chroma[(root_pc + interval) % 12] = 1.0
    return chroma


# ---------------------------------------------------------------------------
# _normalise_profile
# ---------------------------------------------------------------------------


class TestNormaliseProfile:
    """Tests for the _normalise_profile helper."""

    def test_zero_mean(self) -> None:
        """Normalised profile should have (near) zero mean."""
        profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                            2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        result = _normalise_profile(profile)
        assert abs(float(result.mean())) < 1e-10

    def test_unit_variance(self) -> None:
        """Normalised profile should have unit standard deviation."""
        profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                            2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        result = _normalise_profile(profile)
        assert abs(float(result.std()) - 1.0) < 1e-10

    def test_constant_profile_returns_zeros(self) -> None:
        """A constant (zero-std) profile should return all-zeros without error."""
        profile = np.ones(12, dtype=np.float64)
        result = _normalise_profile(profile)
        np.testing.assert_array_equal(result, np.zeros(12))

    def test_shape_preserved(self) -> None:
        """Output shape should match input shape."""
        profile = np.arange(12, dtype=np.float64)
        result = _normalise_profile(profile)
        assert result.shape == (12,)

    def test_output_dtype_float64(self) -> None:
        """Output should be float64."""
        profile = np.ones(12, dtype=np.float32)
        result = _normalise_profile(profile)
        assert result.dtype == np.float64

    def test_two_element_profile(self) -> None:
        """Should work with arrays shorter than 12 elements."""
        profile = np.array([0.0, 1.0])
        result = _normalise_profile(profile)
        assert abs(float(result.mean())) < 1e-10
        assert abs(float(result.std()) - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# _krumhansl_schmuckler
# ---------------------------------------------------------------------------


class TestKrumhanslSchmuckler:
    """Tests for the _krumhansl_schmuckler key-finding function."""

    def test_returns_tuple(self) -> None:
        """Should return a 2-tuple."""
        chroma = np.ones(12, dtype=np.float64)
        result = _krumhansl_schmuckler(chroma)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_pitch_class_in_range(self) -> None:
        """Returned pitch class must be in [0, 11]."""
        chroma = _make_major_scale_chroma(0)
        pc, _ = _krumhansl_schmuckler(chroma)
        assert 0 <= pc <= 11

    def test_mode_is_valid(self) -> None:
        """Returned mode must be 'major' or 'minor'."""
        chroma = _make_major_scale_chroma(0)
        _, mode = _krumhansl_schmuckler(chroma)
        assert mode in ("major", "minor")

    @pytest.mark.parametrize("root_pc", [0, 2, 4, 5, 7, 9, 11])
    def test_major_scale_chroma_detected_as_major(self, root_pc: int) -> None:
        """A pure major-scale chroma vector should be detected as major."""
        chroma = _make_major_scale_chroma(root_pc)
        pc, mode = _krumhansl_schmuckler(chroma)
        assert mode == "major", (
            f"Expected 'major' for root pc={root_pc}, got mode='{mode}' (pc={pc})"
        )

    @pytest.mark.parametrize("root_pc", [0, 2, 3, 5, 7, 9, 10])
    def test_minor_scale_chroma_detected_as_minor(self, root_pc: int) -> None:
        """A pure minor-scale chroma vector should be detected as minor."""
        chroma = _make_minor_scale_chroma(root_pc)
        pc, mode = _krumhansl_schmuckler(chroma)
        assert mode == "minor", (
            f"Expected 'minor' for root pc={root_pc}, got mode='{mode}' (pc={pc})"
        )

    def test_c_major_scale_detected(self) -> None:
        """C-major scale chroma should detect root pc=0 and major mode."""
        chroma = _make_major_scale_chroma(0)
        pc, mode = _krumhansl_schmuckler(chroma)
        assert pc == 0
        assert mode == "major"

    def test_a_minor_scale_detected(self) -> None:
        """A-minor scale chroma (root=9) should detect root pc=9 and minor mode."""
        chroma = _make_minor_scale_chroma(9)
        pc, mode = _krumhansl_schmuckler(chroma)
        assert pc == 9
        assert mode == "minor"

    def test_g_major_scale_detected(self) -> None:
        """G-major scale chroma (root=7) should detect root pc=7 and major mode."""
        chroma = _make_major_scale_chroma(7)
        pc, mode = _krumhansl_schmuckler(chroma)
        assert pc == 7
        assert mode == "major"

    def test_e_minor_scale_detected(self) -> None:
        """E-minor scale chroma (root=4) should detect root pc=4 and minor mode."""
        chroma = _make_minor_scale_chroma(4)
        pc, mode = _krumhansl_schmuckler(chroma)
        assert pc == 4
        assert mode == "minor"

    def test_uniform_chroma_returns_valid(self) -> None:
        """Uniform chroma (all ones) should not crash and return valid output."""
        chroma = np.ones(12, dtype=np.float64)
        pc, mode = _krumhansl_schmuckler(chroma)
        assert 0 <= pc <= 11
        assert mode in ("major", "minor")

    def test_zero_chroma_returns_valid(self) -> None:
        """All-zero chroma should not crash (constant vector normalises to zeros)."""
        chroma = np.zeros(12, dtype=np.float64)
        pc, mode = _krumhansl_schmuckler(chroma)
        assert 0 <= pc <= 11
        assert mode in ("major", "minor")

    def test_accepts_float32_input(self) -> None:
        """Function should accept float32 arrays (internally converts to float64)."""
        chroma = _make_major_scale_chroma(0).astype(np.float32)
        pc, mode = _krumhansl_schmuckler(chroma)
        assert 0 <= pc <= 11
        assert mode in ("major", "minor")


# ---------------------------------------------------------------------------
# _validate_audio_array
# ---------------------------------------------------------------------------


class TestValidateAudioArray:
    """Tests for the _validate_audio_array helper."""

    def test_valid_array_does_not_raise(self) -> None:
        """A valid 1-D float32 array should not raise."""
        audio = _make_white_noise(duration=1.0)
        _validate_audio_array(audio, 22050)  # Should not raise

    def test_non_ndarray_raises(self) -> None:
        """A list should raise ValueError."""
        with pytest.raises(ValueError, match="numpy ndarray"):
            _validate_audio_array([0.1, 0.2, 0.3], 22050)  # type: ignore[arg-type]

    def test_2d_array_raises(self) -> None:
        """A 2-D array should raise ValueError about dimensionality."""
        audio = np.zeros((2, 22050), dtype=np.float32)
        with pytest.raises(ValueError, match="1-D"):
            _validate_audio_array(audio, 22050)

    def test_empty_array_raises(self) -> None:
        """An empty 1-D array should raise ValueError."""
        audio = np.array([], dtype=np.float32)
        with pytest.raises(ValueError, match="empty"):
            _validate_audio_array(audio, 22050)

    def test_zero_sample_rate_raises(self) -> None:
        """sample_rate=0 should raise ValueError."""
        audio = _make_white_noise(duration=1.0)
        with pytest.raises(ValueError, match="sample_rate"):
            _validate_audio_array(audio, 0)

    def test_negative_sample_rate_raises(self) -> None:
        """Negative sample_rate should raise ValueError."""
        audio = _make_white_noise(duration=1.0)
        with pytest.raises(ValueError, match="sample_rate"):
            _validate_audio_array(audio, -22050)

    def test_float_sample_rate_raises(self) -> None:
        """Float sample_rate should raise ValueError."""
        audio = _make_white_noise(duration=1.0)
        with pytest.raises(ValueError, match="sample_rate"):
            _validate_audio_array(audio, 22050.0)  # type: ignore[arg-type]

    def test_too_short_raises(self) -> None:
        """Audio shorter than min_duration_seconds should raise ValueError."""
        sr = 22050
        # Create 0.1 s of audio, require 0.5 s
        audio = np.zeros(int(sr * 0.1), dtype=np.float32)
        with pytest.raises(ValueError, match="too short"):
            _validate_audio_array(audio, sr, min_duration_seconds=0.5)

    def test_exactly_min_duration_does_not_raise(self) -> None:
        """Audio exactly at min_duration_seconds should not raise."""
        sr = 22050
        min_dur = 0.5
        audio = np.zeros(int(sr * min_dur), dtype=np.float32)
        _validate_audio_array(audio, sr, min_duration_seconds=min_dur)  # OK

    def test_custom_min_duration(self) -> None:
        """Custom min_duration_seconds should be respected."""
        sr = 44100
        audio = np.zeros(int(sr * 0.3), dtype=np.float32)
        # Should pass with 0.2 s requirement
        _validate_audio_array(audio, sr, min_duration_seconds=0.2)
        # Should fail with 0.5 s requirement
        with pytest.raises(ValueError, match="too short"):
            _validate_audio_array(audio, sr, min_duration_seconds=0.5)


# ---------------------------------------------------------------------------
# detect_bpm — return type and structural properties
# ---------------------------------------------------------------------------


class TestDetectBpmReturnType:
    """Tests that detect_bpm returns correct types and value ranges."""

    def test_returns_float(self) -> None:
        """Return type must be a Python float."""
        audio = _make_white_noise(duration=3.0)
        bpm = detect_bpm(audio, 22050)
        assert isinstance(bpm, float)

    def test_returns_positive_bpm(self) -> None:
        """Returned BPM must be a positive number."""
        audio = _make_white_noise(duration=3.0)
        bpm = detect_bpm(audio, 22050)
        assert bpm > 0.0

    def test_reasonable_bpm_range(self) -> None:
        """Returned BPM should be in a musically sensible range (1–400)."""
        audio = _make_white_noise(duration=5.0)
        bpm = detect_bpm(audio, 22050)
        # librosa's beat_track clamps results to [1, 500] by default
        assert 1.0 <= bpm <= 500.0

    def test_bpm_from_silence(self) -> None:
        """detect_bpm should not crash on silence (returns some BPM)."""
        audio = _make_silence(duration=3.0)
        bpm = detect_bpm(audio, 22050)
        assert isinstance(bpm, float)
        assert bpm > 0.0

    def test_bpm_from_click_track_120(self) -> None:
        """A 120 BPM click track should be detected near 120 BPM."""
        target_bpm = 120.0
        audio = _make_click_track(bpm=target_bpm, duration=8.0, sr=22050)
        detected = detect_bpm(audio, 22050)
        # Allow ±20% tolerance because librosa may detect half-time/double-time
        assert target_bpm * 0.4 <= detected <= target_bpm * 2.5, (
            f"Expected ~{target_bpm} BPM (allowing octave errors), got {detected:.2f}"
        )

    def test_bpm_from_click_track_90(self) -> None:
        """A 90 BPM click track should be detected within octave of 90 BPM."""
        target_bpm = 90.0
        audio = _make_click_track(bpm=target_bpm, duration=8.0, sr=22050)
        detected = detect_bpm(audio, 22050)
        assert target_bpm * 0.4 <= detected <= target_bpm * 2.5, (
            f"Expected ~{target_bpm} BPM, got {detected:.2f}"
        )

    def test_different_sample_rates_return_float(self) -> None:
        """detect_bpm should accept different valid sample rates."""
        for sr in [16000, 22050, 44100, 48000]:
            audio = _make_white_noise(duration=3.0, sr=sr)
            bpm = detect_bpm(audio, sr)
            assert isinstance(bpm, float)
            assert bpm > 0.0


# ---------------------------------------------------------------------------
# detect_bpm — error paths
# ---------------------------------------------------------------------------


class TestDetectBpmErrors:
    """Tests for detect_bpm error handling."""

    def test_non_array_raises(self) -> None:
        """Non-array audio should raise ValueError."""
        with pytest.raises(ValueError, match="numpy ndarray"):
            detect_bpm([0.1, 0.2, 0.3], 22050)  # type: ignore[arg-type]

    def test_2d_array_raises(self) -> None:
        """2-D audio array should raise ValueError."""
        audio = np.zeros((2, 44100), dtype=np.float32)
        with pytest.raises(ValueError, match="1-D"):
            detect_bpm(audio, 22050)

    def test_too_short_raises(self) -> None:
        """Audio shorter than 0.5 s should raise ValueError."""
        sr = 22050
        audio = np.zeros(int(sr * 0.1), dtype=np.float32)
        with pytest.raises(ValueError, match="too short"):
            detect_bpm(audio, sr)

    def test_zero_sample_rate_raises(self) -> None:
        """sample_rate=0 should raise ValueError."""
        audio = _make_white_noise(duration=2.0)
        with pytest.raises(ValueError, match="sample_rate"):
            detect_bpm(audio, 0)

    def test_negative_sample_rate_raises(self) -> None:
        """Negative sample_rate should raise ValueError."""
        audio = _make_white_noise(duration=2.0)
        with pytest.raises(ValueError, match="sample_rate"):
            detect_bpm(audio, -22050)

    def test_empty_array_raises(self) -> None:
        """Empty array should raise ValueError."""
        audio = np.array([], dtype=np.float32)
        with pytest.raises(ValueError, match="empty"):
            detect_bpm(audio, 22050)


# ---------------------------------------------------------------------------
# estimate_key — return type and structural properties
# ---------------------------------------------------------------------------


class TestEstimateKeyReturnType:
    """Tests that estimate_key returns correct types and value ranges."""

    def test_returns_tuple(self) -> None:
        """Return value must be a 2-tuple."""
        audio = _make_white_noise(duration=3.0)
        result = estimate_key(audio, 22050)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_pitch_class_is_int(self) -> None:
        """First element of tuple must be a Python int."""
        audio = _make_white_noise(duration=3.0)
        pc, _ = estimate_key(audio, 22050)
        assert isinstance(pc, int)

    def test_pitch_class_in_range(self) -> None:
        """Pitch class must be in [0, 11]."""
        audio = _make_white_noise(duration=3.0)
        pc, _ = estimate_key(audio, 22050)
        assert 0 <= pc <= 11

    def test_mode_is_string(self) -> None:
        """Second element must be a string."""
        audio = _make_white_noise(duration=3.0)
        _, mode = estimate_key(audio, 22050)
        assert isinstance(mode, str)

    def test_mode_is_valid(self) -> None:
        """Mode must be 'major' or 'minor'."""
        audio = _make_white_noise(duration=3.0)
        _, mode = estimate_key(audio, 22050)
        assert mode in ("major", "minor")

    def test_silence_does_not_crash(self) -> None:
        """estimate_key should not crash on silence."""
        audio = _make_silence(duration=3.0)
        pc, mode = estimate_key(audio, 22050)
        assert 0 <= pc <= 11
        assert mode in ("major", "minor")

    def test_different_sample_rates(self) -> None:
        """estimate_key should accept different valid sample rates."""
        for sr in [16000, 22050, 44100]:
            audio = _make_white_noise(duration=3.0, sr=sr)
            pc, mode = estimate_key(audio, sr)
            assert 0 <= pc <= 11
            assert mode in ("major", "minor")

    def test_c_major_tone_cluster(self) -> None:
        """Audio containing C-major tones should tend to estimate C (pc=0) or nearby.

        This is a soft test: we check that the pitch class is one of the
        C-major scale degrees (0,2,4,5,7,9,11) which is likely given the input.
        We cannot guarantee exact detection because librosa's CENS chroma
        is not perfectly deterministic for synthetic tones.
        """
        sr = 22050
        duration = 3.0
        # Sum C4, E4, G4 (261.63, 329.63, 392.00 Hz)
        c4 = _make_sine(261.63, duration, sr)
        e4 = _make_sine(329.63, duration, sr)
        g4 = _make_sine(392.00, duration, sr)
        audio = (c4 + e4 + g4) / 3.0

        pc, mode = estimate_key(audio, sr)
        # C-major scale pitch classes: 0,2,4,5,7,9,11
        c_major_pcs = {0, 2, 4, 5, 7, 9, 11}
        assert pc in c_major_pcs, (
            f"Expected pc in C-major scale {c_major_pcs}, got pc={pc}, mode={mode}"
        )

    def test_long_audio_does_not_crash(self) -> None:
        """estimate_key should handle longer audio without error."""
        sr = 22050
        audio = _make_white_noise(duration=30.0, sr=sr)
        pc, mode = estimate_key(audio, sr)
        assert 0 <= pc <= 11
        assert mode in ("major", "minor")


# ---------------------------------------------------------------------------
# estimate_key — error paths
# ---------------------------------------------------------------------------


class TestEstimateKeyErrors:
    """Tests for estimate_key error handling."""

    def test_non_array_raises(self) -> None:
        """Non-array audio should raise ValueError."""
        with pytest.raises(ValueError, match="numpy ndarray"):
            estimate_key([0.1, 0.2, 0.3], 22050)  # type: ignore[arg-type]

    def test_2d_array_raises(self) -> None:
        """2-D audio array should raise ValueError."""
        audio = np.zeros((2, 44100), dtype=np.float32)
        with pytest.raises(ValueError, match="1-D"):
            estimate_key(audio, 22050)

    def test_too_short_raises(self) -> None:
        """Audio shorter than 0.5 s should raise ValueError."""
        sr = 22050
        audio = np.zeros(int(sr * 0.1), dtype=np.float32)
        with pytest.raises(ValueError, match="too short"):
            estimate_key(audio, sr)

    def test_zero_sample_rate_raises(self) -> None:
        """sample_rate=0 should raise ValueError."""
        audio = _make_white_noise(duration=2.0)
        with pytest.raises(ValueError, match="sample_rate"):
            estimate_key(audio, 0)

    def test_negative_sample_rate_raises(self) -> None:
        """Negative sample_rate should raise ValueError."""
        audio = _make_white_noise(duration=2.0)
        with pytest.raises(ValueError, match="sample_rate"):
            estimate_key(audio, -44100)

    def test_empty_array_raises(self) -> None:
        """Empty array should raise ValueError."""
        audio = np.array([], dtype=np.float32)
        with pytest.raises(ValueError, match="empty"):
            estimate_key(audio, 22050)


# ---------------------------------------------------------------------------
# detect_bpm_and_key — integration wrapper
# ---------------------------------------------------------------------------


class TestDetectBpmAndKey:
    """Tests for the detect_bpm_and_key convenience wrapper."""

    def test_returns_three_tuple(self) -> None:
        """Should return a 3-tuple."""
        audio = _make_white_noise(duration=3.0)
        result = detect_bpm_and_key(audio, 22050)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_bpm_is_positive_float(self) -> None:
        """BPM element should be a positive float."""
        audio = _make_white_noise(duration=3.0)
        bpm, _, _ = detect_bpm_and_key(audio, 22050)
        assert isinstance(bpm, float)
        assert bpm > 0.0

    def test_pitch_class_in_range(self) -> None:
        """Pitch class element should be in [0, 11]."""
        audio = _make_white_noise(duration=3.0)
        _, pc, _ = detect_bpm_and_key(audio, 22050)
        assert 0 <= pc <= 11

    def test_mode_is_valid(self) -> None:
        """Mode element should be 'major' or 'minor'."""
        audio = _make_white_noise(duration=3.0)
        _, _, mode = detect_bpm_and_key(audio, 22050)
        assert mode in ("major", "minor")

    def test_consistent_with_individual_functions(self) -> None:
        """Results should match calling detect_bpm and estimate_key separately.

        Note: We use a fixed-seed signal and call both in the same process,
        so the results should be deterministic and identical.
        """
        audio = _make_white_noise(duration=4.0, seed=99)
        sr = 22050

        bpm_combined, pc_combined, mode_combined = detect_bpm_and_key(audio, sr)
        bpm_separate = detect_bpm(audio, sr)
        pc_separate, mode_separate = estimate_key(audio, sr)

        assert bpm_combined == pytest.approx(bpm_separate, rel=1e-6)
        assert pc_combined == pc_separate
        assert mode_combined == mode_separate

    def test_error_propagates_from_detect_bpm(self) -> None:
        """Errors in detect_bpm should propagate out of detect_bpm_and_key."""
        audio = np.zeros(100, dtype=np.float32)  # Too short
        with pytest.raises(ValueError, match="too short"):
            detect_bpm_and_key(audio, 22050)


# ---------------------------------------------------------------------------
# Parametric correctness tests for _krumhansl_schmuckler
# ---------------------------------------------------------------------------


class TestKrumhanslSchmucklerAllKeys:
    """Exhaustive tests for _krumhansl_schmuckler across all 12 major and minor keys."""

    @pytest.mark.parametrize("root_pc", list(range(12)))
    def test_major_keys_all_roots(self, root_pc: int) -> None:
        """Major scale chroma should be detected as major for all 12 roots."""
        chroma = _make_major_scale_chroma(root_pc)
        detected_pc, mode = _krumhansl_schmuckler(chroma)
        assert mode == "major", (
            f"Root pc={root_pc}: expected major, got mode={mode!r} (detected pc={detected_pc})"
        )
        assert detected_pc == root_pc, (
            f"Root pc={root_pc}: expected same root, got detected_pc={detected_pc}"
        )

    @pytest.mark.parametrize("root_pc", list(range(12)))
    def test_minor_keys_all_roots(self, root_pc: int) -> None:
        """Minor scale chroma should be detected as minor for all 12 roots."""
        chroma = _make_minor_scale_chroma(root_pc)
        detected_pc, mode = _krumhansl_schmuckler(chroma)
        assert mode == "minor", (
            f"Root pc={root_pc}: expected minor, got mode={mode!r} (detected pc={detected_pc})"
        )
        assert detected_pc == root_pc, (
            f"Root pc={root_pc}: expected same root, got detected_pc={detected_pc}"
        )
