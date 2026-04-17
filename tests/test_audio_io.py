"""test_audio_io: Unit tests for sample_forge.audio_io.

Tests cover:
- Loading WAV and FLAC files (written via soundfile to temp dirs)
- Writing mono and stereo arrays as WAV and FLAC
- Sample rate handling (native, resampled)
- Format and subtype validation
- Error paths: missing files, bad formats, bad arrays
- _prepare_array_for_write shape normalisation
- get_supported_formats
- get_audio_info metadata helper
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from sample_forge.audio_io import (
    _prepare_array_for_write,
    get_audio_info,
    get_supported_formats,
    load_audio,
    write_audio,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sine(frequency: float = 440.0, duration: float = 1.0, sr: int = 22050) -> np.ndarray:
    """Return a mono sine wave as a float32 array."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def _write_temp_wav(
    directory: Path,
    filename: str = "test.wav",
    sr: int = 22050,
    channels: int = 1,
    duration: float = 0.5,
) -> Path:
    """Write a temporary WAV file and return its path."""
    n_samples = int(sr * duration)
    if channels == 1:
        data = _make_sine(duration=duration, sr=sr)
    else:
        data = np.stack([_make_sine(440, duration, sr), _make_sine(880, duration, sr)], axis=1)
    out_path = directory / filename
    sf.write(str(out_path), data, sr, subtype="PCM_16", format="WAV")
    return out_path


def _write_temp_flac(
    directory: Path,
    filename: str = "test.flac",
    sr: int = 44100,
    duration: float = 0.5,
) -> Path:
    """Write a temporary FLAC file and return its path."""
    data = _make_sine(duration=duration, sr=sr)
    out_path = directory / filename
    sf.write(str(out_path), data, sr, subtype="PCM_24", format="FLAC")
    return out_path


# ---------------------------------------------------------------------------
# get_supported_formats
# ---------------------------------------------------------------------------


class TestGetSupportedFormats:
    """Tests for get_supported_formats."""

    def test_returns_list(self) -> None:
        """Should return a list."""
        result = get_supported_formats()
        assert isinstance(result, list)

    def test_contains_wav(self) -> None:
        """WAV must be in the supported formats."""
        assert "WAV" in get_supported_formats()

    def test_contains_flac(self) -> None:
        """FLAC must be in the supported formats."""
        assert "FLAC" in get_supported_formats()

    def test_all_uppercase(self) -> None:
        """All format strings should be uppercase."""
        for fmt in get_supported_formats():
            assert fmt == fmt.upper()


# ---------------------------------------------------------------------------
# load_audio — happy paths
# ---------------------------------------------------------------------------


class TestLoadAudioHappy:
    """Tests for load_audio with valid files."""

    def test_load_mono_wav(self) -> None:
        """Loading a mono WAV file should return a 1-D float32 array."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = _write_temp_wav(Path(tmpdir), sr=22050, channels=1)
            audio, sr = load_audio(wav_path)

        assert isinstance(audio, np.ndarray)
        assert audio.ndim == 1
        assert audio.dtype == np.float32
        assert sr == 22050

    def test_load_stereo_wav_as_mono(self) -> None:
        """Stereo WAV loaded with mono=True should be 1-D."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = _write_temp_wav(Path(tmpdir), sr=22050, channels=2)
            audio, sr = load_audio(wav_path, mono=True)

        assert audio.ndim == 1
        assert sr == 22050

    def test_load_stereo_wav_as_stereo(self) -> None:
        """Stereo WAV loaded with mono=False should be 2-D (channels, samples)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = _write_temp_wav(Path(tmpdir), sr=22050, channels=2)
            audio, sr = load_audio(wav_path, mono=False)

        assert audio.ndim == 2
        # librosa returns (channels, samples)
        assert audio.shape[0] == 2

    def test_load_flac(self) -> None:
        """Loading a FLAC file should succeed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            flac_path = _write_temp_flac(Path(tmpdir), sr=44100)
            audio, sr = load_audio(flac_path)

        assert isinstance(audio, np.ndarray)
        assert audio.ndim == 1
        assert sr == 44100

    def test_load_with_target_sr(self) -> None:
        """Specifying target_sr should resample to that rate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = _write_temp_wav(Path(tmpdir), sr=44100)
            audio, sr = load_audio(wav_path, target_sr=22050)

        assert sr == 22050
        # Duration should be ~0.5s at 22050 Hz
        expected_samples = int(22050 * 0.5)
        assert abs(len(audio) - expected_samples) < 100  # allow small rounding

    def test_returned_dtype_is_float32(self) -> None:
        """Returned audio array should always be float32."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = _write_temp_wav(Path(tmpdir))
            audio, _ = load_audio(wav_path)

        assert audio.dtype == np.float32

    def test_audio_values_in_range(self) -> None:
        """Float32 audio values should be in [-1.0, 1.0]."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = _write_temp_wav(Path(tmpdir))
            audio, _ = load_audio(wav_path)

        assert float(np.max(np.abs(audio))) <= 1.0 + 1e-5

    def test_accepts_string_path(self) -> None:
        """load_audio should accept string paths as well as Path objects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = _write_temp_wav(Path(tmpdir))
            audio, sr = load_audio(str(wav_path))

        assert audio.ndim == 1
        assert sr > 0

    def test_native_sr_preserved_when_no_target(self) -> None:
        """When target_sr is None, the native sample rate is returned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = _write_temp_wav(Path(tmpdir), sr=48000)
            _, sr = load_audio(wav_path, target_sr=None)

        assert sr == 48000

    def test_sample_count_matches_duration(self) -> None:
        """Number of samples should match duration * sample_rate."""
        duration = 0.5
        native_sr = 22050
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = _write_temp_wav(Path(tmpdir), sr=native_sr, duration=duration)
            audio, sr = load_audio(wav_path)

        expected = int(native_sr * duration)
        # Allow 1 sample tolerance for rounding
        assert abs(len(audio) - expected) <= 1


# ---------------------------------------------------------------------------
# load_audio — error paths
# ---------------------------------------------------------------------------


class TestLoadAudioErrors:
    """Tests for load_audio error handling."""

    def test_missing_file_raises_file_not_found(self) -> None:
        """Trying to load a non-existent path should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_audio("/nonexistent/path/audio.wav")

    def test_directory_path_raises_value_error(self) -> None:
        """Passing a directory instead of a file should raise ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError):
                load_audio(tmpdir)

    def test_invalid_target_sr_zero_raises(self) -> None:
        """target_sr=0 should raise ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = _write_temp_wav(Path(tmpdir))
            with pytest.raises(ValueError):
                load_audio(wav_path, target_sr=0)

    def test_invalid_target_sr_negative_raises(self) -> None:
        """Negative target_sr should raise ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = _write_temp_wav(Path(tmpdir))
            with pytest.raises(ValueError):
                load_audio(wav_path, target_sr=-1)

    def test_corrupt_file_raises_runtime_error(self) -> None:
        """A corrupt (non-audio) file should raise RuntimeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_file = Path(tmpdir) / "bad.wav"
            bad_file.write_bytes(b"not audio data at all")
            with pytest.raises(RuntimeError):
                load_audio(bad_file)


# ---------------------------------------------------------------------------
# write_audio — happy paths
# ---------------------------------------------------------------------------


class TestWriteAudioHappy:
    """Tests for write_audio with valid arguments."""

    def test_write_mono_wav(self) -> None:
        """Writing a mono float32 array as WAV should produce a readable file."""
        audio = _make_sine(duration=0.5, sr=22050)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "out.wav"
            write_audio(out_path, audio, 22050, fmt="WAV")

            assert out_path.exists()
            loaded, sr = sf.read(str(out_path))

        assert sr == 22050
        assert loaded.ndim == 1
        assert len(loaded) == len(audio)

    def test_write_mono_flac(self) -> None:
        """Writing a mono float32 array as FLAC should produce a readable file."""
        audio = _make_sine(duration=0.5, sr=44100)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "out.flac"
            write_audio(out_path, audio, 44100, fmt="FLAC")

            assert out_path.exists()
            loaded, sr = sf.read(str(out_path))

        assert sr == 44100
        assert abs(len(loaded) - len(audio)) <= 1

    def test_write_case_insensitive_fmt(self) -> None:
        """fmt='wav' (lowercase) should be accepted."""
        audio = _make_sine()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "out.wav"
            write_audio(out_path, audio, 22050, fmt="wav")
            assert out_path.exists()

    def test_write_creates_parent_dirs(self) -> None:
        """write_audio should create missing parent directories."""
        audio = _make_sine()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "subdir" / "nested" / "out.wav"
            write_audio(out_path, audio, 22050)
            assert out_path.exists()

    def test_write_stereo_array(self) -> None:
        """Writing a (channels, samples) stereo array should produce a stereo file."""
        left = _make_sine(440, duration=0.5, sr=22050)
        right = _make_sine(880, duration=0.5, sr=22050)
        stereo = np.stack([left, right], axis=0)  # (2, samples)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "stereo.wav"
            write_audio(out_path, stereo, 22050, fmt="WAV")

            loaded, sr = sf.read(str(out_path))

        assert loaded.ndim == 2
        assert loaded.shape[1] == 2  # (samples, channels)

    def test_write_accepts_string_path(self) -> None:
        """write_audio should accept string paths."""
        audio = _make_sine()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = str(Path(tmpdir) / "out.wav")
            write_audio(out_path, audio, 22050)
            assert Path(out_path).exists()

    def test_write_with_explicit_pcm_24_subtype(self) -> None:
        """Explicitly passing subtype='PCM_24' for WAV should work."""
        audio = _make_sine()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "out.wav"
            write_audio(out_path, audio, 22050, fmt="WAV", subtype="PCM_24")
            assert out_path.exists()

    def test_write_with_explicit_pcm_16_subtype_for_flac(self) -> None:
        """Explicitly passing subtype='PCM_16' for FLAC should work."""
        audio = _make_sine()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "out.flac"
            write_audio(out_path, audio, 22050, fmt="FLAC", subtype="PCM_16")
            assert out_path.exists()

    def test_roundtrip_wav_values_close(self) -> None:
        """Audio values should survive a WAV write/read roundtrip (within PCM_16 quantisation)."""
        original = _make_sine(duration=0.1, sr=22050)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "roundtrip.wav"
            write_audio(out_path, original, 22050, fmt="WAV", subtype="PCM_16")
            loaded, _ = sf.read(str(out_path), dtype="float32")

        np.testing.assert_allclose(original, loaded, atol=1e-3)

    def test_roundtrip_flac_values_close(self) -> None:
        """Audio values should survive a FLAC write/read roundtrip (within PCM_24 quantisation)."""
        original = _make_sine(duration=0.1, sr=44100)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "roundtrip.flac"
            write_audio(out_path, original, 44100, fmt="FLAC", subtype="PCM_24")
            loaded, _ = sf.read(str(out_path), dtype="float32")

        np.testing.assert_allclose(original, loaded, atol=1e-4)

    def test_default_subtype_wav_is_pcm16(self) -> None:
        """Default subtype for WAV should be PCM_16."""
        audio = _make_sine()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "out.wav"
            write_audio(out_path, audio, 22050, fmt="WAV")
            info = sf.info(str(out_path))

        assert "PCM_16" in info.subtype

    def test_default_subtype_flac_is_pcm24(self) -> None:
        """Default subtype for FLAC should be PCM_24."""
        audio = _make_sine()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "out.flac"
            write_audio(out_path, audio, 22050, fmt="FLAC")
            info = sf.info(str(out_path))

        assert "PCM_24" in info.subtype


# ---------------------------------------------------------------------------
# write_audio — error paths
# ---------------------------------------------------------------------------


class TestWriteAudioErrors:
    """Tests for write_audio error handling."""

    def test_unsupported_format_raises(self) -> None:
        """Unsupported fmt should raise ValueError."""
        audio = _make_sine()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "out.mp3"
            with pytest.raises(ValueError, match="Unsupported output format"):
                write_audio(out_path, audio, 22050, fmt="MP3")

    def test_zero_sample_rate_raises(self) -> None:
        """sample_rate=0 should raise ValueError."""
        audio = _make_sine()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "out.wav"
            with pytest.raises(ValueError):
                write_audio(out_path, audio, 0)

    def test_negative_sample_rate_raises(self) -> None:
        """Negative sample_rate should raise ValueError."""
        audio = _make_sine()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "out.wav"
            with pytest.raises(ValueError):
                write_audio(out_path, audio, -44100)

    def test_3d_array_raises(self) -> None:
        """A 3-D audio array should raise ValueError."""
        bad_audio = np.zeros((2, 100, 3), dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "out.wav"
            with pytest.raises(ValueError):
                write_audio(out_path, bad_audio, 22050)

    def test_empty_array_raises(self) -> None:
        """An empty audio array should raise ValueError."""
        bad_audio = np.array([], dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "out.wav"
            with pytest.raises(ValueError):
                write_audio(out_path, bad_audio, 22050)


# ---------------------------------------------------------------------------
# _prepare_array_for_write
# ---------------------------------------------------------------------------


class TestPrepareArrayForWrite:
    """Unit tests for the _prepare_array_for_write helper."""

    def test_1d_unchanged_shape(self) -> None:
        """1-D array should be returned as-is (still 1-D)."""
        arr = np.zeros(100, dtype=np.float32)
        out = _prepare_array_for_write(arr)
        assert out.ndim == 1
        assert out.shape == (100,)

    def test_1d_dtype_is_float32(self) -> None:
        """Output dtype should be float32 even if input is float64."""
        arr = np.zeros(100, dtype=np.float64)
        out = _prepare_array_for_write(arr)
        assert out.dtype == np.float32

    def test_channels_samples_transposed(self) -> None:
        """(channels, samples) array should be transposed to (samples, channels)."""
        arr = np.zeros((2, 1000), dtype=np.float32)  # 2 channels, 1000 samples
        out = _prepare_array_for_write(arr)
        assert out.shape == (1000, 2)

    def test_samples_channels_kept(self) -> None:
        """(samples, channels) array where rows > cols should not be transposed."""
        arr = np.zeros((1000, 2), dtype=np.float32)  # 1000 samples, 2 channels
        out = _prepare_array_for_write(arr)
        assert out.shape == (1000, 2)

    def test_3d_array_raises(self) -> None:
        """3-D arrays should raise ValueError."""
        arr = np.zeros((2, 100, 3), dtype=np.float32)
        with pytest.raises(ValueError):
            _prepare_array_for_write(arr)

    def test_non_array_raises(self) -> None:
        """Non-ndarray input should raise ValueError."""
        with pytest.raises(ValueError):
            _prepare_array_for_write([1.0, 2.0, 3.0])  # type: ignore[arg-type]

    def test_empty_array_raises(self) -> None:
        """Empty array should raise ValueError."""
        with pytest.raises(ValueError):
            _prepare_array_for_write(np.array([], dtype=np.float32))

    def test_output_is_copy_or_view_float32(self) -> None:
        """Output dtype is always float32."""
        arr = np.ones(50, dtype=np.int16)
        out = _prepare_array_for_write(arr)
        assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# get_audio_info
# ---------------------------------------------------------------------------


class TestGetAudioInfo:
    """Tests for get_audio_info metadata helper."""

    def test_returns_dict(self) -> None:
        """Should return a dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = _write_temp_wav(Path(tmpdir))
            info = get_audio_info(wav_path)

        assert isinstance(info, dict)

    def test_sample_rate_key(self) -> None:
        """dict should contain 'sample_rate'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = _write_temp_wav(Path(tmpdir), sr=22050)
            info = get_audio_info(wav_path)

        assert info["sample_rate"] == 22050

    def test_channels_mono(self) -> None:
        """Mono file should report 1 channel."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = _write_temp_wav(Path(tmpdir), channels=1)
            info = get_audio_info(wav_path)

        assert info["channels"] == 1

    def test_channels_stereo(self) -> None:
        """Stereo file should report 2 channels."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = _write_temp_wav(Path(tmpdir), channels=2)
            info = get_audio_info(wav_path)

        assert info["channels"] == 2

    def test_duration_approximate(self) -> None:
        """Reported duration should be close to the actual duration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = _write_temp_wav(Path(tmpdir), duration=0.5)
            info = get_audio_info(wav_path)

        assert abs(float(info["duration"]) - 0.5) < 0.02

    def test_frames_key_present(self) -> None:
        """dict should contain 'frames'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = _write_temp_wav(Path(tmpdir))
            info = get_audio_info(wav_path)

        assert "frames" in info
        assert isinstance(info["frames"], int)

    def test_format_key_present(self) -> None:
        """dict should contain 'format'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = _write_temp_wav(Path(tmpdir))
            info = get_audio_info(wav_path)

        assert "format" in info

    def test_subtype_key_present(self) -> None:
        """dict should contain 'subtype'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = _write_temp_wav(Path(tmpdir))
            info = get_audio_info(wav_path)

        assert "subtype" in info

    def test_missing_file_raises(self) -> None:
        """Missing file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            get_audio_info("/nonexistent/audio.wav")

    def test_corrupt_file_raises_runtime_error(self) -> None:
        """A corrupt file should raise RuntimeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_file = Path(tmpdir) / "bad.wav"
            bad_file.write_bytes(b"this is not audio")
            with pytest.raises(RuntimeError):
                get_audio_info(bad_file)

    def test_accepts_string_path(self) -> None:
        """Should accept a string path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = _write_temp_wav(Path(tmpdir))
            info = get_audio_info(str(wav_path))

        assert "sample_rate" in info

    def test_flac_info(self) -> None:
        """Should work for FLAC files as well."""
        with tempfile.TemporaryDirectory() as tmpdir:
            flac_path = _write_temp_flac(Path(tmpdir), sr=44100)
            info = get_audio_info(flac_path)

        assert info["sample_rate"] == 44100
        assert info["channels"] == 1
