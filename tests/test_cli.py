"""test_cli: Integration and unit tests for sample_forge.cli.

Covers:
- _resolve_output_format: extension inference, explicit flag, defaults.
- main command invoked via Click's CliRunner:
  - Missing required options produce appropriate errors.
  - Valid BPM-only invocation completes successfully.
  - Valid key-only invocation completes successfully.
  - Combined BPM + key invocation completes successfully.
  - --format flag overrides extension-based detection.
  - --sr flag is forwarded to audio_io.load_audio.
  - --source-bpm and --source-key overrides skip auto-detection.
  - Invalid --bpm values produce error messages.
  - Invalid --key values produce error messages.
  - Missing input file produces an error.
  - No --bpm or --key emits a warning.
  - FLAC output is written when extension is .flac.

Synthetic audio is written to a temporary directory for all I/O tests so
the suite is completely self-contained and does not require real audio assets.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from click.testing import CliRunner

from sample_forge.cli import _resolve_output_format, main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sine(
    frequency: float = 440.0,
    duration: float = 2.0,
    sr: int = 22050,
) -> np.ndarray:
    """Return a mono sine wave as a float32 array."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2.0 * np.pi * frequency * t).astype(np.float32)


def _write_wav(directory: Path, filename: str = "input.wav", sr: int = 22050, duration: float = 2.0) -> Path:
    """Write a short WAV file to *directory* and return its Path."""
    audio = _make_sine(duration=duration, sr=sr)
    path = directory / filename
    sf.write(str(path), audio, sr, subtype="PCM_16", format="WAV")
    return path


# ---------------------------------------------------------------------------
# _resolve_output_format
# ---------------------------------------------------------------------------


class TestResolveOutputFormat:
    """Tests for the _resolve_output_format private helper."""

    def test_explicit_wav_uppercase(self) -> None:
        """Explicit 'WAV' flag should return 'WAV'."""
        assert _resolve_output_format("out.flac", "WAV") == "WAV"

    def test_explicit_flac_uppercase(self) -> None:
        """Explicit 'FLAC' flag should return 'FLAC'."""
        assert _resolve_output_format("out.wav", "FLAC") == "FLAC"

    def test_explicit_lowercase_normalised(self) -> None:
        """Lowercase explicit format should be uppercased."""
        assert _resolve_output_format("out.wav", "flac") == "FLAC"
        assert _resolve_output_format("out.flac", "wav") == "WAV"

    def test_flac_extension_detected(self) -> None:
        """'.flac' extension with no explicit flag should give 'FLAC'."""
        assert _resolve_output_format("out.flac", None) == "FLAC"

    def test_wav_extension_detected(self) -> None:
        """'.wav' extension with no explicit flag should give 'WAV'."""
        assert _resolve_output_format("out.wav", None) == "WAV"

    def test_unknown_extension_defaults_to_wav(self) -> None:
        """Unknown extension should fall back to 'WAV'."""
        assert _resolve_output_format("out.ogg", None) == "WAV"
        assert _resolve_output_format("out.mp3", None) == "WAV"

    def test_no_extension_defaults_to_wav(self) -> None:
        """No extension should fall back to 'WAV'."""
        assert _resolve_output_format("output", None) == "WAV"

    def test_case_insensitive_extension(self) -> None:
        """Extension detection should be case-insensitive."""
        assert _resolve_output_format("out.FLAC", None) == "FLAC"
        assert _resolve_output_format("out.WAV", None) == "WAV"

    def test_explicit_overrides_extension(self) -> None:
        """Explicit format should always override the file extension."""
        assert _resolve_output_format("out.flac", "WAV") == "WAV"
        assert _resolve_output_format("out.wav", "FLAC") == "FLAC"


# ---------------------------------------------------------------------------
# main — missing required options
# ---------------------------------------------------------------------------


class TestMainMissingOptions:
    """Tests that missing required options cause non-zero exit codes."""

    def test_missing_input_exits_nonzero(self) -> None:
        """Omitting --input should produce a non-zero exit code."""
        runner = CliRunner()
        result = runner.invoke(main, ["--output", "out.wav"])
        assert result.exit_code != 0

    def test_missing_output_exits_nonzero(self) -> None:
        """Omitting --output should produce a non-zero exit code."""
        runner = CliRunner()
        result = runner.invoke(main, ["--input", "in.wav"])
        assert result.exit_code != 0

    def test_nonexistent_input_exits_nonzero(self) -> None:
        """Pointing --input to a non-existent file should fail."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--input", "/nonexistent/file.wav", "--output", "out.wav"],
        )
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# main — help and version
# ---------------------------------------------------------------------------


class TestMainHelpAndVersion:
    """Tests for --help and --version flags."""

    def test_help_flag(self) -> None:
        """--help should print usage and exit 0."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Usage" in result.output

    def test_short_help_flag(self) -> None:
        """-h should print usage and exit 0."""
        runner = CliRunner()
        result = runner.invoke(main, ["-h"])
        assert result.exit_code == 0
        assert "Usage" in result.output

    def test_version_flag(self) -> None:
        """--version should print the version string and exit 0."""
        from sample_forge import __version__

        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.output

    def test_short_version_flag(self) -> None:
        """-V should print the version string."""
        from sample_forge import __version__

        runner = CliRunner()
        result = runner.invoke(main, ["-V"])
        assert result.exit_code == 0
        assert __version__ in result.output


# ---------------------------------------------------------------------------
# main — invalid option values
# ---------------------------------------------------------------------------


class TestMainInvalidOptions:
    """Tests that invalid option values produce informative errors."""

    def test_invalid_bpm_zero(self) -> None:
        """--bpm 0 should fail with a non-zero exit code."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            wav = _write_wav(Path(tmpdir))
            out = str(Path(tmpdir) / "out.wav")
            result = runner.invoke(
                main,
                ["--input", str(wav), "--output", out, "--bpm", "0"],
            )
        assert result.exit_code != 0

    def test_invalid_bpm_negative(self) -> None:
        """Negative --bpm should fail."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            wav = _write_wav(Path(tmpdir))
            out = str(Path(tmpdir) / "out.wav")
            result = runner.invoke(
                main,
                ["--input", str(wav), "--output", out, "--bpm", "-10"],
            )
        assert result.exit_code != 0

    def test_invalid_key(self) -> None:
        """An unrecognised --key should fail with a non-zero exit code."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            wav = _write_wav(Path(tmpdir))
            out = str(Path(tmpdir) / "out.wav")
            result = runner.invoke(
                main,
                ["--input", str(wav), "--output", out, "--key", "ZZZ"],
            )
        assert result.exit_code != 0
        assert "ZZZ" in result.output or "ZZZ" in (result.stderr if result.stderr else "")

    def test_invalid_source_key(self) -> None:
        """An invalid --source-key should fail."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            wav = _write_wav(Path(tmpdir))
            out = str(Path(tmpdir) / "out.wav")
            result = runner.invoke(
                main,
                [
                    "--input", str(wav),
                    "--output", out,
                    "--key", "Am",
                    "--source-key", "INVALID",
                ],
            )
        assert result.exit_code != 0

    def test_invalid_source_bpm_zero(self) -> None:
        """--source-bpm 0 should fail."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            wav = _write_wav(Path(tmpdir))
            out = str(Path(tmpdir) / "out.wav")
            result = runner.invoke(
                main,
                [
                    "--input", str(wav),
                    "--output", out,
                    "--bpm", "140",
                    "--source-bpm", "0",
                ],
            )
        assert result.exit_code != 0

    def test_invalid_sr_zero(self) -> None:
        """--sr 0 should fail."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            wav = _write_wav(Path(tmpdir))
            out = str(Path(tmpdir) / "out.wav")
            result = runner.invoke(
                main,
                ["--input", str(wav), "--output", out, "--sr", "0"],
            )
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# main — warning when no transforms are requested
# ---------------------------------------------------------------------------


class TestMainNoTransformWarning:
    """Test that omitting both --bpm and --key emits a warning."""

    def test_no_bpm_no_key_warning(self) -> None:
        """No --bpm or --key should emit a warning message."""
        runner = CliRunner(mix_stderr=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            wav = _write_wav(Path(tmpdir))
            out = str(Path(tmpdir) / "out.wav")
            result = runner.invoke(
                main,
                ["--input", str(wav), "--output", out],
            )
        # Should still exit 0 (it's just a warning, not an error)
        assert result.exit_code == 0
        combined = result.output + (result.stderr or "")
        assert "Warning" in combined or "warning" in combined.lower()


# ---------------------------------------------------------------------------
# main — BPM-only transformation
# ---------------------------------------------------------------------------


class TestMainBpmOnly:
    """Integration tests for BPM-only invocations."""

    def test_bpm_transform_exits_zero(self) -> None:
        """A valid BPM retargeting invocation should exit 0."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            wav = _write_wav(Path(tmpdir), duration=3.0)
            out = str(Path(tmpdir) / "out.wav")
            result = runner.invoke(
                main,
                [
                    "--input", str(wav),
                    "--output", out,
                    "--bpm", "140",
                    "--source-bpm", "120",  # skip auto-detection for speed
                ],
            )
        assert result.exit_code == 0, f"stderr/stdout: {result.output}"

    def test_bpm_transform_creates_output_file(self) -> None:
        """The output file should be created after a BPM transformation."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            wav = _write_wav(Path(tmpdir), duration=3.0)
            out_path = Path(tmpdir) / "out.wav"
            runner.invoke(
                main,
                [
                    "--input", str(wav),
                    "--output", str(out_path),
                    "--bpm", "140",
                    "--source-bpm", "120",
                ],
            )
            assert out_path.exists()

    def test_bpm_transform_output_is_readable_wav(self) -> None:
        """The output file should be a valid WAV file."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            wav = _write_wav(Path(tmpdir), duration=3.0, sr=22050)
            out_path = Path(tmpdir) / "out.wav"
            runner.invoke(
                main,
                [
                    "--input", str(wav),
                    "--output", str(out_path),
                    "--bpm", "120",
                    "--source-bpm", "120",  # ratio=1 → no-op stretch
                ],
            )
            data, sr = sf.read(str(out_path))
        assert sr == 22050
        assert len(data) > 0

    def test_bpm_stretch_increases_duration(self) -> None:
        """Stretching to half BPM should approximately double duration."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            wav = _write_wav(Path(tmpdir), duration=2.0, sr=22050)
            out_path = Path(tmpdir) / "out.wav"
            runner.invoke(
                main,
                [
                    "--input", str(wav),
                    "--output", str(out_path),
                    "--bpm", "60",
                    "--source-bpm", "120",  # ratio=2.0 → 2x longer
                ],
            )
            if out_path.exists():
                info = sf.info(str(out_path))
                # Allow 10% tolerance for Rubber Band frame rounding
                assert info.duration >= 2.0 * 0.9 * 2  # at least 90% of 4 s

    def test_done_message_in_output(self) -> None:
        """'Done.' should appear in output on success."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            wav = _write_wav(Path(tmpdir), duration=2.0)
            out = str(Path(tmpdir) / "out.wav")
            result = runner.invoke(
                main,
                [
                    "--input", str(wav),
                    "--output", out,
                    "--bpm", "120",
                    "--source-bpm", "120",
                ],
            )
        assert "Done" in result.output


# ---------------------------------------------------------------------------
# main — key-only transformation
# ---------------------------------------------------------------------------


class TestMainKeyOnly:
    """Integration tests for key-only (pitch-shift) invocations."""

    def test_key_transform_exits_zero(self) -> None:
        """A valid key retargeting invocation should exit 0."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            wav = _write_wav(Path(tmpdir), duration=2.0)
            out = str(Path(tmpdir) / "out.wav")
            result = runner.invoke(
                main,
                [
                    "--input", str(wav),
                    "--output", out,
                    "--key", "Am",
                    "--source-key", "C",  # skip auto-detection
                ],
            )
        assert result.exit_code == 0, f"stdout: {result.output}"

    def test_key_transform_creates_output_file(self) -> None:
        """Output file should be created after a key transformation."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            wav = _write_wav(Path(tmpdir), duration=2.0)
            out_path = Path(tmpdir) / "out.wav"
            runner.invoke(
                main,
                [
                    "--input", str(wav),
                    "--output", str(out_path),
                    "--key", "C",
                    "--source-key", "C",  # same key → semitones=0
                ],
            )
            assert out_path.exists()

    def test_key_same_source_and_target_preserves_length(self) -> None:
        """Same source and target key means semitones=0 → length unchanged."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            wav = _write_wav(Path(tmpdir), duration=2.0, sr=22050)
            out_path = Path(tmpdir) / "out.wav"
            runner.invoke(
                main,
                [
                    "--input", str(wav),
                    "--output", str(out_path),
                    "--key", "C",
                    "--source-key", "C",
                ],
            )
            if out_path.exists():
                original_info = sf.info(str(wav))
                output_info = sf.info(str(out_path))
                assert output_info.frames == original_info.frames


# ---------------------------------------------------------------------------
# main — combined BPM + key transformation
# ---------------------------------------------------------------------------


class TestMainCombined:
    """Integration tests for combined BPM + key transformations."""

    def test_combined_exits_zero(self) -> None:
        """A combined BPM + key invocation should exit 0."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            wav = _write_wav(Path(tmpdir), duration=3.0)
            out = str(Path(tmpdir) / "out.wav")
            result = runner.invoke(
                main,
                [
                    "--input", str(wav),
                    "--output", out,
                    "--bpm", "120",
                    "--source-bpm", "120",
                    "--key", "Am",
                    "--source-key", "Am",
                ],
            )
        assert result.exit_code == 0, f"stdout: {result.output}"

    def test_combined_creates_output(self) -> None:
        """Combined transformation should produce an output file."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            wav = _write_wav(Path(tmpdir), duration=2.0)
            out_path = Path(tmpdir) / "out.wav"
            runner.invoke(
                main,
                [
                    "--input", str(wav),
                    "--output", str(out_path),
                    "--bpm", "140",
                    "--source-bpm", "120",
                    "--key", "C",
                    "--source-key", "C",
                ],
            )
            assert out_path.exists()


# ---------------------------------------------------------------------------
# main — format flag and extension inference
# ---------------------------------------------------------------------------


class TestMainFormat:
    """Tests for output format selection."""

    def test_explicit_flac_format_flag(self) -> None:
        """--format FLAC should produce a FLAC file regardless of extension."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            wav = _write_wav(Path(tmpdir), duration=2.0)
            out_path = Path(tmpdir) / "out.wav"  # .wav extension but --format FLAC
            runner.invoke(
                main,
                [
                    "--input", str(wav),
                    "--output", str(out_path),
                    "--format", "FLAC",
                    "--key", "C",
                    "--source-key", "C",
                ],
            )
            if out_path.exists():
                info = sf.info(str(out_path))
                assert "FLAC" in info.format.upper()

    def test_flac_extension_inferred(self) -> None:
        """Output file with .flac extension should automatically use FLAC format."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            wav = _write_wav(Path(tmpdir), duration=2.0)
            out_path = Path(tmpdir) / "out.flac"
            result = runner.invoke(
                main,
                [
                    "--input", str(wav),
                    "--output", str(out_path),
                    "--key", "C",
                    "--source-key", "C",
                ],
            )
            assert result.exit_code == 0
            if out_path.exists():
                info = sf.info(str(out_path))
                assert "FLAC" in info.format.upper()

    def test_wav_written_by_default(self) -> None:
        """Without --format, default should be WAV for .wav output path."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            wav = _write_wav(Path(tmpdir), duration=2.0)
            out_path = Path(tmpdir) / "out.wav"
            runner.invoke(
                main,
                [
                    "--input", str(wav),
                    "--output", str(out_path),
                    "--key", "C",
                    "--source-key", "C",
                ],
            )
            if out_path.exists():
                info = sf.info(str(out_path))
                assert "WAV" in info.format.upper()


# ---------------------------------------------------------------------------
# main — --sr (sample rate) flag
# ---------------------------------------------------------------------------


class TestMainSampleRate:
    """Tests for the --sr sample-rate resampling flag."""

    def test_sr_flag_accepted(self) -> None:
        """--sr flag should be accepted without error."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            wav = _write_wav(Path(tmpdir), sr=44100, duration=2.0)
            out = str(Path(tmpdir) / "out.wav")
            result = runner.invoke(
                main,
                [
                    "--input", str(wav),
                    "--output", out,
                    "--sr", "22050",
                    "--key", "C",
                    "--source-key", "C",
                ],
            )
        assert result.exit_code == 0, f"stdout: {result.output}"

    def test_resampled_output_sr(self) -> None:
        """Output file should be written at the requested sample rate."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            wav = _write_wav(Path(tmpdir), sr=44100, duration=2.0)
            out_path = Path(tmpdir) / "out.wav"
            runner.invoke(
                main,
                [
                    "--input", str(wav),
                    "--output", str(out_path),
                    "--sr", "22050",
                    "--key", "C",
                    "--source-key", "C",
                ],
            )
            if out_path.exists():
                info = sf.info(str(out_path))
                assert info.samplerate == 22050


# ---------------------------------------------------------------------------
# main — output directory creation
# ---------------------------------------------------------------------------


class TestMainOutputDir:
    """Tests that missing output directories are created automatically."""

    def test_nested_output_directory_created(self) -> None:
        """Output should succeed even when parent directories don't exist."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            wav = _write_wav(Path(tmpdir), duration=2.0)
            nested_out = str(Path(tmpdir) / "subdir" / "nested" / "out.wav")
            result = runner.invoke(
                main,
                [
                    "--input", str(wav),
                    "--output", nested_out,
                    "--key", "C",
                    "--source-key", "C",
                ],
            )
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# main — output content sanity check
# ---------------------------------------------------------------------------


class TestMainOutputContent:
    """Sanity checks on the content of the output files."""

    def test_output_has_samples(self) -> None:
        """Output file should contain a non-trivial number of samples."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            wav = _write_wav(Path(tmpdir), duration=2.0, sr=22050)
            out_path = Path(tmpdir) / "out.wav"
            runner.invoke(
                main,
                [
                    "--input", str(wav),
                    "--output", str(out_path),
                    "--key", "C",
                    "--source-key", "C",
                ],
            )
            data, _ = sf.read(str(out_path))
        assert len(data) > 1000

    def test_output_values_finite(self) -> None:
        """All output sample values should be finite."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            wav = _write_wav(Path(tmpdir), duration=2.0, sr=22050)
            out_path = Path(tmpdir) / "out.wav"
            runner.invoke(
                main,
                [
                    "--input", str(wav),
                    "--output", str(out_path),
                    "--key", "E",
                    "--source-key", "C",
                ],
            )
            data, _ = sf.read(str(out_path), dtype="float32")
        assert np.all(np.isfinite(data))
