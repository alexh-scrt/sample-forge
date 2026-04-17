"""test_smoke: Basic import and version sanity checks for Phase 1.

Ensures that the package and all stub modules import cleanly and that the
version string is present.
"""

import importlib

import pytest


def test_package_imports() -> None:
    """The top-level package must import without errors."""
    import sample_forge

    assert sample_forge is not None


def test_version_string() -> None:
    """__version__ must be a non-empty string."""
    from sample_forge import __version__

    assert isinstance(__version__, str)
    assert len(__version__) > 0


def test_version_format() -> None:
    """__version__ should follow MAJOR.MINOR.PATCH."""
    from sample_forge import __version__

    parts = __version__.split(".")
    assert len(parts) == 3, f"Expected 3 version parts, got {parts}"
    for part in parts:
        assert part.isdigit(), f"Non-numeric version part: {part}"


@pytest.mark.parametrize(
    "module_name",
    [
        "sample_forge.analyzer",
        "sample_forge.processor",
        "sample_forge.audio_io",
        "sample_forge.key_utils",
        "sample_forge.cli",
    ],
)
def test_module_imports(module_name: str) -> None:
    """Every stub module must be importable."""
    mod = importlib.import_module(module_name)
    assert mod is not None


def test_key_utils_public_api() -> None:
    """key_utils must expose the expected public functions."""
    from sample_forge import key_utils

    assert callable(key_utils.parse_key)
    assert callable(key_utils.validate_key)
    assert callable(key_utils.semitone_offset)
    assert callable(key_utils.key_to_pitch_class)
    assert callable(key_utils.pitch_class_to_name)


def test_audio_io_public_api() -> None:
    """audio_io must expose the expected public functions."""
    from sample_forge import audio_io

    assert callable(audio_io.load_audio)
    assert callable(audio_io.write_audio)
    assert callable(audio_io.get_supported_formats)


def test_analyzer_public_api() -> None:
    """analyzer must expose the expected public functions."""
    from sample_forge import analyzer

    assert callable(analyzer.detect_bpm)
    assert callable(analyzer.estimate_key)


def test_processor_public_api() -> None:
    """processor must expose the expected public functions."""
    from sample_forge import processor

    assert callable(processor.time_stretch)
    assert callable(processor.pitch_shift)
    assert callable(processor.process)
    assert callable(processor.compute_time_stretch_ratio)
    assert callable(processor.compute_pitch_shift_semitones)


def test_supported_formats() -> None:
    """get_supported_formats must return a list containing WAV and FLAC."""
    from sample_forge.audio_io import get_supported_formats

    fmts = get_supported_formats()
    assert isinstance(fmts, list)
    assert "WAV" in fmts
    assert "FLAC" in fmts


def test_parse_key_basic() -> None:
    """parse_key must handle simple major and minor inputs."""
    from sample_forge.key_utils import parse_key

    pc, mode = parse_key("C")
    assert pc == 0
    assert mode == "major"

    pc, mode = parse_key("Am")
    assert pc == 9
    assert mode == "minor"


def test_semitone_offset_identity() -> None:
    """semitone_offset of same key must be 0."""
    from sample_forge.key_utils import semitone_offset

    assert semitone_offset("C", "C") == 0
    assert semitone_offset("Am", "A") == 0


def test_compute_time_stretch_ratio_identity() -> None:
    """compute_time_stretch_ratio with equal BPMs must return 1.0."""
    from sample_forge.processor import compute_time_stretch_ratio

    assert compute_time_stretch_ratio(120.0, 120.0) == pytest.approx(1.0)


def test_compute_time_stretch_ratio_double() -> None:
    """Halving the BPM should give a ratio of 2.0 (slow down by 2x)."""
    from sample_forge.processor import compute_time_stretch_ratio

    assert compute_time_stretch_ratio(140.0, 70.0) == pytest.approx(2.0)
