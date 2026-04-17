"""cli: Command-line interface for sample_forge.

Provides a Click-based CLI entry point that accepts --input, --output, --bpm,
--key, and --format flags and orchestrates the full analysis → process → export
pipeline.

This module is intentionally thin: it delegates all heavy work to the
``analyzer``, ``processor``, ``audio_io``, and ``key_utils`` modules.

Example usage::

    $ sample-forge -i break.wav -o break_140.wav --bpm 140
    $ sample-forge -i riff.wav -o riff_Am.wav --key Am
    $ sample-forge -i loop.wav -o loop_out.flac --bpm 128 --key C --format FLAC
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import click

from sample_forge import __version__


# ---------------------------------------------------------------------------
# CLI definition
# ---------------------------------------------------------------------------


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, "-V", "--version", prog_name="sample-forge")
@click.option(
    "-i",
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    help="Path to the source audio file (WAV, FLAC, MP3, OGG, …).",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    required=True,
    type=click.Path(file_okay=True, dir_okay=False, writable=True),
    help="Destination file path for the processed audio.",
)
@click.option(
    "--bpm",
    "target_bpm",
    default=None,
    type=float,
    help="Target BPM. If omitted, BPM is not changed.",
)
@click.option(
    "--key",
    "target_key",
    default=None,
    type=str,
    help=(
        "Target musical key (e.g. 'C', 'F#m', 'Bbmaj'). "
        "If omitted, key is not changed."
    ),
)
@click.option(
    "--format",
    "output_format",
    default=None,
    type=click.Choice(["WAV", "FLAC", "wav", "flac"], case_sensitive=False),
    help=(
        "Output format. Defaults to the extension of --output, "
        "or WAV if the extension is not recognised."
    ),
)
@click.option(
    "--sr",
    "sample_rate",
    default=None,
    type=int,
    help=(
        "Resample the source audio to this sample rate before processing "
        "(e.g. 44100, 48000). Uses the native sample rate if omitted."
    ),
)
@click.option(
    "--source-bpm",
    "source_bpm_override",
    default=None,
    type=float,
    help=(
        "Manually specify the source BPM instead of auto-detecting it. "
        "Only used when --bpm is also set."
    ),
)
@click.option(
    "--source-key",
    "source_key_override",
    default=None,
    type=str,
    help=(
        "Manually specify the source key instead of auto-detecting it. "
        "Only used when --key is also set."
    ),
)
def main(
    input_path: str,
    output_path: str,
    target_bpm: Optional[float],
    target_key: Optional[str],
    output_format: Optional[str],
    sample_rate: Optional[int],
    source_bpm_override: Optional[float],
    source_key_override: Optional[str],
) -> None:
    """sample-forge — offline BPM and key retargeting for audio samples.

    Loads the INPUT audio file, optionally detects its BPM and/or musical
    key, applies time-stretching and/or pitch-shifting to reach the TARGET
    values, and writes the result to OUTPUT.

    \b
    Examples:
      # Stretch a loop from its native BPM to 140
      sample-forge -i break.wav -o break_140.wav --bpm 140

      # Transpose to A minor
      sample-forge -i riff.wav -o riff_Am.wav --key Am

      # Both at once, output as FLAC
      sample-forge -i loop.wav -o loop_out.flac --bpm 128 --key C --format FLAC

      # Provide known source values to skip auto-detection
      sample-forge -i loop.wav -o out.wav --bpm 140 --source-bpm 120 --key Am --source-key C
    """
    # Deferred imports so the CLI starts quickly even when heavy deps are slow.
    from sample_forge import analyzer, audio_io, key_utils, processor

    # ------------------------------------------------------------------
    # 1. Validate user-supplied values early, before any I/O
    # ------------------------------------------------------------------
    if target_key is not None:
        if not key_utils.validate_key(target_key):
            click.echo(
                f"Error: '{target_key}' is not a valid key name. "
                "Examples: C, F#m, Bbmaj, Aminor.",
                err=True,
            )
            sys.exit(1)

    if source_key_override is not None:
        if not key_utils.validate_key(source_key_override):
            click.echo(
                f"Error: '--source-key {source_key_override}' is not a valid key name. "
                "Examples: C, F#m, Bbmaj, Aminor.",
                err=True,
            )
            sys.exit(1)

    if target_bpm is not None and target_bpm <= 0:
        click.echo(
            f"Error: --bpm must be a positive number, got {target_bpm}.",
            err=True,
        )
        sys.exit(1)

    if source_bpm_override is not None and source_bpm_override <= 0:
        click.echo(
            f"Error: --source-bpm must be a positive number, got {source_bpm_override}.",
            err=True,
        )
        sys.exit(1)

    if sample_rate is not None and sample_rate <= 0:
        click.echo(
            f"Error: --sr must be a positive integer, got {sample_rate}.",
            err=True,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Resolve output format from explicit flag or file extension
    # ------------------------------------------------------------------
    resolved_format = _resolve_output_format(output_path, output_format)

    # ------------------------------------------------------------------
    # 3. Load the source audio file
    # ------------------------------------------------------------------
    click.echo(f"Loading {input_path} …")
    try:
        audio, sr = audio_io.load_audio(input_path, target_sr=sample_rate, mono=True)
    except FileNotFoundError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    except (ValueError, RuntimeError) as exc:
        click.echo(f"Error loading audio: {exc}", err=True)
        sys.exit(1)

    duration_s = len(audio) / sr
    click.echo(f"  Loaded {duration_s:.2f} s  |  {sr} Hz  |  mono")

    # ------------------------------------------------------------------
    # 4. Analyse the source audio (BPM and/or key) as needed
    # ------------------------------------------------------------------
    source_bpm: Optional[float] = None
    source_key: Optional[str] = None

    need_bpm = target_bpm is not None
    need_key = target_key is not None

    if need_bpm:
        if source_bpm_override is not None:
            source_bpm = source_bpm_override
            click.echo(f"  Source BPM (manual override): {source_bpm:.2f}")
        else:
            click.echo("  Detecting source BPM …")
            try:
                source_bpm = analyzer.detect_bpm(audio, sr)
            except ValueError as exc:
                click.echo(f"Error: audio unsuitable for BPM detection: {exc}", err=True)
                sys.exit(1)
            except RuntimeError as exc:
                click.echo(f"Error during BPM detection: {exc}", err=True)
                sys.exit(1)
            click.echo(f"  Detected BPM: {source_bpm:.2f}")

    if need_key:
        if source_key_override is not None:
            source_key = source_key_override
            click.echo(f"  Source key (manual override): {source_key}")
        else:
            click.echo("  Estimating source key …")
            try:
                pc, mode = analyzer.estimate_key(audio, sr)
            except ValueError as exc:
                click.echo(f"Error: audio unsuitable for key estimation: {exc}", err=True)
                sys.exit(1)
            except RuntimeError as exc:
                click.echo(f"Error during key estimation: {exc}", err=True)
                sys.exit(1)

            # Build a canonical key string like 'Am' or 'C'
            note_name = key_utils.pitch_class_to_name(pc, prefer_sharps=True)
            source_key = note_name + ("m" if mode == "minor" else "")
            click.echo(f"  Estimated key: {source_key} ({mode})")

    # ------------------------------------------------------------------
    # 5. Warn when no transforms are requested
    # ------------------------------------------------------------------
    if not need_bpm and not need_key:
        click.echo(
            "Warning: neither --bpm nor --key was specified. "
            "The output will be identical to the input.",
            err=True,
        )

    # ------------------------------------------------------------------
    # 6. Process (time-stretch and/or pitch-shift)
    # ------------------------------------------------------------------
    click.echo("Processing …")
    try:
        processed = processor.process(
            audio,
            sr,
            source_bpm=source_bpm,
            target_bpm=target_bpm,
            source_key=source_key,
            target_key=target_key,
        )
    except ValueError as exc:
        click.echo(f"Error: invalid processing parameters: {exc}", err=True)
        sys.exit(1)
    except RuntimeError as exc:
        click.echo(f"Error during processing: {exc}", err=True)
        sys.exit(1)

    out_duration_s = len(processed) / sr
    click.echo(f"  Output duration: {out_duration_s:.2f} s")

    if need_bpm and source_bpm is not None and target_bpm is not None:
        click.echo(
            f"  BPM: {source_bpm:.2f} → {target_bpm:.2f}  "
            f"(ratio {source_bpm / target_bpm:.4f})"
        )

    if need_key and source_key is not None and target_key is not None:
        semitones = key_utils.semitone_offset(source_key, target_key)
        direction = "up" if semitones > 0 else "down" if semitones < 0 else "unchanged"
        click.echo(
            f"  Key: {source_key} → {target_key}  "
            f"({abs(semitones)} semitone(s) {direction})"
        )

    # ------------------------------------------------------------------
    # 7. Write the processed audio to disk
    # ------------------------------------------------------------------
    click.echo(f"Writing {output_path} ({resolved_format}) …")
    try:
        audio_io.write_audio(
            output_path,
            processed,
            sr,
            fmt=resolved_format,
        )
    except ValueError as exc:
        click.echo(f"Error: invalid output parameters: {exc}", err=True)
        sys.exit(1)
    except OSError as exc:
        click.echo(f"Error writing file: {exc}", err=True)
        sys.exit(1)
    except RuntimeError as exc:
        click.echo(f"Error writing output: {exc}", err=True)
        sys.exit(1)

    click.echo("Done.")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _resolve_output_format(output_path: str, explicit_format: Optional[str]) -> str:
    """Determine the output format string from the path extension or explicit flag.

    Priority order:

    1. The explicit ``--format`` flag, if provided.
    2. The file extension of *output_path* (``.wav`` → ``'WAV'``, ``.flac`` → ``'FLAC'``).
    3. ``'WAV'`` as the default fallback.

    Args:
        output_path: The destination file path string.
        explicit_format: User-supplied format string from the ``--format`` flag,
            or ``None`` if the flag was not specified.

    Returns:
        An uppercase format string — either ``'WAV'`` or ``'FLAC'``.
    """
    if explicit_format is not None:
        return explicit_format.upper()

    suffix = Path(output_path).suffix.lower()
    if suffix == ".flac":
        return "FLAC"
    # Default to WAV for .wav, unknown extensions, and no extension at all.
    return "WAV"
