"""cli: Command-line interface for sample_forge.

Provides a Click-based CLI entry point that accepts --input, --output, --bpm,
--key, and --format flags and orchestrates the full analysis → process → export
pipeline.

This module is intentionally thin: it delegates all heavy work to the
``analyzer``, ``processor``, ``audio_io``, and ``key_utils`` modules.
"""

from __future__ import annotations

import sys
from pathlib import Path

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
    show_default=True,
    help="Target BPM. If omitted, BPM is not changed.",
)
@click.option(
    "--key",
    "target_key",
    default=None,
    type=str,
    help="Target musical key (e.g. 'C', 'F#m', 'Bbmaj'). If omitted, key is not changed.",
)
@click.option(
    "--format",
    "output_format",
    default=None,
    type=click.Choice(["WAV", "FLAC", "wav", "flac"], case_sensitive=False),
    help=(
        "Output format. Defaults to the extension of --output, or WAV if unknown."
    ),
)
@click.option(
    "--sr",
    "sample_rate",
    default=None,
    type=int,
    help="Resample to this sample rate before processing (e.g. 44100, 48000).",
)
@click.option(
    "--source-bpm",
    "source_bpm_override",
    default=None,
    type=float,
    help="Manually specify the source BPM instead of auto-detecting it.",
)
@click.option(
    "--source-key",
    "source_key_override",
    default=None,
    type=str,
    help="Manually specify the source key instead of auto-detecting it.",
)
def main(
    input_path: str,
    output_path: str,
    target_bpm: float | None,
    target_key: str | None,
    output_format: str | None,
    sample_rate: int | None,
    source_bpm_override: float | None,
    source_key_override: str | None,
) -> None:
    """sample-forge — offline BPM and key retargeting for audio samples.

    Examples::

        # Stretch a loop from its native BPM to 140
        sample-forge -i break.wav -o break_140.wav --bpm 140

        # Transpose to A minor
        sample-forge -i riff.wav -o riff_Am.wav --key Am

        # Both at once, output as FLAC
        sample-forge -i loop.wav -o loop_out.flac --bpm 128 --key C --format FLAC
    """
    # Deferred imports so the CLI starts quickly and stubs import cleanly
    from sample_forge import analyzer, audio_io, key_utils, processor

    # ------------------------------------------------------------------
    # 1. Validate inputs early
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
                f"Error: '--source-key {source_key_override}' is not a valid key name.",
                err=True,
            )
            sys.exit(1)

    if target_bpm is not None and target_bpm <= 0:
        click.echo(
            f"Error: --bpm must be a positive number, got {target_bpm}.", err=True
        )
        sys.exit(1)

    if source_bpm_override is not None and source_bpm_override <= 0:
        click.echo(
            f"Error: --source-bpm must be a positive number, got {source_bpm_override}.",
            err=True,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Resolve output format
    # ------------------------------------------------------------------
    resolved_format = _resolve_output_format(output_path, output_format)

    # ------------------------------------------------------------------
    # 3. Load audio
    # ------------------------------------------------------------------
    click.echo(f"Loading {input_path} …")
    try:
        audio, sr = audio_io.load_audio(input_path, target_sr=sample_rate, mono=True)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        click.echo(f"Error loading audio: {exc}", err=True)
        sys.exit(1)

    click.echo(f"  Loaded {len(audio) / sr:.2f} s  |  {sr} Hz  |  mono")

    # ------------------------------------------------------------------
    # 4. Analyse source (if needed)
    # ------------------------------------------------------------------
    source_bpm: float | None = None
    source_key: str | None = None

    need_bpm = target_bpm is not None
    need_key = target_key is not None

    if need_bpm:
        if source_bpm_override is not None:
            source_bpm = source_bpm_override
            click.echo(f"  Source BPM (manual): {source_bpm:.2f}")
        else:
            click.echo("  Detecting BPM …")
            try:
                source_bpm = analyzer.detect_bpm(audio, sr)
            except (ValueError, RuntimeError) as exc:
                click.echo(f"Error during BPM detection: {exc}", err=True)
                sys.exit(1)
            click.echo(f"  Detected BPM: {source_bpm:.2f}")

    if need_key:
        if source_key_override is not None:
            source_key = source_key_override
            click.echo(f"  Source key (manual): {source_key}")
        else:
            click.echo("  Estimating key …")
            try:
                pc, mode = analyzer.estimate_key(audio, sr)
            except (ValueError, RuntimeError) as exc:
                click.echo(f"Error during key estimation: {exc}", err=True)
                sys.exit(1)
            source_key = key_utils.pitch_class_to_name(pc) + ("m" if mode == "minor" else "")
            click.echo(f"  Estimated key: {source_key} ({mode})")

    # ------------------------------------------------------------------
    # 5. Process
    # ------------------------------------------------------------------
    if not need_bpm and not need_key:
        click.echo(
            "Warning: neither --bpm nor --key specified; output will be identical to input.",
            err=True,
        )

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
    except (ValueError, RuntimeError) as exc:
        click.echo(f"Error during processing: {exc}", err=True)
        sys.exit(1)

    click.echo(f"  Output duration: {len(processed) / sr:.2f} s")

    # ------------------------------------------------------------------
    # 6. Write output
    # ------------------------------------------------------------------
    click.echo(f"Writing {output_path} ({resolved_format}) …")
    try:
        audio_io.write_audio(output_path, processed, sr, fmt=resolved_format)
    except (ValueError, OSError, RuntimeError) as exc:
        click.echo(f"Error writing output: {exc}", err=True)
        sys.exit(1)

    click.echo("Done.")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _resolve_output_format(output_path: str, explicit_format: str | None) -> str:
    """Determine the output format string from the path extension or explicit flag.

    Args:
        output_path: The destination file path.
        explicit_format: User-supplied format string (may be ``None``).

    Returns:
        An uppercase format string, either ``'WAV'`` or ``'FLAC'``.
    """
    if explicit_format is not None:
        return explicit_format.upper()

    suffix = Path(output_path).suffix.lower()
    if suffix == ".flac":
        return "FLAC"
    # Default to WAV for everything else
    return "WAV"
