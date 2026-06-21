"""Map a target display resolution to a StreamDiffusion generation size.

TensorRT engines are static-shape: an engine only accepts the exact
``width x height`` it was built for, and the engine cache is keyed by that size
(see ``Config.resolved_engine_dir``). The installer asks the user for their
display resolution and must pick the generation size the engine is built/fetched
for; the running app must agree on that size. Both sides call this module so they
can never disagree.

Convention: generation height is fixed at 512; generation width is the
aspect-scaled value snapped to a multiple of 64 (a TensorRT/UNet requirement) and
clamped to the tested range. Example: a 3440x1440 (21:9) display -> 1216x512.
"""

from __future__ import annotations

GEN_HEIGHT = 512
WIDTH_MULTIPLE = 64
MIN_WIDTH = 320
MAX_WIDTH = 1216


def snap_to_multiple(value: float, multiple: int = WIDTH_MULTIPLE) -> int:
    """Round ``value`` to the nearest positive multiple of ``multiple``."""
    if multiple <= 0:
        raise ValueError("multiple must be positive")
    snapped = int(round(value / multiple)) * multiple
    return max(multiple, snapped)


def generation_size_for_display(
    display_width: int,
    display_height: int,
    *,
    gen_height: int = GEN_HEIGHT,
    min_width: int = MIN_WIDTH,
    max_width: int = MAX_WIDTH,
) -> tuple[int, int]:
    """Return the ``(width, height)`` to generate at for a given display size.

    The aspect ratio of the display is preserved as closely as possible while
    keeping ``height == gen_height`` and ``width`` a multiple of 64 within
    ``[min_width, max_width]``.
    """
    if display_width <= 0 or display_height <= 0:
        raise ValueError("display dimensions must be positive")

    aspect = display_width / display_height
    raw_width = gen_height * aspect
    width = snap_to_multiple(raw_width)
    width = max(min_width, min(max_width, width))
    return width, gen_height


def resolution_tag(width: int, height: int) -> str:
    """Engine-cache resolution key, matching ``Config.resolved_engine_dir``."""
    return f"{width}x{height}"


if __name__ == "__main__":  # pragma: no cover - manual sanity check
    samples = [
        (3440, 1440),
        (2560, 1080),
        (1920, 1080),
        (2560, 1440),
        (3840, 2160),
        (1280, 1024),
        (1080, 1920),
    ]
    for dw, dh in samples:
        w, h = generation_size_for_display(dw, dh)
        print(f"{dw}x{dh:<5} -> {resolution_tag(w, h)}  (aspect {dw / dh:.3f})")
