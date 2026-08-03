"""Deterministic semantic checks proving that a model reads one image."""

from __future__ import annotations

import re

from PIL import Image, ImageDraw

VISION_CANARY_PROTOCOL = (
    "msmu_semantic_vision_canary_red_circle_top_left_blue_square_"
    "bottom_right_antialiased512_v2"
)
VISION_CANARY_QUESTION = (
    "Describe every colored shape in this image and state where each one is located. "
    "Answer concisely in English."
)
VISION_CANARY_IMAGE_SIZE = (512, 512)
VISION_CANARY_SUPERSAMPLE = 4
RED_CIRCLE_BOX = (48, 48, 208, 208)
BLUE_SQUARE_BOX = (304, 304, 464, 464)


def make_vision_canary_image() -> Image.Image:
    """Return the canonical white image with two spatially separated shapes."""

    factor = VISION_CANARY_SUPERSAMPLE
    high_resolution_size = tuple(dimension * factor for dimension in VISION_CANARY_IMAGE_SIZE)
    image = Image.new("RGB", high_resolution_size, "white")
    drawing = ImageDraw.Draw(image)
    drawing.ellipse(tuple(coordinate * factor for coordinate in RED_CIRCLE_BOX), fill=(255, 0, 0))
    drawing.rectangle(
        tuple(coordinate * factor for coordinate in BLUE_SQUARE_BOX),
        fill=(0, 0, 255),
    )
    return image.resize(VISION_CANARY_IMAGE_SIZE, Image.Resampling.LANCZOS)


def _normalized_answer(answer: str) -> str:
    normalized = str(answer).strip().lower()
    normalized = normalized.replace("_", " ")
    normalized = re.sub(
        r"[\N{HYPHEN}\N{NON-BREAKING HYPHEN}\N{EN DASH}\N{EM DASH}-]+",
        " ",
        normalized,
    )
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _match_centers(pattern: str, answer: str) -> list[float]:
    return [(match.start() + match.end()) / 2 for match in re.finditer(pattern, answer)]


def validate_vision_canary_answer(answer: str) -> None:
    """Require the two expected color/shape/location associations.

    Minimum-distance one-to-one pairing rejects a response that merely mentions
    every keyword while swapping the objects' locations.
    """

    normalized = _normalized_answer(answer)
    patterns = {
        "red_circle": r"(?:\bred\b(?:\s+\w+){0,2}\s+\bcircle\b|\bcircle\b(?:\s+\w+){0,2}\s+\bred\b)",
        "blue_square": r"(?:\bblue\b(?:\s+\w+){0,2}\s+\bsquare\b|\bsquare\b(?:\s+\w+){0,2}\s+\bblue\b)",
        "top_left": r"(?:(?:\btop\b|\bupper\b)(?:\s+\w+){0,1}\s+\bleft\b|\bleft\b(?:\s+\w+){0,1}\s+(?:\btop\b|\bupper\b))",
        "bottom_right": r"(?:(?:\bbottom\b|\blower\b)(?:\s+\w+){0,1}\s+\bright\b|\bright\b(?:\s+\w+){0,1}\s+(?:\bbottom\b|\blower\b))",
    }
    matches = {name: _match_centers(pattern, normalized) for name, pattern in patterns.items()}
    missing = [name for name, centers in matches.items() if not centers]
    if missing:
        raise ValueError(
            "Vision canary answer is missing required color/shape/location semantics "
            f"{missing}: {answer!r}"
        )

    red_center = matches["red_circle"][0]
    blue_center = matches["blue_square"][0]
    correct_cost = min(
        abs(red_center - top_left) + abs(blue_center - bottom_right)
        for top_left in matches["top_left"]
        for bottom_right in matches["bottom_right"]
    )
    swapped_cost = min(
        abs(red_center - bottom_right) + abs(blue_center - top_left)
        for top_left in matches["top_left"]
        for bottom_right in matches["bottom_right"]
    )
    if correct_cost >= swapped_cost:
        raise ValueError(
            "Vision canary did not associate the red circle with top-left and the blue square "
            f"with bottom-right: {answer!r}"
        )
