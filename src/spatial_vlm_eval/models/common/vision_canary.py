"""Deterministic semantic checks proving that a model reads one image."""

from __future__ import annotations

import re

from PIL import Image, ImageDraw

VISION_CANARY_PROTOCOL = (
    "msmu_semantic_vision_canary_red_circle_top_left_blue_square_"
    "bottom_right_quadrant_prompt_words_or_normalized_bbox_antialiased512_v4"
)
VISION_CANARY_QUESTION = (
    "Identify every colored shape in this image. For each one, state its color, shape, "
    "and which image quadrant or corner area contains it. Do not answer only with a "
    "relative relation between the shapes. Answer concisely in English."
)
VISION_CANARY_IMAGE_SIZE = (512, 512)
VISION_CANARY_SUPERSAMPLE = 4
RED_CIRCLE_BOX = (48, 48, 208, 208)
BLUE_SQUARE_BOX = (304, 304, 464, 464)
CVBENCH_COLOR_CANARY_PROTOCOL = (
    "cvbench_minimum_vision_receipt_solid_red_blue_or_stricter_rgb512_v1"
)
COLOR_CANARY_QUESTION = (
    "This image is filled with one solid color. What color is the image? "
    "Answer concisely in English."
)


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


def make_solid_color_canary(color: str) -> Image.Image:
    """Return one of the two CV-Bench minimum-capability color canaries."""

    pixels = {"red": (255, 0, 0), "blue": (0, 0, 255)}
    if color not in pixels:
        raise ValueError(f"Unsupported solid-color canary: {color!r}")
    return Image.new("RGB", VISION_CANARY_IMAGE_SIZE, pixels[color])


def validate_solid_color_canary_answer(answer: str, expected_color: str) -> None:
    """Require explicit evidence of the expected color; nearby color terms are allowed."""

    if expected_color not in {"red", "blue"}:
        raise ValueError(f"Unsupported expected canary color: {expected_color!r}")
    normalized = _normalized_answer(answer)
    colors = set(
        re.findall(
            r"\b(?:red|blue|green|yellow|orange|purple|violet|pink|brown|black|white|gray|grey)\b",
            normalized,
        )
    )
    if expected_color not in colors:
        raise ValueError(
            f"Solid-color vision canary requires evidence of {expected_color}: {answer!r}"
        )


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


def _normalized_bbox_matches(answer: str) -> list[tuple[float, tuple[float, float]]]:
    number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
    pattern = re.compile(
        rf"\[\s*({number})\s*,\s*({number})\s*,\s*({number})\s*,\s*({number})\s*\]"
    )
    boxes: list[tuple[float, tuple[float, float]]] = []
    for match in pattern.finditer(answer):
        x1, y1, x2, y2 = (float(value) for value in match.groups())
        if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
            continue
        boxes.append(
            (
                (match.start() + match.end()) / 2,
                ((x1 + x2) / 2, (y1 + y2) / 2),
            )
        )
    return boxes


def _bbox_locations_are_correct(
    red_object_center: float,
    blue_object_center: float,
    boxes: list[tuple[float, tuple[float, float]]],
) -> bool:
    _, red_box, blue_box = min(
        (
            abs(red_object_center - boxes[red_index][0])
            + abs(blue_object_center - boxes[blue_index][0]),
            boxes[red_index][1],
            boxes[blue_index][1],
        )
        for red_index in range(len(boxes))
        for blue_index in range(len(boxes))
        if red_index != blue_index
    )
    red_x, red_y = red_box
    blue_x, blue_y = blue_box
    return red_x < 0.5 and red_y < 0.5 and blue_x > 0.5 and blue_y > 0.5


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
    missing_objects = [name for name in ("red_circle", "blue_square") if not matches[name]]
    if missing_objects:
        raise ValueError(
            "Vision canary answer is missing required color/shape semantics "
            f"{missing_objects}: {answer!r}"
        )

    red_center = matches["red_circle"][0]
    blue_center = matches["blue_square"][0]
    has_verbal_locations = bool(matches["top_left"] and matches["bottom_right"])
    if has_verbal_locations:
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

    boxes = _normalized_bbox_matches(normalized)
    has_bbox_locations = len(boxes) >= 2
    if has_bbox_locations and not _bbox_locations_are_correct(red_center, blue_center, boxes):
        raise ValueError(
            "Vision canary normalized boxes did not place the red circle in top-left and the "
            f"blue square in bottom-right: {answer!r}"
        )
    if not has_verbal_locations and not has_bbox_locations:
        raise ValueError(
            "Vision canary answer is missing complete top-left/bottom-right words or two valid "
            f"normalized bounding boxes: {answer!r}"
        )
