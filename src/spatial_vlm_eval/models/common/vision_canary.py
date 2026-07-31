"""Deterministic semantic checks proving that a model distinguishes images."""

from __future__ import annotations

SOLID_COLOR_QUESTION = (
    "Name the single solid color filling this image. Answer with one English color word."
)


def validate_solid_color_answers(red_answer: str, blue_answer: str) -> None:
    red = str(red_answer).strip().lower()
    blue = str(blue_answer).strip().lower()
    if "red" not in red:
        raise ValueError(f"Vision canary did not identify the red image: {red_answer!r}")
    if "blue" not in blue:
        raise ValueError(f"Vision canary did not identify the blue image: {blue_answer!r}")
    if red == blue:
        raise ValueError("Vision canary returned identical responses for red and blue images")
