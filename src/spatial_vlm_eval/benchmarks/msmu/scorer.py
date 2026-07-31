"""MSMU-Bench official-compatible scoring with a local OpenAI-style judge.

The task routing, numeric thresholds, and macro-8 aggregation follow SD-VLM's
published code.  This is intentionally *not* byte-for-byte official scoring:
the judge model is configurable, JSON-only output constraints are added, and
grounding is split into its coordinate and object subtypes to repair the
    official prompt's meters/object mismatch. Prediction validation is a
    mandatory preflight and runs before any judge request.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import sys
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from .data import official_type_for_raw_type
from .prediction_validation import validate_predictions, write_validation_report


OFFICIAL_QUANT_TYPES = {
    "scale_estimation",
    "absolute_distance",
    "count",
    "grounding",
    "refer_obj_estimation",
}
OFFICIAL_QUAL_TYPES = {"relative_position", "scale_compare", "existence"}
JUDGE_CACHE_PROTOCOL = (
    "sdvlm_official_compat_local_judge_v3_grounding_split_malformed_zero"
)
SCORER_PROTOCOL = (
    "sdvlm_official_compat_local_judge_v4_grounding_split_"
    "strict_quant_length_malformed_zero"
)
MALFORMED_JUDGE_ZERO_FALLBACK = "malformed_judge_response_zero"
MSMU_OFFICIAL_TEST_SIZE = 987


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument(
        "--validation-report",
        default=None,
        help="Defaults to <output-dir>/prediction_validation.json.",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:18080/v1")
    parser.add_argument("--model", default="msmu-judge")
    parser.add_argument("--api-key", default="local")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--retries", type=int, default=4)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Atomically replace a JSONL artifact with the rows from this run."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary_path.replace(path)


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Atomically replace a JSON artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def validated_predictions_for_scoring(
    prediction_path: str | Path,
    dataset_root: str | Path,
    validation_report_path: str | Path,
) -> list[dict[str, Any]]:
    """Run the mandatory full-split preflight used by the scorer.

    Empty predictions remain warnings and therefore do not block scoring. Any
    structural/provenance error aborts before the judge cache is read or a
    judge request is submitted.
    """

    predictions, report = validate_predictions(
        prediction_path,
        dataset_root,
        allow_subset=False,
    )
    write_validation_report(validation_report_path, report)
    for warning in report["warnings"]:
        print(f"[prediction-validation] warning: {warning}")
    if not report["passed"]:
        errors = "; ".join(report["errors"][:5])
        raise ValueError(
            "Prediction validation failed; no judge requests were made. "
            f"See {Path(validation_report_path).resolve()}. {errors}"
        )
    print(
        "[prediction-validation] passed: "
        f"{report['num_prediction_rows']} rows, "
        f"{len(report['warnings'])} warning(s)"
    )
    return predictions


def official_type(row: dict[str, Any]) -> str:
    """Derive scorer routing only from the dataset-owned raw type."""

    return official_type_for_raw_type(str(row.get("raw_type") or ""))


def evaluate_quan_dist_question(question: str, answer: str, pred: str) -> str:
    # Text copied from SD-VLM/evaluation/llm_generate.py with only the final
    # sentence tightened to make local models return parseable JSON objects.
    prompt = f"""
You should help me to evaluate the response given the question and the correct answer.
You need to convert the measurement of the correct answer and response to meters. The conversion factors are as follows: 1 inch = 0.0254 meters. 1 foot = 0.3048 meters. 1 centimeter (cm) = 0.01 meters.
You should output two floats in meters, one for the answer, and one for the response. If the answer or reponse contains more than one number for prediction, you should output the List contains the number.
The output should be in JSON format.


Example 1:
Question: How tall is the long brown table opposite the crossed table?
Answer: The height of the long brown table opposite the crossed table is 1.02 m.
Response: It is 2.17 meters wide.
"answer_in_meters": 1.02, "response_in_meters": 2.17

Example 2:
Question: what's the total number of chairs in the image?
Answer: 2.
Response: There are 2 chairs.
"answer_in_meters": 2, "response_in_meters": 2

Example 3:
Question: What is the size of the dark pillow?
Answer: The dark pillow is with the size of 0.8 m x 0.63 m x 0.55 m
Response: It is 35.9 inches wide.
"answer_in_meters": [0.78,0.63,0.55], "response_in_meters": 0.91

Example 4:
Question: The height of the bed is 0.81 m, what is the height of the table and the nightstand ?
Answer: Since the height of the bed is 0.81 m, i think the height of the table is 1.02 meters and the height of the nightstand is 0.93 meters.
Response: Since the height of the bed is 0.81 m, i think the height of the table is 1.36 meters and the height of the nightstand is 0.77 meters
"answer_in_meters": [1.02,0.93], "response_in_meters":[1.36,0.77]

Your Turn:
Question: {question}
Answer: {answer}
Response: {pred}

Return only a JSON object with keys "answer_in_meters" and "response_in_meters".
"""
    return prompt


def evaluate_qual_question(question: str, answer: str, pred: str) -> str:
    prompt = f"""
You should help me to evaluate the response given the question and the correct answer.
To mark a response, you should output a single integer between 0 and 1.
1 means that the response perfectly matches the answer.
0 means that the response is completely different from the answer.
The output should be in JSON format.

Example 1:
Question: Is the blue bed to the left of the curtain from the viewer's perspective ?
Answer: Indeed, the bed is to the left of the curtain.
Response: Yes, the blue bed is positioned on the left side of the curtain.
"your_mark": 1

Example 2:
Question: Between the wooden table and the black chair, which on is taller?
Answer: The wooden table is taller.
Response: The chair.
"your_mark": 0

Example 3:
Question: What is the tallest among the table, the chaird, and the curtain?
Answer: The tallest is the curtain.
Response: The curtain.
"your_mark": 1


Your Turn:
Question: {question}
Answer: {answer}
Response: {pred}

Return only a JSON object with key "your_mark".
    """
    return prompt


def grounding_subtype(question: str) -> str:
    text = str(question).lower()
    if re.search(r"\bcoordinates?\s+of\b", text):
        return "coordinate_of_object"
    return "object_at_coordinate"


def evaluate_grounding_coordinate_question(question: str, answer: str, pred: str) -> str:
    prompt = f"""
You should help me evaluate a visual grounding response.
The question asks for the normalized image coordinate of an object. Coordinates
are already normalized image coordinates, not meters. Extract only the requested
coordinate pair(s) from the correct answer and from the response.

Return JSON only with keys:
{{"answer_coordinates":[number,...],"response_coordinates":[number,...]}}

Examples:
Question: What is the coordinate of the black TV ?
Answer: (0.72,0.28).
Response: (0.64,0.13).
{{"answer_coordinates":[0.72,0.28],"response_coordinates":[0.64,0.13]}}

Question: What is the coordinate of the porcelain toilet ?
Answer: It is located at (0.58,0.41) in the image.
Response: The toilet is around (0.58, 0.26).
{{"answer_coordinates":[0.58,0.41],"response_coordinates":[0.58,0.26]}}

If either side does not contain usable coordinates, return an empty list for that side.

Your Turn:
Question: {question}
Answer: {answer}
Response: {pred}

Return only the JSON object.
"""
    return prompt


def evaluate_grounding_object_question(question: str, answer: str, pred: str) -> str:
    prompt = f"""
You should help me evaluate a visual grounding response.
The question gives a normalized image coordinate and asks which object is at that
coordinate. Compare the object named in the response with the object named in the
correct answer.

Mark 1 only if the response refers to the same object as the answer. Harmless
articles, punctuation, and small paraphrases are fine. A different color,
different object category, or different instance should be marked 0.

Return JSON only with keys:
{{"answer_object":"...","response_object":"...","your_mark":0_or_1}}

Examples:
Question: What object is located at (0.97,0.6)?
Answer: the brown cabinet.
Response: the brown wooden cabinet.
{{"answer_object":"brown cabinet","response_object":"brown wooden cabinet","your_mark":1}}

Question: What can you find at (0.31,0.96)?
Answer: It is the dark chair.
Response: the brown chair.
{{"answer_object":"dark chair","response_object":"brown chair","your_mark":0}}

Question: What object does the position (0.42,0.72) belong to?
Answer: the table.
Response: the table.
{{"answer_object":"table","response_object":"table","your_mark":1}}

Your Turn:
Question: {question}
Answer: {answer}
Response: {pred}

Return only the JSON object.
"""
    return prompt


def judge_prompt(row: dict[str, Any]) -> str:
    kind = official_type(row)
    if kind == "grounding":
        if grounding_subtype(row["question"]) == "coordinate_of_object":
            return evaluate_grounding_coordinate_question(row["question"], row["reference"], row["prediction"])
        return evaluate_grounding_object_question(row["question"], row["reference"], row["prediction"])
    if kind in OFFICIAL_QUANT_TYPES:
        return evaluate_quan_dist_question(row["question"], row["reference"], row["prediction"])
    if kind in OFFICIAL_QUAL_TYPES:
        return evaluate_qual_question(row["question"], row["reference"], row["prediction"])
    raise ValueError(f"Unknown official type: {kind}")


def cache_key(row: dict[str, Any], *, base_url: str, model: str) -> str:
    payload = {
        "protocol": JUDGE_CACHE_PROTOCOL,
        "official_type": official_type(row),
        "question": row["question"],
        "reference": row["reference"],
        "prediction": row["prediction"],
        "prompt": judge_prompt(row),
        # Never reuse responses produced by another judge model/endpoint.
        "judge_model": model,
        "judge_base_url": base_url.rstrip("/"),
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def extract_json(
    text: str,
    *,
    recovery_index: int | None = None,
) -> dict[str, Any]:
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        value = json.loads(stripped)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    try:
        value = ast.literal_eval(stripped)
        if isinstance(value, dict):
            return value
    except (SyntaxError, ValueError):
        pass
    match = re.search(r"\{.*\}", stripped, flags=re.S)
    if match:
        candidate = match.group(0)
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            try:
                value = ast.literal_eval(candidate)
                if isinstance(value, dict):
                    return value
            except (SyntaxError, ValueError):
                value = eval(candidate, {"__builtins__": {}}, {})  # noqa: S307 - official scorer uses eval.
                if isinstance(value, dict):
                    return value
    # Some local judges emit the requested qualitative mark as the first bare
    # JSON member, then add prose despite the JSON-only instruction. Recover
    # only that unambiguous first-line 0/1 mark; never search explanatory text
    # or accept other keys and values through this fallback.
    leading_mark = re.match(
        r'^\s*"your_mark"[ \t]*:[ \t]*(0|1)[ \t]*\r?\n',
        stripped,
    )
    if leading_mark and stripped[leading_mark.end() :].strip():
        mark = int(leading_mark.group(1))
        if recovery_index is not None:
            print(
                "[msmu-score] judge_parse_recovery "
                f"index={recovery_index} "
                "strategy=leading_bare_your_mark "
                f"your_mark={mark} trailing_text_ignored=true",
                file=sys.stderr,
                flush=True,
            )
        return {"your_mark": mark}
    # Official evaluate_final.py uses eval(data["answer"]). Support local models
    # that return `"key": value` without surrounding braces, as in the examples.
    bare = "{" + stripped.strip().strip(",") + "}"
    try:
        value = json.loads(bare)
    except json.JSONDecodeError:
        try:
            value = ast.literal_eval(bare)
        except (SyntaxError, ValueError):
            value = eval(bare, {"__builtins__": {}}, {})  # noqa: S307 - official scorer uses eval.
    if not isinstance(value, dict):
        raise ValueError("Judge JSON is not an object")
    return value


def judge_response_error(row: dict[str, Any], response: Any) -> str | None:
    """Return why a judge response is unusable, or ``None`` when usable.

    A response with the expected keys may still describe a model-answer
    extraction failure (for example an empty coordinate list). That is a valid
    judge response and remains scoreable as zero. Infrastructure, parsing, and
    response-schema failures are not scoreable and must be retried.
    """

    if not isinstance(response, dict):
        return "judge response is not a JSON object"
    if response.get("__score_fallback__") == MALFORMED_JUDGE_ZERO_FALLBACK:
        return None
    if "__parse_error__" in response:
        return f"judge request/parse failure: {response['__parse_error__']}"

    kind = official_type(row)
    if kind == "grounding":
        if grounding_subtype(row["question"]) == "coordinate_of_object":
            coordinate_keys = {"answer_coordinates", "response_coordinates"}
            legacy_keys = {"answer_in_meters", "response_in_meters"}
            if not (coordinate_keys <= set(response) or legacy_keys <= set(response)):
                return (
                    "grounding coordinate response lacks either "
                    "answer/response_coordinates or answer/response_in_meters"
                )
            return None
        required = {"answer_object", "response_object", "your_mark"}
    elif kind in OFFICIAL_QUANT_TYPES:
        required = {"answer_in_meters", "response_in_meters"}
    elif kind in OFFICIAL_QUAL_TYPES:
        required = {"your_mark"}
    else:
        return f"unknown official type: {kind}"

    missing = sorted(required - set(response))
    if missing:
        return f"judge response missing required keys: {missing}"
    if "your_mark" in required and response.get("your_mark") is None:
        return "judge response has null your_mark"
    return None


def cache_entry_is_usable(
    row: dict[str, Any],
    entry: Any,
    *,
    expected_cache_key: str,
) -> bool:
    """Return whether a cached response can safely skip a judge request."""

    if not isinstance(entry, dict):
        return False
    if entry.get("cache_key") != expected_cache_key:
        return False
    return judge_response_error(row, entry.get("judge")) is None


def judge_cache_candidate(
    row: dict[str, Any],
    result: Any,
    *,
    base_url: str,
    model: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Build either a successful cache row or a non-cacheable failure row."""

    judge_result = result
    if (
        isinstance(result, dict)
        and "__parse_error__" in result
        and result.get("__raw_content__") is not None
    ):
        judge_result = {
            **result,
            "__score_fallback__": MALFORMED_JUDGE_ZERO_FALLBACK,
        }

    error = judge_response_error(row, judge_result)
    if error is not None:
        return None, {
            "index": int(row["index"]),
            "error": error,
            "judge": judge_result,
        }
    return {
        "index": int(row["index"]),
        "cache_key": cache_key(row, base_url=base_url, model=model),
        "judge_model": model,
        "judge_base_url": base_url,
        "judge": judge_result,
    }, None


def publication_gate_status(
    *,
    validation_passed: bool,
    num_samples: int,
    indices: list[int],
    missing_official_types: list[str],
    judge_failure_count: int,
) -> tuple[bool, dict[str, bool], list[str]]:
    """Evaluate the machine-readable gates for a publishable summary."""

    checks = {
        "prediction_validation_passed": bool(validation_passed),
        "full_official_test_split": (
            num_samples == MSMU_OFFICIAL_TEST_SIZE
            and sorted(indices) == list(range(MSMU_OFFICIAL_TEST_SIZE))
        ),
        "all_official_types_present": not missing_official_types,
        "judge_failures_zero": judge_failure_count == 0,
    }
    failures = [name for name, passed in checks.items() if not passed]
    return not failures, checks, failures


def call_chat(row: dict[str, Any], base_url: str, model: str, api_key: str, retries: int) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": judge_prompt(row)}],
        "temperature": 0,
        "max_tokens": 256,
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    last_error: Exception | None = None
    last_content: str | None = None
    for attempt in range(retries):
        try:
            with opener.open(request, timeout=180) as response:
                result = json.loads(response.read().decode("utf-8"))
            last_content = str(result["choices"][0]["message"]["content"])
            parsed = extract_json(
                last_content,
                recovery_index=int(row["index"]),
            )
            schema_error = judge_response_error(row, parsed)
            if schema_error is not None:
                raise ValueError(schema_error)
            return parsed
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(2**attempt)
    return {
        "__parse_error__": str(last_error),
        "__raw_content__": last_content,
    }


_SENTENCE_MODEL = None


def sentence_similarity(a: str, b: str) -> float:
    global _SENTENCE_MODEL
    if _SENTENCE_MODEL is None:
        from sentence_transformers import SentenceTransformer, util

        _SENTENCE_MODEL = (SentenceTransformer("all-MiniLM-L6-v2"), util)
    model, util = _SENTENCE_MODEL
    embeddings = model.encode([a, b])
    return float(util.cos_sim(embeddings[0], embeddings[1]))


def as_float(value: Any) -> float:
    return float(value)


def ratio_delta(answer: float, response: float) -> float:
    if abs(answer) < 1e-12 and abs(response) < 1e-12:
        return 1.0
    if abs(answer) < 1e-12 or abs(response) < 1e-12:
        return math.inf
    return max(response / answer, answer / response)


def semantic_score(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value > 0.5)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "correct"}:
            return 1.0
        if normalized in {"false", "0", "no", "incorrect"}:
            return 0.0
    return 0.0


def normalize_object_name(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"\b(it is|that is|this is|there is|the object is)\b", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\b(the|a|an|object|at|position|coordinate|image)\b", " ", text)
    return " ".join(text.split())


def coordinate_values(value: Any) -> list[float]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    return [float(item) for item in values]


def official_grounding_score(question: str, judge: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    subtype = grounding_subtype(question)
    if subtype == "object_at_coordinate":
        try:
            answer = judge.get("answer_object", judge.get("answer_in_meters"))
            response = judge.get("response_object", judge.get("response_in_meters"))
            if "your_mark" in judge:
                score = semantic_score(judge["your_mark"])
            else:
                answer_norm = normalize_object_name(answer)
                response_norm = normalize_object_name(response)
                score = float(bool(answer_norm and response_norm and (answer_norm in response_norm or response_norm in answer_norm)))
            if answer is None or response is None:
                raise ValueError("grounding object extraction returned null")
            return score, {
                "match_success": True,
                "delta": 0,
                "grounding_subtype": subtype,
                "official_answer": answer,
                "official_response": response,
                "official_mark": judge.get("your_mark"),
            }
        except Exception as exc:  # noqa: BLE001
            return 0.0, {"match_success": False, "delta": None, "grounding_subtype": subtype, "error": str(exc)}

    try:
        answer = judge.get("answer_coordinates", judge.get("answer_in_meters"))
        response = judge.get("response_coordinates", judge.get("response_in_meters"))
        answer_list = coordinate_values(answer)
        response_list = coordinate_values(response)
        if not answer_list or len(answer_list) != len(response_list):
            raise ValueError("grounding coordinate lengths differ")
        diffs = [abs(r - a) for a, r in zip(answer_list, response_list)]
        ratio = sum(diffs) / len(diffs)
        error_rate = sum(abs(r - a) / (abs(a) + 1e-4) for a, r in zip(answer_list, response_list)) / len(diffs)
        return float(ratio <= 0.1), {
            "match_success": True,
            "delta": ratio,
            "error_rate": error_rate,
            "grounding_subtype": subtype,
            "official_answer": answer,
            "official_response": response,
        }
    except Exception as exc:  # noqa: BLE001
        return 0.0, {"match_success": False, "delta": None, "grounding_subtype": subtype, "error": str(exc)}


def official_quant_score(kind: str, judge: dict[str, Any], question: str = "") -> tuple[float, dict[str, Any]]:
    if kind == "grounding":
        if question:
            return official_grounding_score(question, judge)
        try:
            answer = judge["answer_in_meters"]
            response = judge["response_in_meters"]
            if isinstance(answer, str):
                score = float(sentence_similarity(answer, str(response)) >= 0.5)
                return score, {"match_success": True, "delta": 0, "official_answer": answer, "official_response": response}
            answer_list = list(answer) if isinstance(answer, list) else [answer]
            response_list = list(response) if isinstance(response, list) else [response]
            if len(answer_list) != len(response_list):
                raise ValueError("grounding coordinate lengths differ")
            diffs = [abs(as_float(r) - as_float(a)) for a, r in zip(answer_list, response_list)]
            ratio = sum(diffs) / len(diffs)
            error_rate = sum(abs(as_float(r) - as_float(a)) / (as_float(a) + 1e-4) for a, r in zip(answer_list, response_list)) / len(diffs)
            return float(ratio <= 0.1), {
                "match_success": True,
                "delta": ratio,
                "error_rate": error_rate,
                "official_answer": answer,
                "official_response": response,
            }
        except Exception as exc:  # noqa: BLE001
            return 0.0, {"match_success": False, "delta": None, "error": str(exc)}

    try:
        answer = judge["answer_in_meters"]
        response = judge["response_in_meters"]
        if isinstance(answer, list):
            if not isinstance(response, list):
                raise ValueError("official list answer with scalar response is a match failure")
            # SD-VLM's evaluate_final.py indexes response_in_meters once for
            # every reference value. A shorter response therefore raises and
            # is counted as a match failure; extra response values are ignored.
            if len(response) < len(answer):
                raise ValueError(
                    "official list response is shorter than the reference "
                    f"({len(response)} < {len(answer)})"
                )
            ratios = []
            error_rates = []
            for index, a in enumerate(answer):
                r = response[index]
                a_f = as_float(a)
                r_f = as_float(r)
                ratios.append(ratio_delta(a_f, r_f))
                error_rates.append(abs(r_f - a_f) / (abs(a_f) + 1e-4))
            ratio = sum(ratios) / len(ratios)
            error_rate = sum(error_rates) / len(error_rates)
        else:
            a_f = as_float(answer)
            r_f = as_float(response)
            ratio = ratio_delta(a_f, r_f)
            error_rate = abs(r_f - a_f) / (abs(a_f) + 1e-4)
        return float(ratio < 1.25), {
            "match_success": True,
            "delta": ratio,
            "error_rate": error_rate,
            "official_answer": answer,
            "official_response": response,
            "a1": float(ratio < 1.25),
            "a2": float(ratio < 1.25**2),
            "a3": float(ratio < 1.25**3),
        }
    except Exception as exc:  # noqa: BLE001
        return 0.0, {"match_success": False, "delta": None, "error": str(exc)}


def official_qual_score(judge: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    try:
        mark = judge["your_mark"]
        if mark is None:
            return 0.0, {"match_success": False, "official_mark": None}
        return float(float(mark) > 0.5), {"match_success": True, "official_mark": mark}
    except Exception as exc:  # noqa: BLE001
        return 0.0, {"match_success": False, "official_mark": "N/A", "error": str(exc)}


def score_judge_response(
    row: dict[str, Any],
    judge: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    """Score one schema-valid or protocol-approved zero-fallback response."""

    kind = official_type(row)
    if judge.get("__score_fallback__") == MALFORMED_JUDGE_ZERO_FALLBACK:
        details: dict[str, Any] = {
            "match_success": False,
            "delta": None,
            "judge_fallback": MALFORMED_JUDGE_ZERO_FALLBACK,
            "error": str(judge.get("__parse_error__") or "malformed judge response"),
        }
        if kind == "grounding":
            details["grounding_subtype"] = grounding_subtype(row["question"])
        if kind in OFFICIAL_QUAL_TYPES:
            details["official_mark"] = "N/A"
        return 0.0, details
    if kind in OFFICIAL_QUANT_TYPES:
        return official_quant_score(kind, judge, row["question"])
    if kind in OFFICIAL_QUAL_TYPES:
        return official_qual_score(judge)
    raise ValueError(f"Unknown official type: {kind}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_report_path = (
        Path(args.validation_report).resolve()
        if args.validation_report
        else output_dir / "prediction_validation.json"
    )
    try:
        predictions = validated_predictions_for_scoring(
            args.predictions,
            args.dataset_root,
            validation_report_path,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    cache_path = output_dir / "judge_cache.jsonl"
    judge_failures_path = output_dir / "judge_failures.jsonl"
    cached = {}
    if cache_path.exists():
        cached = {int(row["index"]): row for row in read_jsonl(cache_path)}

    pending = []
    for row in predictions:
        expected_cache_key = cache_key(row, base_url=args.base_url, model=args.model)
        if not cache_entry_is_usable(
            row,
            cached.get(int(row["index"])),
            expected_cache_key=expected_cache_key,
        ):
            pending.append(row)

    judge_failures: list[dict[str, Any]] = []
    if pending:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(call_chat, row, args.base_url, args.model, args.api_key, args.retries): row
                for row in pending
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Official-compatible local judge"):
                row = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    result = {"__parse_error__": f"unhandled judge worker failure: {exc}"}
                cached_row, failure = judge_cache_candidate(
                    row,
                    result,
                    base_url=args.base_url,
                    model=args.model,
                )
                if failure is not None:
                    judge_failures.append(failure)
                    continue
                assert cached_row is not None
                cached[int(row["index"])] = cached_row
                append_jsonl(cache_path, cached_row)

    unresolved_rows = []
    for row in predictions:
        expected_cache_key = cache_key(row, base_url=args.base_url, model=args.model)
        if not cache_entry_is_usable(
            row,
            cached.get(int(row["index"])),
            expected_cache_key=expected_cache_key,
        ):
            unresolved_rows.append(row)

    failure_by_index = {int(failure["index"]): failure for failure in judge_failures}
    unresolved_failures = [
        failure_by_index.get(
            int(row["index"]),
            {
                "index": int(row["index"]),
                "error": "no usable judge response is cached",
            },
        )
        for row in unresolved_rows
    ]
    write_jsonl(judge_failures_path, unresolved_failures)

    prediction_official_types = {official_type(row) for row in predictions}
    expected_official_types = OFFICIAL_QUANT_TYPES | OFFICIAL_QUAL_TYPES
    prediction_missing_types = sorted(expected_official_types - prediction_official_types)
    if unresolved_failures:
        publishable, publication_gates, publication_gate_failures = publication_gate_status(
            validation_passed=True,
            num_samples=len(predictions),
            indices=[int(row["index"]) for row in predictions],
            missing_official_types=prediction_missing_types,
            judge_failure_count=len(unresolved_failures),
        )
        failure_summary = {
            "publishable": publishable,
            "status": "blocked_by_judge_failures",
            "result_kind": "official-compatible internal score",
            "num_prediction_rows": len(predictions),
            "num_scored_samples": 0,
            "num_judge_failures": len(unresolved_failures),
            "judge_failure_indices": [
                int(failure["index"]) for failure in unresolved_failures
            ],
            "missing_official_types": prediction_missing_types,
            "publication_gates": publication_gates,
            "publication_gate_failures": publication_gate_failures,
            "judge_model": args.model,
            "judge_base_url": args.base_url,
            "protocol": SCORER_PROTOCOL,
            "judge_failure_report": str(judge_failures_path),
        }
        write_json(output_dir / "summary.json", failure_summary)
        print(json.dumps(failure_summary, ensure_ascii=False, indent=2))
        raise SystemExit(
            f"Scoring blocked: {len(unresolved_failures)} judge request/schema "
            f"failure(s); see {judge_failures_path}. Failed responses were not cached."
        )

    scored_rows: list[dict[str, Any]] = []
    family_scores: dict[str, list[float]] = defaultdict(list)
    official_type_scores: dict[str, list[float]] = defaultdict(list)
    match_successes = []
    malformed_judge_zero_indices = []
    for row in predictions:
        kind = official_type(row)
        judge = cached[int(row["index"])]["judge"]
        score, details = score_judge_response(row, judge)
        if kind in OFFICIAL_QUANT_TYPES:
            match_successes.append(float(details.get("match_success", False)))
        if details.get("judge_fallback") == MALFORMED_JUDGE_ZERO_FALLBACK:
            malformed_judge_zero_indices.append(int(row["index"]))
        family = str(row["task_family"])
        family_scores[family].append(score)
        official_type_scores[kind].append(score)
        scored_rows.append({**row, "official_type": kind, "score": score, "judge": judge, **details})

    official_summary = {
        kind: {"count": len(values), "accuracy": sum(values) / len(values)}
        for kind, values in sorted(official_type_scores.items())
    }
    family_summary = {
        family: {"count": len(values), "accuracy": sum(values) / len(values)}
        for family, values in sorted(family_scores.items())
    }
    missing = sorted(expected_official_types - set(official_summary))
    official_macro8 = None if missing else sum(official_summary[k]["accuracy"] for k in expected_official_types) / 8
    publishable, publication_gates, publication_gate_failures = publication_gate_status(
        validation_passed=True,
        num_samples=len(scored_rows),
        indices=[int(row["index"]) for row in scored_rows],
        missing_official_types=missing,
        judge_failure_count=0,
    )
    if not publishable:
        blocked_summary = {
            "publishable": False,
            "status": "blocked_by_publication_gates",
            "result_kind": "official-compatible internal score",
            "num_prediction_rows": len(predictions),
            "num_scored_samples": len(scored_rows),
            "num_judge_failures": 0,
            "missing_official_types": missing,
            "publication_gates": publication_gates,
            "publication_gate_failures": publication_gate_failures,
            "judge_model": args.model,
            "judge_base_url": args.base_url,
            "protocol": SCORER_PROTOCOL,
        }
        write_json(output_dir / "summary.json", blocked_summary)
        print(json.dumps(blocked_summary, ensure_ascii=False, indent=2))
        raise SystemExit(
            "Scoring completed but publication gates failed: "
            + ", ".join(publication_gate_failures)
        )

    summary = {
        "publishable": True,
        "status": "complete",
        "result_kind": "official-compatible internal score",
        "publication_gates": publication_gates,
        "publication_gate_failures": publication_gate_failures,
        "num_judge_failures": 0,
        "num_malformed_judge_zero_fallbacks": len(malformed_judge_zero_indices),
        "malformed_judge_zero_fallback_indices": malformed_judge_zero_indices,
        "num_samples": len(scored_rows),
        "micro_accuracy": sum(row["score"] for row in scored_rows) / len(scored_rows),
        "official_macro8_accuracy": official_macro8,
        "macro_8_task_accuracy": official_macro8,
        "missing_official_types": missing,
        "quantitative_match_success_rate": sum(match_successes) / len(match_successes) if match_successes else None,
        "judge_model": args.model,
        "judge_base_url": args.base_url,
        "protocol": SCORER_PROTOCOL,
        "official_types": official_summary,
        "families": family_summary,
        "notes": [
            "Matches SD-VLM quantitative/qualitative type split, success thresholds, and macro-8 aggregation.",
            "Uses a configurable local OpenAI-compatible judge instead of GPT-4-Turbo.",
            "Quantitative/qualitative prompts add JSON-only constraints and judge temperature is fixed to zero.",
            "Grounding uses subtype-specific prompts; object-at-coordinate uses judge semantic marking rather than official MiniLM cosine similarity.",
            "For non-grounding numeric lists, a response shorter than the reference is a match failure; extra trailing response values are ignored, matching SD-VLM evaluate_final.py.",
            "After retries, a returned judge response that remains unparsable or schema-invalid is cached and scored as zero; transport failures with no judge content still block publication.",
            "This protocol must not be described as differing from official evaluation only by judge model.",
        ],
    }
    with (output_dir / "scored_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in scored_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
