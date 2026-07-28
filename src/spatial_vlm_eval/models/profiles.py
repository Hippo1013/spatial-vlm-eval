"""Locked MSMU inference profiles, independent from the scorer protocol."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class InferenceProfile:
    key: str
    family: str
    model: str
    revision: str
    inference_protocol: str
    input_profile: str
    max_new_tokens: int
    temperature: float | None
    reasoning_effort: str | None
    chat_template: str
    default_tensor_parallel_size: int = 1
    served_model_name: str | None = None
    deployable_on_two_a800_80gb: bool = True
    upstream_url: str | None = None
    upstream_commit: str | None = None


def _profile(
    key: str,
    family: str,
    model: str,
    revision: str,
    input_profile: str,
    *,
    max_new_tokens: int = 192,
    temperature: float | None = 0.0,
    reasoning_effort: str | None = None,
    chat_template: str,
    tp: int = 1,
    served_model_name: str | None = None,
    deployable: bool = True,
    upstream_url: str | None = None,
    upstream_commit: str | None = None,
    inference_protocol: str | None = None,
) -> InferenceProfile:
    return InferenceProfile(
        key=key,
        family=family,
        model=model,
        revision=revision,
        inference_protocol=inference_protocol or f"msmu_{key}_{input_profile}_v1",
        input_profile=input_profile,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        chat_template=chat_template,
        default_tensor_parallel_size=tp,
        served_model_name=served_model_name,
        deployable_on_two_a800_80gb=deployable,
        upstream_url=upstream_url,
        upstream_commit=upstream_commit,
    )


PROFILES = {
    profile.key: profile
    for profile in [
        _profile(
            "gpt5",
            "closed",
            "openai/gpt-5",
            "provider-managed:gpt-5",
            "question_only",
            temperature=None,
            reasoning_effort="low",
            chat_template="openai-compatible single user multimodal message",
        ),
        _profile(
            "gemini31pro",
            "closed",
            "google/gemini-3.1-pro-preview",
            "provider-managed:gemini-3.1-pro-preview",
            "question_only",
            temperature=0.0,
            reasoning_effort="low",
            chat_template="openai-compatible single user multimodal message",
        ),
        _profile(
            "llava_next_mistral_7b",
            "llava_next",
            "llava-hf/llava-v1.6-mistral-7b-hf",
            "2424fdd47412fccc66d91719126b420e9fbd7065",
            "question_only",
            chat_template="checkpoint native processor chat template",
            served_model_name="llava-next-mistral-7b-msmu",
        ),
        _profile(
            "llava_next_yi_34b",
            "llava_next",
            "llava-hf/llava-v1.6-34b-hf",
            "84e4488fffae48f9da316ec31288b7c03f102ec7",
            "question_only",
            chat_template="checkpoint native processor chat template",
            tp=2,
            served_model_name="llava-next-yi-34b-msmu",
        ),
        _profile(
            "internvl3_8b",
            "internvl3",
            "OpenGVLab/InternVL3-8B-hf",
            "259a3b64a14623c0ec91a045cb43f7c5af5fa6af",
            "question_only",
            chat_template="checkpoint native processor chat template",
            served_model_name="internvl3-8b-msmu",
        ),
        _profile(
            "internvl3_38b",
            "internvl3",
            "OpenGVLab/InternVL3-38B-hf",
            "b2a05c0c325235f7530d8274c313a1d01082e069",
            "question_only",
            chat_template="checkpoint native processor chat template",
            tp=2,
            served_model_name="internvl3-38b-msmu",
        ),
        _profile(
            "internvl3_78b",
            "internvl3",
            "OpenGVLab/InternVL3-78B-hf",
            "3aecc2b26fd0ea29ea9f41e0ecaf877a1351f356",
            "question_only",
            chat_template="checkpoint native processor chat template",
            tp=2,
            served_model_name="internvl3-78b-msmu",
            deployable=False,
        ),
        _profile(
            "qwen25_vl_7b",
            "qwen25_vl",
            "Qwen/Qwen2.5-VL-7B-Instruct",
            "cc594898137f460bfe9f0759e9844b3ce807cfb5",
            "question_only",
            chat_template="Qwen2.5-VL native structured image chat template",
            inference_protocol="msmu_qwen25_vl_question_only_deterministic_v1",
        ),
        _profile(
            "qwen25_vl_32b",
            "qwen25_vl",
            "Qwen/Qwen2.5-VL-32B-Instruct",
            "7cfb30d71a1f4f49a57592323337a4a4727301da",
            "question_only",
            chat_template="Qwen2.5-VL native structured image chat template",
            inference_protocol="msmu_qwen25_vl_32b_question_only_deterministic_v1",
        ),
        _profile(
            "qwen25_vl_72b",
            "qwen25_vl",
            "Qwen/Qwen2.5-VL-72B-Instruct",
            "89c86200743eec961a297729e7990e8f2ddbc4c5",
            "question_only",
            chat_template="Qwen2.5-VL native structured image chat template",
            tp=2,
            inference_protocol="msmu_qwen25_vl_72b_question_only_deterministic_v1",
        ),
        _profile(
            "ssr",
            "ssr",
            "yliu-cs/SSR-VLM-7B",
            "7bcb4636f1396325f27f7fbb2f2df121128931bf",
            "rgb_only",
            chat_template="Qwen2.5-VL native structured image chat template",
            upstream_url="https://github.com/yliu-cs/SSR",
            upstream_commit="52a21a14a84a98f07575721dd3200f76c11930d8",
        ),
        _profile(
            "ssr_native",
            "ssr",
            "yliu-cs/SSR-VLM-7B + yliu-cs/SSR-MIDI-7B",
            "7bcb4636f1396325f27f7fbb2f2df121128931bf+8ed878fa16e3e440741ed8c1fedfcfe40710258d",
            "depthpro_midi_tor10_native",
            chat_template="Qwen2.5-VL native structured image chat template with 10 TOR tokens",
            upstream_url="https://github.com/yliu-cs/SSR",
            upstream_commit="52a21a14a84a98f07575721dd3200f76c11930d8",
        ),
        _profile(
            "spatialrgpt",
            "spatialrgpt",
            "a8cheng/SpatialRGPT-VILA1.5-8B",
            "64df7902f82b5053f5a53455095805e6de3a1f87",
            "rgb_only",
            chat_template="SpatialRGPT VILA llama_3 conversation",
            upstream_url="https://github.com/AnjieCheng/SpatialRGPT",
            upstream_commit="16715d4f1419997da18926c6ce574802d1eb3a37",
        ),
        _profile(
            "3dthinker",
            "3dthinker",
            "jankin123/3DThinker-Mindcube",
            "69a70411605f86ec69bada0a625bb96ddee995d9",
            "question_only",
            chat_template="3DThinker modified Qwen2.5-VL processor",
            upstream_url="https://github.com/zhangquanchen/3DThinker",
            upstream_commit="c9469e01b719310b0eaecc1133317e4ecfc74d8c",
        ),
        _profile(
            "3dthinker_native",
            "3dthinker",
            "jankin123/3DThinker-Mindcube",
            "69a70411605f86ec69bada0a625bb96ddee995d9",
            "mental3d_native",
            max_new_tokens=2048,
            chat_template="3DThinker modified Qwen2.5-VL processor plus official mental-3D control prompt",
            upstream_url="https://github.com/zhangquanchen/3DThinker",
            upstream_commit="c9469e01b719310b0eaecc1133317e4ecfc74d8c",
        ),
        _profile(
            "spatialbot",
            "spatialbot",
            "RussRobin/SpatialBot-3B",
            "41d3b52c642058dfb087885bec0b8e37e0e67f8d",
            "rgb_only",
            chat_template="SpatialBot Bunny conversation",
            upstream_url="https://github.com/BAAI-DCAI/SpatialBot",
            upstream_commit="775ad8cf2f9251261dcd70b2639133d506ff583f",
        ),
        _profile(
            "spatialbot_native",
            "spatialbot",
            "RussRobin/SpatialBot-3B",
            "41d3b52c642058dfb087885bec0b8e37e0e67f8d",
            "zoedepth_rgbd_native",
            chat_template="SpatialBot Bunny two-image RGB-D conversation",
            upstream_url="https://github.com/BAAI-DCAI/SpatialBot",
            upstream_commit="775ad8cf2f9251261dcd70b2639133d506ff583f",
        ),
    ]
}


def get_profile(key: str) -> InferenceProfile:
    try:
        return PROFILES[str(key)]
    except KeyError as exc:
        raise ValueError(f"Unknown inference profile {key!r}; choose from {sorted(PROFILES)}") from exc


def profile_keys(*families: str) -> list[str]:
    allowed = set(families)
    return sorted(key for key, profile in PROFILES.items() if profile.family in allowed)
