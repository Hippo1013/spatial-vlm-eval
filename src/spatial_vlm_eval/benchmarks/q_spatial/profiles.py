"""Locked inference-profile registry for the 21 Q-Spatial target tracks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class QSpatialProfile:
    key: str
    display_name: str
    group: str
    family: str
    model: str
    revision: str
    input_profile: str
    comparison_group: str
    inference_protocol: str
    adapter_kind: str
    default_backend: str
    chat_template: str
    system_role_supported: bool
    image_processing: dict[str, Any]
    decoding: dict[str, Any]
    seed_strategy: str
    provider_nondeterministic: bool
    model_path_env: str
    served_model_name: str | None = None
    default_tensor_parallel_size: int = 1
    min_free_gpu_mib: int = 30_000
    upstream_url: str | None = None
    upstream_commit: str | None = None
    processor_family: str | None = None
    api_policy_key: str | None = None
    default_workers: int = 1
    requires_runtime_generation_manifest: bool = False
    prompt_prefix: str | None = None
    known_deviation: str | None = None
    llava_two_stage: bool = False

    @property
    def registry_digest(self) -> str:
        return hashlib.sha256(
            json.dumps(
                {field: getattr(self, field) for field in self.__dataclass_fields__},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    @property
    def is_rgb_only(self) -> bool:
        return self.comparison_group == "rgb_only"


def _general(
    key: str,
    display_name: str,
    family: str,
    model: str,
    revision: str,
    *,
    model_path_env: str,
    served_model_name: str,
    processor_family: str,
    tp: int = 1,
    min_free_gpu_mib: int = 30_000,
    system_role_supported: bool = True,
    llava_two_stage: bool = False,
    qwen_sampling: bool = False,
) -> QSpatialProfile:
    decoding = (
        {
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "presence_penalty": 1.5,
            "num_beams": 1,
            "max_new_tokens": 1024,
            "seed": 3407,
        }
        if qwen_sampling
        else {
            "do_sample": False,
            "temperature": 0.0,
            "top_p": None,
            "num_beams": 1,
            "max_new_tokens": 512,
            "seed": 42,
            **({"second_stage_max_new_tokens": 64} if llava_two_stage else {}),
        }
    )
    decode_name = "qwen_recommended_sampling" if qwen_sampling else "greedy"
    if llava_two_stage:
        decode_name = "official_llava_two_stage_format_repair"
    return QSpatialProfile(
        key=key,
        display_name=display_name,
        group="general_open",
        family=family,
        model=model,
        revision=revision,
        input_profile="rgb",
        comparison_group="rgb_only",
        inference_protocol=f"q_spatial_{key}_standard_prompt_{decode_name}_per_request_seed_v1",
        adapter_kind="openai_compatible",
        default_backend="vllm",
        chat_template="checkpoint native Transformers processor template audited before vLLM",
        system_role_supported=system_role_supported,
        image_processing={"source": "Q-Spatial RGB", "image_count": 1},
        decoding=decoding,
        seed_strategy="per_request_fixed_base_seed",
        provider_nondeterministic=False,
        model_path_env=model_path_env,
        served_model_name=served_model_name,
        default_tensor_parallel_size=tp,
        min_free_gpu_mib=min_free_gpu_mib,
        processor_family=processor_family,
        llava_two_stage=llava_two_stage,
    )


def _specialized(
    key: str,
    display_name: str,
    family: str,
    model: str,
    revision: str,
    input_profile: str,
    comparison_group: str,
    decoding: dict[str, Any],
    *,
    model_path_env: str,
    upstream_url: str,
    upstream_commit: str,
    image_processing: dict[str, Any],
    chat_template: str,
    requires_manifest: bool = False,
    known_deviation: str | None = None,
) -> QSpatialProfile:
    return QSpatialProfile(
        key=key,
        display_name=display_name,
        group="specialized",
        family=family,
        model=model,
        revision=revision,
        input_profile=input_profile,
        comparison_group=comparison_group,
        inference_protocol=(
            f"q_spatial_{key}_{input_profile}_standard_prompt_folded_user_"
            "upstream_locked_per_request_seed_v1"
        ),
        adapter_kind="upstream_command",
        default_backend="upstream_transformers",
        chat_template=chat_template,
        system_role_supported=False,
        image_processing=image_processing,
        decoding=decoding,
        seed_strategy="per_request_fixed_base_seed",
        provider_nondeterministic=False,
        model_path_env=model_path_env,
        upstream_url=upstream_url,
        upstream_commit=upstream_commit,
        requires_runtime_generation_manifest=requires_manifest,
        known_deviation=known_deviation,
    )


_GENERAL = [
    _general(
        "llava_next_mistral_7b",
        "LLaVA-NeXT-Mistral-7B",
        "llava_next",
        "llava-hf/llava-v1.6-mistral-7b-hf",
        "2424fdd47412fccc66d91719126b420e9fbd7065",
        model_path_env="LLAVA_MISTRAL_7B_MODEL",
        served_model_name="llava-next-mistral-7b-qspatial",
        processor_family="llava_next",
        system_role_supported=False,
        llava_two_stage=True,
    ),
    _general(
        "llava_next_yi_34b",
        "LLaVA-NeXT-Yi-34B",
        "llava_next",
        "llava-hf/llava-v1.6-34b-hf",
        "84e4488fffae48f9da316ec31288b7c03f102ec7",
        model_path_env="LLAVA_YI_34B_MODEL",
        served_model_name="llava-next-yi-34b-qspatial",
        processor_family="llava_next",
        tp=2,
        min_free_gpu_mib=60_000,
        system_role_supported=False,
        llava_two_stage=True,
    ),
    _general(
        "internvl3_8b",
        "InternVL3-8B",
        "internvl3",
        "OpenGVLab/InternVL3-8B-hf",
        "259a3b64a14623c0ec91a045cb43f7c5af5fa6af",
        model_path_env="INTERNVL3_8B_MODEL",
        served_model_name="internvl3-8b-qspatial",
        processor_family="internvl3",
    ),
    _general(
        "internvl3_38b",
        "InternVL3-38B",
        "internvl3",
        "OpenGVLab/InternVL3-38B-hf",
        "b2a05c0c325235f7530d8274c313a1d01082e069",
        model_path_env="INTERNVL3_38B_MODEL",
        served_model_name="internvl3-38b-qspatial",
        processor_family="internvl3",
        tp=2,
        min_free_gpu_mib=60_000,
    ),
    _general(
        "internvl3_78b",
        "InternVL3-78B",
        "internvl3",
        "OpenGVLab/InternVL3-78B-hf",
        "3aecc2b26fd0ea29ea9f41e0ecaf877a1351f356",
        model_path_env="INTERNVL3_78B_MODEL",
        served_model_name="internvl3-78b-qspatial",
        processor_family="internvl3",
        tp=4,
        min_free_gpu_mib=76_000,
    ),
    _general(
        "qwen3_vl_2b",
        "Qwen3-VL-2B",
        "qwen3_vl",
        "Qwen/Qwen3-VL-2B-Instruct",
        "89644892e4d85e24eaac8bacfd4f463576704203",
        model_path_env="QWEN3_VL_2B_MODEL",
        served_model_name="qwen3-vl-2b-qspatial",
        processor_family="qwen3_vl",
        qwen_sampling=True,
    ),
    _general(
        "qwen3_vl_4b",
        "Qwen3-VL-4B",
        "qwen3_vl",
        "Qwen/Qwen3-VL-4B-Instruct",
        "ebb281ec70b05090aa6165b016eac8ec08e71b17",
        model_path_env="QWEN3_VL_4B_MODEL",
        served_model_name="qwen3-vl-4b-qspatial",
        processor_family="qwen3_vl",
        qwen_sampling=True,
    ),
    _general(
        "qwen3_vl_8b",
        "Qwen3-VL-8B",
        "qwen3_vl",
        "Qwen/Qwen3-VL-8B-Instruct",
        "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        model_path_env="QWEN3_VL_8B_MODEL",
        served_model_name="qwen3-vl-8b-qspatial",
        processor_family="qwen3_vl",
        qwen_sampling=True,
    ),
    _general(
        "qwen3_vl_32b",
        "Qwen3-VL-32B",
        "qwen3_vl",
        "Qwen/Qwen3-VL-32B-Instruct",
        "0cfaf48183f594c314753d30a4c4974bc75f3ccb",
        model_path_env="QWEN3_VL_32B_MODEL",
        served_model_name="qwen3-vl-32b-qspatial",
        processor_family="qwen3_vl",
        tp=2,
        min_free_gpu_mib=60_000,
        qwen_sampling=True,
    ),
]

_CLOSED = [
    QSpatialProfile(
        key="gpt5_openrouter_non_zdr",
        display_name="GPT-5",
        group="closed_api",
        family="closed",
        model="openai/gpt-5-2025-08-07",
        revision="openrouter-canonical:openai/gpt-5-2025-08-07",
        input_profile="rgb",
        comparison_group="rgb_only",
        inference_protocol="q_spatial_gpt5_standard_prompt_openrouter_non_zdr_medium_16384_v1",
        adapter_kind="openai_compatible",
        default_backend="openrouter",
        chat_template="OpenRouter system + single-image user structured messages",
        system_role_supported=True,
        image_processing={"source": "Q-Spatial RGB", "image_count": 1},
        decoding={"temperature": None, "reasoning_effort": "medium", "max_new_tokens": 16384},
        seed_strategy="provider_nondeterministic",
        provider_nondeterministic=True,
        model_path_env="",
        api_policy_key="gpt5_openrouter_non_zdr",
        default_workers=8,
    ),
    QSpatialProfile(
        key="gemini31pro_openrouter_non_zdr",
        display_name="Gemini 3.1 Pro",
        group="closed_api",
        family="closed",
        model="google/gemini-3.1-pro-preview-20260219",
        revision="openrouter-canonical:google/gemini-3.1-pro-preview-20260219",
        input_profile="rgb",
        comparison_group="rgb_only",
        inference_protocol="q_spatial_gemini31pro_standard_prompt_openrouter_non_zdr_medium_16384_v1",
        adapter_kind="openai_compatible",
        default_backend="openrouter",
        chat_template="OpenRouter system + single-image user structured messages",
        system_role_supported=True,
        image_processing={"source": "Q-Spatial RGB", "image_count": 1},
        decoding={"temperature": 0.0, "reasoning_effort": "medium", "max_new_tokens": 16384},
        seed_strategy="provider_nondeterministic",
        provider_nondeterministic=True,
        model_path_env="",
        api_policy_key="gemini31pro_openrouter_non_zdr",
        default_workers=8,
    ),
]

_SPECIALIZED = [
    _specialized(
        "ssr_rgb", "SSR", "ssr", "yliu-cs/SSR-VLM-7B",
        "7bcb4636f1396325f27f7fbb2f2df121128931bf", "rgb", "rgb_only",
        {"do_sample": True, "temperature": 0.1, "top_p": 0.001, "top_k": 1,
         "repetition_penalty": 1.05, "max_new_tokens": 128, "seed": 42},
        model_path_env="SSR_VLM_MODEL", upstream_url="https://github.com/yliu-cs/SSR",
        upstream_commit="52a21a14a84a98f07575721dd3200f76c11930d8",
        image_processing={"source": "Q-Spatial RGB", "image_count": 1},
        chat_template="SSR official Qwen2.5-VL folded-user template", requires_manifest=True,
    ),
    _specialized(
        "ssr_native", "SSR", "ssr", "yliu-cs/SSR-VLM-7B + yliu-cs/SSR-MIDI-7B",
        "7bcb4636f1396325f27f7fbb2f2df121128931bf+8ed878fa16e3e440741ed8c1fedfcfe40710258d",
        "rgb_depthpro_midi_tor10", "rgb_derived_depth",
        {"do_sample": True, "temperature": 0.1, "top_p": 0.001, "top_k": 1,
         "repetition_penalty": 1.05, "max_new_tokens": 128, "seed": 42, "tor_count": 10},
        model_path_env="SSR_VLM_MODEL", upstream_url="https://github.com/yliu-cs/SSR",
        upstream_commit="52a21a14a84a98f07575721dd3200f76c11930d8",
        image_processing={"source": "Q-Spatial RGB", "image_count": 1, "derived_depth": "DepthPro",
                          "midi": True, "tor_count": 10},
        chat_template="SSR official Qwen2.5-VL folded-user template with TOR tokens",
        requires_manifest=True,
    ),
    _specialized(
        "spatialrgpt_rgb", "SpatialRGPT", "spatialrgpt", "a8cheng/SpatialRGPT-VILA1.5-8B",
        "64df7902f82b5053f5a53455095805e6de3a1f87", "rgb", "rgb_only",
        {"do_sample": False, "num_beams": 1, "max_new_tokens": 128, "seed": 42},
        model_path_env="SPATIALRGPT_MODEL", upstream_url="https://github.com/AnjieCheng/SpatialRGPT",
        upstream_commit="16715d4f1419997da18926c6ce574802d1eb3a37",
        image_processing={"source": "Q-Spatial RGB", "image_count": 1, "region_masks": None, "depth": None},
        chat_template="SpatialRGPT official VILA folded-user llama_3 template",
        known_deviation="No region, mask, or depth input is fabricated.",
    ),
    _specialized(
        "3dthinker_rgb", "3DThinker-Mindcube", "3dthinker", "jankin123/3DThinker-Mindcube",
        "69a70411605f86ec69bada0a625bb96ddee995d9", "rgb", "rgb_only",
        {"do_sample": True, "temperature": 0.7, "top_p": 0.9, "max_new_tokens": 2048, "seed": 42},
        model_path_env="THREEDTHINKER_MODEL", upstream_url="https://github.com/zhangquanchen/3DThinker",
        upstream_commit="c9469e01b719310b0eaecc1133317e4ecfc74d8c",
        image_processing={"source": "Q-Spatial RGB", "image_count": 1},
        chat_template="3DThinker official modified Qwen2.5-VL folded-user template",
        requires_manifest=True,
        known_deviation="The Mental-3D control prompt is intentionally disabled.",
    ),
    _specialized(
        "spatialbot_rgb", "SpatialBot-3B", "spatialbot", "RussRobin/SpatialBot-3B",
        "41d3b52c642058dfb087885bec0b8e37e0e67f8d", "rgb", "rgb_only",
        {"do_sample": False, "num_beams": 1, "max_new_tokens": 128, "seed": 42},
        model_path_env="SPATIALBOT_MODEL", upstream_url="https://github.com/BAAI-DCAI/SpatialBot",
        upstream_commit="775ad8cf2f9251261dcd70b2639133d506ff583f",
        image_processing={"source": "Q-Spatial RGB", "image_count": 1},
        chat_template="SpatialBot official Bunny folded-user template",
    ),
    _specialized(
        "spatialbot_zoedepth", "SpatialBot-3B", "spatialbot", "RussRobin/SpatialBot-3B",
        "41d3b52c642058dfb087885bec0b8e37e0e67f8d", "rgb_zoedepth", "rgb_derived_depth",
        {"do_sample": False, "num_beams": 1, "max_new_tokens": 128, "seed": 42},
        model_path_env="SPATIALBOT_MODEL", upstream_url="https://github.com/BAAI-DCAI/SpatialBot",
        upstream_commit="775ad8cf2f9251261dcd70b2639133d506ff583f",
        image_processing={"source": "Q-Spatial RGB", "image_count": 1, "derived_depth": "ZoeDepth"},
        chat_template="SpatialBot official Bunny RGB-D folded-user template",
    ),
    _specialized(
        "robobrain25_8b_nv_rgb", "RoboBrain2.5-8B-NV", "robobrain", "BAAI/RoboBrain2.5-8B-NV",
        "3d77a19a3ddd8616b3979e03de56096edfb12ff6", "rgb", "rgb_only",
        {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "max_new_tokens": 768, "seed": 42},
        model_path_env="ROBOBRAIN25_8B_NV_MODEL", upstream_url="https://github.com/FlagOpen/RoboBrain2.5",
        upstream_commit="af98c932aac9ff715d70da177088d7bb95573ff7",
        image_processing={"source": "Q-Spatial RGB", "image_count": 1},
        chat_template="RoboBrain2.5 official processor folded-user template",
    ),
    _specialized(
        "robobrain25_8b_mt_rgb", "RoboBrain2.5-8B-MT", "robobrain", "BAAI/RoboBrain2.5-8B-MT",
        "01145b89a0fe49f78f5d677d25af7351088d7c7d", "rgb", "rgb_only",
        {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "max_new_tokens": 768, "seed": 42},
        model_path_env="ROBOBRAIN25_8B_MT_MODEL", upstream_url="https://github.com/FlagOpen/RoboBrain2.5",
        upstream_commit="af98c932aac9ff715d70da177088d7bb95573ff7",
        image_processing={"source": "Q-Spatial RGB", "image_count": 1},
        chat_template="RoboBrain2.5 official processor folded-user template",
    ),
    _specialized(
        "hispatial3b_moge2_xyz", "HiSpatial-3B", "hispatial", "lhzzzzzy/HiSpatial-3B",
        "75a5e3d65351d7602c492aa91533f62b8a252604", "rgb_moge2_xyz", "rgb_derived_xyz",
        {"do_sample": False, "max_new_tokens": 100, "seed": 42},
        model_path_env="HISPATIAL_3B_MODEL", upstream_url="https://github.com/microsoft/HiSpatial",
        upstream_commit="9b0a5718ed0fb3b8bd9d9e0b36b6192bd3e99be1",
        image_processing={"source": "Q-Spatial RGB", "image_count": 1, "derived_xyz": "MoGe-2",
                          "derived_xyz_model": "Ruicheng/moge-2-vitl-normal",
                          "derived_xyz_revision": "b135031bae30b5ac2ae141a0e68717795ce38340",
                          "derived_xyz_upstream_commit": "925b8ed835a7a9cdb7578ba15c658a0afc969030"},
        chat_template="HiSpatial official predictor folded-user template",
        known_deviation="Both splits use MoGe-2 estimated XYZ; ScanNet ground-truth depth is forbidden.",
    ),
    _specialized(
        "spatialladder3b_rgb", "SpatialLadder-3B", "spatialladder", "hongxingli/SpatialLadder-3B",
        "0819c3adf8827a2ea6c0348d49a23503ecb1f428", "rgb", "rgb_only",
        {"do_sample": True, "temperature": 0.01, "top_p": 1.0, "repetition_penalty": 1.05,
         "max_new_tokens": 128, "seed": 42},
        model_path_env="SPATIALLADDER_3B_MODEL", upstream_url="https://github.com/ZJU-REAL/SpatialLadder",
        upstream_commit="7a0d2ee85c28728835300310a349a53a15967f2e",
        image_processing={"source": "Q-Spatial RGB", "image_count": 1,
                          "min_pixels": 16 * 28 * 28, "max_pixels": 512 * 28 * 28},
        chat_template="SpatialLadder official Qwen2.5-VL folded-user template",
        requires_manifest=True,
        known_deviation="The optional thinking prompt is intentionally disabled.",
    ),
]

PROFILE_SEQUENCE = tuple(profile.key for profile in [*_GENERAL, *_CLOSED, *_SPECIALIZED])
PROFILES = {profile.key: profile for profile in [*_GENERAL, *_CLOSED, *_SPECIALIZED]}
RGB_PROFILE_KEYS = tuple(key for key in PROFILE_SEQUENCE if PROFILES[key].is_rgb_only)

if len(PROFILE_SEQUENCE) != 21 or len(PROFILES) != 21:
    raise RuntimeError("Q-Spatial target registry must contain exactly 21 unique profiles")
if len(RGB_PROFILE_KEYS) != 18:
    raise RuntimeError("Q-Spatial registry must contain exactly 18 RGB-only profiles")
if any("mental" in key or "thinking" in key for key in PROFILE_SEQUENCE):
    raise RuntimeError("Q-Spatial registry must not contain Mental-3D or thinking-prompt tracks")


def get_profile(key: str) -> QSpatialProfile:
    try:
        return PROFILES[str(key)]
    except KeyError as exc:
        raise ValueError(f"Unknown Q-Spatial profile: {key!r}") from exc


def ordered_profiles(keys: list[str] | tuple[str, ...]) -> list[QSpatialProfile]:
    requested = {str(key) for key in keys}
    unknown = sorted(requested - set(PROFILES))
    if unknown:
        raise ValueError(f"Unknown Q-Spatial profiles: {unknown}")
    return [PROFILES[key] for key in PROFILE_SEQUENCE if key in requested]
