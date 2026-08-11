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
    do_sample: bool = False
    top_p: float | None = None
    repetition_penalty: float | None = None
    num_beams: int = 1
    use_cache: bool = True
    seed: int = 42


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
    do_sample: bool = False,
    top_p: float | None = None,
    repetition_penalty: float | None = None,
    num_beams: int = 1,
    use_cache: bool = True,
    seed: int = 42,
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
        do_sample=do_sample,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        num_beams=num_beams,
        use_cache=use_cache,
        seed=seed,
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
            "gpt5_openrouter_non_zdr",
            "closed",
            "openai/gpt-5",
            "openrouter-canonical:openai/gpt-5-2025-08-07",
            "question_only",
            max_new_tokens=16384,
            temperature=None,
            reasoning_effort="medium",
            chat_template="openai-compatible single user multimodal message",
            inference_protocol="msmu_gpt5_question_only_openrouter_non_zdr_v3_medium_16384",
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
            "gemini31pro_openrouter_non_zdr",
            "closed",
            "google/gemini-3.1-pro-preview",
            "openrouter-canonical:google/gemini-3.1-pro-preview-20260219",
            "question_only",
            max_new_tokens=16384,
            temperature=0.0,
            reasoning_effort="medium",
            chat_template="openai-compatible single user multimodal message",
            inference_protocol="msmu_gemini31pro_question_only_openrouter_non_zdr_v3_medium_16384",
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
            tp=4,
            served_model_name="internvl3-78b-msmu",
            # Four 80GB GPUs are supported; two A800 80GB GPUs remain insufficient.
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
            "qwen3_vl_2b",
            "qwen3_vl",
            "Qwen/Qwen3-VL-2B-Instruct",
            "89644892e4d85e24eaac8bacfd4f463576704203",
            "question_only",
            chat_template="Qwen3-VL native structured image chat template; no system message",
            inference_protocol="msmu_qwen3_vl_2b_question_only_deterministic_v1",
        ),
        _profile(
            "qwen3_vl_4b",
            "qwen3_vl",
            "Qwen/Qwen3-VL-4B-Instruct",
            "ebb281ec70b05090aa6165b016eac8ec08e71b17",
            "question_only",
            chat_template="Qwen3-VL native structured image chat template; no system message",
            inference_protocol="msmu_qwen3_vl_4b_question_only_deterministic_v1",
        ),
        _profile(
            "qwen3_vl_8b",
            "qwen3_vl",
            "Qwen/Qwen3-VL-8B-Instruct",
            "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
            "question_only",
            chat_template="Qwen3-VL native structured image chat template; no system message",
            inference_protocol="msmu_qwen3_vl_8b_question_only_deterministic_v1",
        ),
        _profile(
            "qwen3_vl_32b",
            "qwen3_vl",
            "Qwen/Qwen3-VL-32B-Instruct",
            "0cfaf48183f594c314753d30a4c4974bc75f3ccb",
            "question_only",
            chat_template="Qwen3-VL native structured image chat template; no system message",
            inference_protocol="msmu_qwen3_vl_32b_question_only_deterministic_v1",
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
        _profile(
            "robobrain25_8b_nv_rgb",
            "robobrain25",
            "BAAI/RoboBrain2.5-8B-NV",
            "3d77a19a3ddd8616b3979e03de56096edfb12ff6",
            "rgb_original_first_question",
            max_new_tokens=768,
            temperature=0.7,
            chat_template="RoboBrain2.5 official AutoProcessor structured-image template",
            upstream_url="https://github.com/FlagOpen/RoboBrain2.5",
            upstream_commit="af98c932aac9ff715d70da177088d7bb95573ff7",
            inference_protocol=(
                "msmu_robobrain25_8b_nv_rgb_original_first_question_"
                "official_general_sampling_t07_top_p08_768_v1"
            ),
            do_sample=True,
            top_p=0.8,
        ),
        _profile(
            "robobrain25_8b_mt_rgb",
            "robobrain25",
            "BAAI/RoboBrain2.5-8B-MT",
            "01145b89a0fe49f78f5d677d25af7351088d7c7d",
            "rgb_original_first_question",
            max_new_tokens=768,
            temperature=0.7,
            chat_template="RoboBrain2.5 official AutoProcessor structured-image template",
            upstream_url="https://github.com/FlagOpen/RoboBrain2.5",
            upstream_commit="af98c932aac9ff715d70da177088d7bb95573ff7",
            inference_protocol=(
                "msmu_robobrain25_8b_mt_rgb_original_first_question_"
                "official_general_sampling_t07_top_p08_768_v1"
            ),
            do_sample=True,
            top_p=0.8,
        ),
        _profile(
            "hispatial3b_moge2_xyz",
            "hispatial",
            "lhzzzzzy/HiSpatial-3B",
            "75a5e3d65351d7602c492aa91533f62b8a252604",
            "same_rgb_moge2_xyz_original_first_question",
            max_new_tokens=100,
            temperature=None,
            chat_template="HiSpatial official PaliGemma predictor template",
            upstream_url="https://github.com/microsoft/HiSpatial",
            upstream_commit="9b0a5718ed0fb3b8bd9d9e0b36b6192bd3e99be1",
            inference_protocol=(
                "msmu_hispatial3b_same_rgb_moge2_xyz_original_first_question_"
                "official_predictor_greedy100_v1"
            ),
        ),
        _profile(
            "spatialladder3b_rgb",
            "spatialladder",
            "hongxingli/SpatialLadder-3B",
            "0819c3adf8827a2ea6c0348d49a23503ecb1f428",
            "rgb_original_first_question_direct",
            max_new_tokens=128,
            temperature=0.01,
            chat_template="SpatialLadder official Qwen2.5-VL structured-image template",
            upstream_url="https://github.com/ZJU-REAL/SpatialLadder",
            upstream_commit="7a0d2ee85c28728835300310a349a53a15967f2e",
            inference_protocol=(
                "msmu_spatialladder3b_rgb_original_first_question_direct_"
                "flashattn2_leftpad_native_batch_128_v1"
            ),
            do_sample=True,
            top_p=1.0,
            repetition_penalty=1.05,
        ),
        _profile(
            "spatialladder3b_thinking",
            "spatialladder",
            "hongxingli/SpatialLadder-3B",
            "0819c3adf8827a2ea6c0348d49a23503ecb1f428",
            "rgb_official_generic_special_thinking",
            max_new_tokens=1024,
            temperature=0.01,
            chat_template=(
                "SpatialLadder official Qwen2.5-VL structured-image template plus "
                "SPAR-Bench generic special thinking prompt"
            ),
            upstream_url="https://github.com/ZJU-REAL/SpatialLadder",
            upstream_commit="7a0d2ee85c28728835300310a349a53a15967f2e",
            inference_protocol=(
                "msmu_spatialladder3b_rgb_official_generic_special_thinking_"
                "flashattn2_leftpad_native_batch_last_answer_1024_v1"
            ),
            do_sample=True,
            top_p=1.0,
            repetition_penalty=1.05,
        ),
    ]
}


# Current MSMU comparison scope. Registered profiles outside this tuple remain
# available for historical reproduction but are not part of the target matrix.
CURRENT_TARGET_PROFILE_KEYS = (
    "gpt5_openrouter_non_zdr",
    "gemini31pro_openrouter_non_zdr",
    "llava_next_mistral_7b",
    "llava_next_yi_34b",
    "internvl3_8b",
    "internvl3_38b",
    "internvl3_78b",
    "qwen3_vl_2b",
    "qwen3_vl_4b",
    "qwen3_vl_8b",
    "qwen3_vl_32b",
    "ssr",
    "ssr_native",
    "spatialrgpt",
    "3dthinker",
    "3dthinker_native",
    "spatialbot",
    "spatialbot_native",
)


# Registered MSMU SOTA supplement. These profiles remain outside
# CURRENT_TARGET_PROFILE_KEYS until their live full-987 validator, scorer, and
# publication artifacts have all been verified. The thinking track is always a
# supplementary row and never becomes a main target profile.
SOTA_SUPPLEMENT_MAIN_PROFILE_KEYS = (
    "robobrain25_8b_nv_rgb",
    "robobrain25_8b_mt_rgb",
    "hispatial3b_moge2_xyz",
    "spatialladder3b_rgb",
)
SOTA_SUPPLEMENT_PROFILE_KEYS = (
    *SOTA_SUPPLEMENT_MAIN_PROFILE_KEYS,
    "spatialladder3b_thinking",
)
SOTA_SUPPLEMENT_REPORT_PROFILE_KEYS = (
    *CURRENT_TARGET_PROFILE_KEYS,
    *SOTA_SUPPLEMENT_PROFILE_KEYS,
)


def get_profile(key: str) -> InferenceProfile:
    try:
        return PROFILES[str(key)]
    except KeyError as exc:
        raise ValueError(f"Unknown inference profile {key!r}; choose from {sorted(PROFILES)}") from exc


def profile_keys(*families: str) -> list[str]:
    allowed = set(families)
    return sorted(key for key, profile in PROFILES.items() if profile.family in allowed)
