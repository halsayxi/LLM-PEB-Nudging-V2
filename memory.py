from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

# Trait fields that are expected in each agent profile and model output.
TRAIT_KEYS = [
    "envSelfEfficacy_pre",
    "envAttitude_pre",
    "envMotivation_pre",
    "svo",
    "sensitivity_score",
]

# Valid ranges used to constrain numeric trait updates across rounds.
NUMERIC_TRAIT_BOUNDS = {
    "envSelfEfficacy_pre": (1.0, 5.0),
    "envAttitude_pre": (1.0, 5.0),
    "envMotivation_pre": (-14.0, 14.0),
    "sensitivity_score": (1.0, 5.0),
}

# Accepted social value orientation labels.
ALLOWED_SVO = {"proself", "prosocial", "none"}


def load_json_file(path: Path) -> Any:
    """Load a JSON file using UTF-8 encoding."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json_file(path: Path, data: Any) -> None:
    """Save JSON data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_effective_group_size(
    n_control: int, n_intervention: int, max_chars: int = -1
) -> int:
    """Return the balanced group size, optionally capped by max_chars."""
    n_group = min(n_control, n_intervention)
    return min(n_group, max_chars) if max_chars != -1 else n_group


def get_source_profile_paths(
    profile_base_relative_path: Path,
    study_id: Any,
    group: str,
    n_control: int,
    n_intervention: int,
) -> tuple[Path, Path]:
    # Source profile filenames depend on both group type and original group size.
    n_group = n_control if group == "control" else n_intervention
    profile_dir = profile_base_relative_path / str(study_id) / "profile"
    agent_data_path = profile_dir / f"agent_data_{group}_{n_group}.json"
    character_path = profile_dir / f"character_{group}_{n_group}.json"
    return agent_data_path, character_path


def get_local_profile_dir(study_group_dir: Path) -> Path:
    """Return the local profile directory for the current study/group run."""
    profile_dir = study_group_dir / "profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    return profile_dir


def get_local_base_profile_paths(
    study_group_dir: Path,
    group: str,
    effective_n: int,
) -> tuple[Path, Path]:
    profile_dir = get_local_profile_dir(study_group_dir)
    agent_data_path = profile_dir / f"agent_data_{group}_{effective_n}.json"
    character_path = profile_dir / f"character_{group}_{effective_n}.json"
    return agent_data_path, character_path


def get_local_round_profile_paths(
    study_group_dir: Path,
    group: str,
    effective_n: int,
    round_num: int,
) -> tuple[Path, Path]:
    profile_dir = get_local_profile_dir(study_group_dir)
    agent_data_path = (
        profile_dir / f"agent_data_{group}_{effective_n}_round_{round_num}.json"
    )
    character_path = (
        profile_dir / f"character_{group}_{effective_n}_round_{round_num}.json"
    )
    return agent_data_path, character_path


def prepare_local_profile_files(
    profile_base_relative_path: Path,
    study_group_dir: Path,
    study_id: Any,
    group: str,
    n_control: int,
    n_intervention: int,
    num_chars: int,
) -> tuple[Path, Path, list[dict[str, Any]]]:
    # Copy the original baseline profiles into the current run directory.
    src_agent_data_path, src_character_path = get_source_profile_paths(
        profile_base_relative_path=profile_base_relative_path,
        study_id=study_id,
        group=group,
        n_control=n_control,
        n_intervention=n_intervention,
    )

    if not src_agent_data_path.exists():
        raise FileNotFoundError(
            f"Source agent_data file not found: {src_agent_data_path}"
        )
    if not src_character_path.exists():
        raise FileNotFoundError(
            f"Source character file not found: {src_character_path}"
        )

    effective_n = num_chars

    dst_agent_data_path, dst_character_path = get_local_base_profile_paths(
        study_group_dir=study_group_dir,
        group=group,
        effective_n=effective_n,
    )

    agent_data = load_json_file(src_agent_data_path)
    if not isinstance(agent_data, list):
        raise ValueError(
            f"Expected list in {src_agent_data_path}, got {type(agent_data).__name__}"
        )

    character_data = load_json_file(src_character_path)
    if not isinstance(character_data, dict):
        raise ValueError(
            f"Expected dict in {src_character_path}, got {type(character_data).__name__}"
        )

    # Keep only the agents used in the current run and avoid mutating source data.
    trimmed_agents = copy.deepcopy(agent_data[:effective_n])
    trimmed_characters = {
        str(i + 1): character_data[str(i + 1)]
        for i in range(effective_n)
        if str(i + 1) in character_data
    }

    if len(trimmed_characters) != effective_n:
        raise ValueError(
            f"Character file {src_character_path} does not contain enough entries for {effective_n} agents."
        )

    save_json_file(dst_agent_data_path, trimmed_agents)
    save_json_file(dst_character_path, trimmed_characters)

    return dst_agent_data_path, dst_character_path, trimmed_agents


def load_characters(chara_file_path: Path) -> list[str]:
    data = load_json_file(chara_file_path)
    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a JSON object in character file, got {type(data).__name__}: {chara_file_path}"
        )
    return [data[str(i)] for i in range(1, len(data) + 1)]


def load_round_characters(
    study_group_dir: Path,
    group: str,
    effective_n: int,
    round_num: int,
) -> list[str]:
    # Rounds 1-2 use the base profile; later rounds use the previous round snapshot.
    if round_num <= 2:
        _, char_path = get_local_base_profile_paths(study_group_dir, group, effective_n)
    else:
        _, char_path = get_local_round_profile_paths(
            study_group_dir, group, effective_n, round_num - 1
        )
    return load_characters(char_path)


def get_agent_profiles_for_round(
    study_group_dir: Path,
    group: str,
    effective_n: int,
    round_num: int,
) -> list[dict[str, Any]]:
    # Agent traits start from the short-term base profile on round 2.
    if round_num <= 2:
        agent_path, _ = get_local_base_profile_paths(
            study_group_dir, group, effective_n
        )
    else:
        agent_path, _ = get_local_round_profile_paths(
            study_group_dir, group, effective_n, round_num - 1
        )

    data = load_json_file(agent_path)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {agent_path}, got {type(data).__name__}")
    return data


def clamp(value: float, lower: float, upper: float) -> float:
    """Clamp a numeric value into the given closed interval."""
    return max(lower, min(upper, value))


def jitter_within_five_percent(
    current_value: float,
    proposed_value: Any,
    lower: float,
    upper: float,
) -> float:
    current_value = float(current_value)

    # Invalid proposed values fall back to the previous trait value.
    try:
        proposed = float(proposed_value)
    except Exception:
        proposed = current_value

    # Limit each trait update to at most ±5% of its current value.
    delta = abs(current_value) * 0.05
    if abs(current_value) < 1e-8:
        delta = 0.05

    lo = max(lower, current_value - delta)
    hi = min(upper, current_value + delta)

    if lo > hi:
        lo, hi = hi, lo

    return round(clamp(proposed, lo, hi), 4)


def normalize_svo(svo_value: Any, fallback: str) -> str:
    """Normalize SVO labels and fall back to the previous value if invalid."""
    candidate = str(svo_value).strip().lower()
    if candidate in ALLOWED_SVO:
        return candidate
    return fallback


def validate_and_update_agent_traits(
    previous_agent: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    agent = copy.deepcopy(previous_agent)

    # Numeric traits are constrained by both valid ranges and the ±5% update rule.
    for key, (lower, upper) in NUMERIC_TRAIT_BOUNDS.items():
        agent[key] = jitter_within_five_percent(
            current_value=float(previous_agent[key]),
            proposed_value=payload.get(key, previous_agent[key]),
            lower=lower,
            upper=upper,
        )

    agent["svo"] = normalize_svo(
        payload.get("svo", previous_agent["svo"]),
        str(previous_agent["svo"]).lower(),
    )
    return agent


def build_trait_output_instruction():
    return """
Output exactly 6 values in ONE line, separated by SINGLE spaces.

Format:
<result:number> <envSelfEfficacy_pre:float> <envAttitude_pre:float> <envMotivation_pre:float> <svo:str> <sensitivity_score:float>

Strict requirements:
1. result MUST be a single numeric value (integer or float).
2. Do NOT output any explanation, words, symbols, or additional text — ONLY the number.
3. Keep the four numeric psychological scores within ±5% of the current score.
4. Keep valid ranges:
   - envSelfEfficacy_pre: 1 to 5
   - envAttitude_pre: 1 to 5
   - envMotivation_pre: -14 to 14
   - sensitivity_score: 1 to 5
5. Use only a valid SVO label (proself, prosocial, none).

Output ONLY the line.
""".strip()


def save_round_profile_snapshots(
    study_group_dir: Path,
    group: str,
    effective_n: int,
    round_num: int,
    updated_agents: list[dict[str, Any]],
    generate_description_func,
) -> None:
    # Save both structured agent traits and regenerated natural-language profiles.
    agent_round_path, char_round_path = get_local_round_profile_paths(
        study_group_dir=study_group_dir,
        group=group,
        effective_n=effective_n,
        round_num=round_num,
    )

    trimmed_agents = copy.deepcopy(updated_agents[:effective_n])
    character_map = {
        str(i + 1): generate_description_func(trimmed_agents[i])
        for i in range(len(trimmed_agents))
    }

    save_json_file(agent_round_path, trimmed_agents)
    save_json_file(char_round_path, character_map)


def build_memory_text(
    data_store: Dict[int, Sequence[Dict[str, Any]]],
    cha_num: int,
    round_num: int,
    exp_data: Dict[str, Any],
    freq: int = 0,
    nudge: bool = True,
    procedural_memory_start_day: int = 2,
) -> Optional[str]:
    """Build memory text for the current participant based on prior rounds."""
    historical_text = None

    if round_num not in [1]:
        historical_choices = data_store.get(cha_num - 1)

        if historical_choices:
            historical_text = "Your memory is summarized as follows:\n\n"

            # Event memory
            historical_text += "[Event Memory]\n"
            context = exp_data["environmental_context"]
            nudge_category = exp_data["intervention_category"]
            nudge_technique = exp_data["intervention_technique"]
            historical_text += f"You are currently in a pro-environmental decision scenario related to {context}.\n"
            if nudge:
                historical_text += f"In your earlier experience, {nudge_category} nudge was used, specifically {nudge_technique}.\n"

            if freq != 0:
                # For scheduled nudges, recall choices since the most recent nudge window.
                nudge_days = [
                    1 + i * freq for i in range(len(historical_choices) // freq + 1)
                ]
                last_nudge_day = max(
                    [
                        day
                        for day in nudge_days
                        if day <= historical_choices[-1]["round"]
                    ],
                    default=1,
                )
                if round_num in nudge_days:
                    recent_choices = None
                else:
                    recent_choices = [
                        entry
                        for entry in historical_choices
                        if entry["round"] >= last_nudge_day
                    ]
                    if len(recent_choices) > 7:
                        recent_choices = recent_choices[-7:]
            else:
                # Without scheduled nudges, keep only the last seven decisions as event memory.
                recent_choices = historical_choices[-7:]

            if recent_choices:
                historical_text += "Your recent decision history in similar situations is as follows:\n"
                for entry in recent_choices:
                    day = entry["round"]
                    result = entry["result"]
                    historical_text += f"- Day {day}: {result}.\n"

            # Procedural memory
            if round_num >= procedural_memory_start_day:
                historical_text += "\n[Procedural Memory]\n"
                # Procedural memory summarizes long-term behavioral tendency and variability.
                results = [item["result"] for item in historical_choices]
                n = len(results)
                average_score = sum(results) / n
                variance = (
                    sum((x - average_score) ** 2 for x in results) / n if n > 0 else 0.0
                )
                historical_text += "Over repeated rounds, you have developed a behavioral habit in similar tasks.\n"
                historical_text += (
                    f"- Long-term average score: {average_score:.3f}\n"
                    f"- Variance: {variance:.3f}\n"
                )

    return historical_text
