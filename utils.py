from __future__ import annotations

import argparse
import json
import math
import os
import json
import random
import re
import copy
import pandas as pd
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from openai import OpenAI
import httpx
from scipy import stats
from scipy.stats import chi2_contingency, norm


# ============================================================================
# Constants
# ============================================================================

Z_975 = norm.ppf(0.975)
RANDOM_SEED = 42
# Note: although the output filename is always "analysis.json", it is saved
# inside a run-specific results directory (determined by model/study_id/etc.).
# Therefore, different analyses do not overwrite one another unless they are
# intentionally written to the same folder.
ANALYSIS_OUTPUT_FILENAME = "analysis.json"
ANALYSIS_OUTPUT_FILENAME = "analysis.json"
# These studies encode the target effect in the opposite behavioral direction.
SPECIAL_REVERSED_STUDIES = {"Dickerson_1", "Dickerson_2", "Catlin_1"}
SYSTEM_PROMPT_RELATIVE_PATH = ("prompt", "system_prompt.json")
NUMBER_PATTERN = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
SYSTEM_ROLE_PREFIX = (
    "Forget you are an AI model. You are a human participant in a social experiment."
)

# ============================================================================
# Path and configuration helpers
# ============================================================================


def check_file_path(path: str) -> str:
    """Validate that the given path points to an existing file."""
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f"{path} is not a valid file path.")
    return path


def _get_current_file_dir() -> Path:
    """Return the directory containing this script."""
    return Path(__file__).resolve().parent


def get_system_prompt() -> str:
    """Load and assemble the system prompt from the JSON config file."""
    prompt_path = _get_current_file_dir().joinpath(*SYSTEM_PROMPT_RELATIVE_PATH)

    with prompt_path.open("r", encoding="utf-8") as file:
        system_prompt_data = json.load(file)

    instruction = system_prompt_data.get("instruction", "")
    traits = system_prompt_data.get("behavioral_traits", [])
    system_prompt = instruction + " " + " ".join(traits)
    return system_prompt


system_prompt = get_system_prompt()


@lru_cache(maxsize=2)
def get_api_client(provider: str) -> OpenAI:
    """Create an API client only when an experiment makes its first request."""
    if provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = os.getenv("DEEPSEEK_API_BASE")
        if not api_key or not base_url:
            raise RuntimeError(
                "DEEPSEEK_API_KEY and DEEPSEEK_API_BASE are required for DeepSeek models."
            )
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_API_BASE")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI-compatible models.")

    client_args = {
        "api_key": api_key,
        "http_client": httpx.Client(follow_redirects=True),
    }
    if base_url:
        client_args["base_url"] = base_url
    return OpenAI(**client_args)

# ============================================================================
# LLM interaction
# ============================================================================


def call_res(role: str, exp: str, model_name: str, temperature: float) -> tuple[str, str]:
    """Call the chat completion API and return the text response and reasoning content."""
    api_client = get_api_client(
        "deepseek" if model_name == "deepseek-v4-flash" else "openai"
    )
    # Reasoning models require an explicit reasoning_effort setting.
    if model_name in ["gpt-5.4"]:
        response = api_client.chat.completions.create(
            model=model_name,
            reasoning_effort="low",
            extra_body={"thinking": {"type": "enabled"}},
            messages=[
                {"role": "system", "content": role},
                {"role": "user", "content": exp},
            ],
            temperature=temperature,
        )
    elif model_name in ["deepseek-v4-flash"]:
        response = api_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": role},
                {"role": "user", "content": exp},
            ],
            stream=False,
            reasoning_effort="low",
            extra_body={"thinking": {"type": "enabled"}}
        )
    else:
        response = api_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": role},
                {"role": "user", "content": exp},
            ],
            temperature=temperature,
        )
    message = response.choices[0].message
    output = message.content or ""
    reason_output = getattr(message, "reasoning_content", None)
    if reason_output is None:
        reasoning_tokens = getattr(
            response.usage.completion_tokens_details,
            "reasoning_tokens",
            0,
        )
        reason_output = f"[Reasoning tokens: {reasoning_tokens}]"
    return output, reason_output



def get_res(
    role: str,
    exp: str,
    model_name: str,
    temperature: float,
    system_prompt: str = system_prompt,
) -> Dict[str, Any]:
    """Build the final prompt and return both input and output content."""
    role = role.strip()
    role = SYSTEM_ROLE_PREFIX + " " + role
    system_prompt = system_prompt.strip()
    role = role + " " + system_prompt

    exp = exp.strip()
    message = role + "\n" + exp
    res, reason = call_res(role, exp, model_name, temperature)

    message_parts = message.split("\n")
    result = {
        "input": [part for part in message_parts],
        "output": res,
        "reason_output": reason,
    }
    return result


def _parse_single_numeric_output(output: str, is_binary: bool) -> Optional[float]:
    """Parse exactly one numeric value from the model output.

    This parser is intentionally lenient: if the output does not contain
    exactly one valid numeric value, it returns None instead of raising
    an exception.

    Returning None here means the current generation is considered invalid.
    The caller (the retry loop in `get_fixed_response`) is expected
    to detect this and trigger another generation attempt.

    Returns:
        Parsed number if valid, otherwise None.
    """
    matches = re.findall(NUMBER_PATTERN, output)
    if len(matches) != 1:
        return None

    num_str = matches[0]

    try:
        num = float(num_str)
    except ValueError:
        return None

    if is_binary:
        return int(num) if num in {0.0, 1.0} else None

    return int(num) if num.is_integer() else num


def get_fixed_response(
    role: str,
    exp: str,
    model_name: str,
    temperature: float,
    cha_num: int,
    max_attempts: int,
    is_binary: bool,
    round_num: Optional[int] = None,
) -> Dict[str, Any]:
    """Retry until a single valid numeric response is produced or attempts end."""
    for attempt in range(max_attempts):
        chara_res = get_res(role, exp, model_name, temperature)
        output = chara_res.get("output", "").strip()

        # Only accept outputs that contain exactly one valid numeric answer.
        parsed_value = _parse_single_numeric_output(output, is_binary)

        if parsed_value is not None:
            result = {
                "input": chara_res.get("input", ""),
                "output": output,
                "result": parsed_value,
                "reason_output": chara_res.get("reason_output", ""),
            }
            if round_num is not None:
                result["round"] = round_num
            return result

        print(f"⚠️ Attempt {attempt + 1} failed for chara {cha_num}: {output}")

    print(f"⚠️ Failed to parse output after {max_attempts} attempts.")

    result = {
        "input": chara_res.get("input", ""),
        "output": output,
        "result": None,
        "reason_output": chara_res.get("reason_output", ""),
    }
    if round_num is not None:
        result["round"] = round_num
    return result


def get_fixed_response_with_traits(
    role: str,
    exp: str,
    model_name: str,
    temperature: float,
    cha_num: int,
    max_attempts: int,
    is_binary: bool,
    round_num: Optional[int] = None,
) -> Dict[str, Any]:
    """Retry until valid response_with_traits is produced or attempts end."""
    for attempt in range(max_attempts):
        chara_res = get_res(role, exp, model_name, temperature)
        output = chara_res.get("output", "").strip()

        # Expected format: result + four psychological/environmental traits + SVO.
        parts = output.split()
        if len(parts) != 6:
            print(
                f"⚠️ Attempt {attempt + 1} failed for chara {cha_num}: expected 6 fields, got {len(parts)} -> {output}"
            )
            continue

        result_value = _parse_single_numeric_output(parts[0], is_binary)
        if result_value is None:
            print(
                f"⚠️ Attempt {attempt + 1} failed for chara {cha_num}: invalid result field -> {parts[0]}"
            )
            continue

        try:
            # Trait fields are parsed separately so invalid metadata can trigger retry.
            env_self_efficacy_pre = float(parts[1])
            env_attitude_pre = float(parts[2])
            env_motivation_pre = float(parts[3])
            svo = parts[4]
            sensitivity_score = float(parts[5])
        except ValueError:
            print(
                f"⚠️ Attempt {attempt + 1} failed for chara {cha_num}: invalid trait fields -> {output}"
            )
            continue

        if not (1 <= env_self_efficacy_pre <= 5):
            print(
                f"⚠️ Attempt {attempt + 1} failed for chara {cha_num}: "
                f"envSelfEfficacy_pre out of range -> {env_self_efficacy_pre}"
            )
            continue

        if not (1 <= env_attitude_pre <= 5):
            print(
                f"⚠️ Attempt {attempt + 1} failed for chara {cha_num}: "
                f"envAttitude_pre out of range -> {env_attitude_pre}"
            )
            continue

        if not (-14 <= env_motivation_pre <= 14):
            print(
                f"⚠️ Attempt {attempt + 1} failed for chara {cha_num}: "
                f"envMotivation_pre out of range -> {env_motivation_pre}"
            )
            continue
        if svo.lower() not in {"proself", "prosocial", "none"}:
            print(
                f"⚠️ Attempt {attempt + 1} failed for chara {cha_num}: "
                f"invalid svo -> {svo}"
            )
            continue

        if not (1 <= sensitivity_score <= 5):
            print(
                f"⚠️ Attempt {attempt + 1} failed for chara {cha_num}: "
                f"sensitivity_score out of range -> {sensitivity_score}"
            )
            continue

        result = {
            "input": chara_res.get("input", ""),
            "output": output,
            "result": result_value,
            "envSelfEfficacy_pre": env_self_efficacy_pre,
            "envAttitude_pre": env_attitude_pre,
            "envMotivation_pre": env_motivation_pre,
            "svo": svo,
            "sensitivity_score": sensitivity_score,
            "reason_output": chara_res.get("reason_output", ""),
        }
        if round_num is not None:
            result["round"] = round_num
        return result

    print(f"⚠️ Failed to parse output with traits after {max_attempts} attempts.")

    result = {
        "input": chara_res.get("input", ""),
        "output": output,
        "result": None,
        "envSelfEfficacy_pre": None,
        "envAttitude_pre": None,
        "envMotivation_pre": None,
        "svo": None,
        "sensitivity_score": None,
        "reason_output": chara_res.get("reason_output", ""),
    }
    if round_num is not None:
        result["round"] = round_num
    return result


def has_existing_json_results(directory: Path) -> bool:
    """Return True if the directory already contains any JSON result file."""
    try:
        return any(
            path.is_file() and path.suffix == ".json" for path in directory.iterdir()
        )
    except OSError as exc:
        raise OSError(f"Failed to inspect directory: {directory}") from exc


def get_existing_max_round(study_group_dir: Path) -> int:
    json_files = list(study_group_dir.glob("*.json"))

    if not json_files:
        return 0

    max_rounds = []

    for jf in json_files:
        try:
            with open(jf, "r") as f:
                data = json.load(f)
                if not data:
                    continue
                rounds = [entry.get("round", 0) for entry in data if "round" in entry]
                if rounds:
                    max_rounds.append(max(rounds))
        except Exception:
            continue

    if not max_rounds:
        return 0

    # Use the minimum completed round to resume only from the safest common point.
    return min(max_rounds)


def load_existing_data_store(study_group_dir: Path) -> dict[int, list[Any]]:
    data_store = {}
    json_files = sorted(
        [p for p in study_group_dir.glob("*.json") if p.stem.isdigit()],
        key=lambda p: int(p.stem),
    )

    for jf in json_files:
        agent_id = int(jf.stem) - 1
        try:
            with open(jf, "r") as f:
                data_store[agent_id] = json.load(f)
        except Exception:
            data_store[agent_id] = []

    return data_store


# Baseline profiles are reused from the corresponding short-term experiment.
PROFILE_BASE_RELATIVE_PATH = (
    Path("..")
    / ".."
    / ".."
    / ".."
    / ".."
    / ".."
    / "study1"
    / "nudge_replication"
    / "res"
    / "deepseek-v4-flash_res"
)
# Contact simulations use a different relative path.
PROFILE_BASE_RELATIVE_PATH_CONTACT = (
    Path("..")
    / ".."
    / ".."
    / ".."
    / ".."
    / ".."
    / ".."
    / ".."
    / "study1"
    / "nudge_replication"
    / "res"
    / "deepseek-v4-flash_res"
)


def load_baseline_round_result(
    round_num: int,
    cha_num: int,
    exp_data: pd.Series,
    data_store: dict[int, list[Any]],
    group: str,
    contact: bool = False,
) -> bool:
    """
    For round 1/2, load historical baseline responses from disk.
    Return True if this round is handled here; otherwise False.
    """
    if contact:
        base_path = PROFILE_BASE_RELATIVE_PATH_CONTACT
    else:
        base_path = PROFILE_BASE_RELATIVE_PATH
    if round_num == 1:
        json_path = Path(
            base_path,
            str(exp_data["study_id"]),
            "control.json",
        ).resolve()
    elif round_num == 2:
        json_path = Path(
            base_path,
            str(exp_data["study_id"]),
            f"{group}.json",
        ).resolve()
    else:
        return False

    if not json_path.exists():
        print(f"[Error] File not found: {json_path}")
        return True

    try:
        with json_path.open("r", encoding="utf-8") as file:
            all_data = json.load(file)

        if isinstance(all_data, list) and 0 <= cha_num - 1 < len(all_data):
            # Copy the stored short-term response.
            result_item = copy.deepcopy(all_data[cha_num - 1])
            result_item["round"] = round_num
            data_store.setdefault(cha_num - 1, []).append(result_item)
        else:
            print(f"[Warning] No agent {cha_num} data found for round {round_num}")
    except json.JSONDecodeError:
        print(
            f"[Error] Failed to parse JSON for cha_num {cha_num} at round {round_num}"
        )
    except OSError as exc:
        print(
            f"[Error] Failed to read file for cha_num {cha_num} at round {round_num}: {exc}"
        )

    return True


def save_checkpoint(round_num: int, num_rounds: int, data_store: dict) -> None:
    """Save recent checkpoint data for each character.

    Checkpoints are written every 10 rounds and at the final round.
    Only rounds that have not yet been written to the result JSON file
    are appended.
    """
    if round_num % 10 != 0 and round_num != num_rounds:
        return

    print(f"Saving checkpoint at round {round_num}...")

    for cha_num, data in data_store.items():
        filename = f"{cha_num + 1}.json"

        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                chara_data = json.load(f)
        else:
            chara_data = []

        if chara_data:
            last_saved_round = max(
                item["round"]
                for item in chara_data
            )
        else:
            last_saved_round = 0

        recent_data = [
            item
            for item in data
            if item["round"] > last_saved_round
        ]

        if not recent_data:
            continue

        chara_data.extend(recent_data)

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(chara_data, f, ensure_ascii=False, indent=4)


# ============================================================================
# Statistical utilities
# ============================================================================


def compute_effect_size(
    mean_intervention: float,
    mean_control: float,
    std_intervention: float,
    std_control: float,
    n_intervention: int,
    n_control: int,
    is_binary: bool,
) -> float:
    """Compute Cohen's h for binary outcomes and Cohen's d for continuous outcomes."""
    if is_binary:
        # Binary outcomes use Cohen's h based on the arcsine square-root transformation.
        if not (0 <= mean_intervention <= 1) or not (0 <= mean_control <= 1):
            raise ValueError("Proportions must be between 0 and 1 when is_binary=True.")
        phi_intervention = 2 * math.asin(math.sqrt(mean_intervention))
        phi_control = 2 * math.asin(math.sqrt(mean_control))
        h = phi_intervention - phi_control
        return h

    # Continuous outcomes use Cohen's d based on the pooled standard deviation.
    pooled_sd = math.sqrt(
        ((n_intervention - 1) * std_intervention**2 + (n_control - 1) * std_control**2)
        / (n_intervention + n_control - 2)
    )
    d = (mean_intervention - mean_control) / pooled_sd
    return d


def p_value_and_stat_from_proportions(
    mean_intervention: float,
    mean_control: float,
    std_intervention: float,
    std_control: float,
    n_intervention: int,
    n_control: int,
    is_binary: bool,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Compute test statistic, p-value, and degrees of freedom."""
    # Degenerate equal-variance case: identical means imply no detectable difference.
    if std_intervention == 0 and std_control == 0:
        if mean_intervention == mean_control:
            t_stat = 0.0
            p_value = 1.0
            df = n_intervention + n_control - 2
            return t_stat, p_value, df
        return None, None, None

    if is_binary:
        # Build a 2x2 contingency table from group proportions and sample sizes.
        success_intervention = mean_intervention * n_intervention
        fail_intervention = n_intervention - success_intervention
        success_control = mean_control * n_control
        fail_control = n_control - success_control

        contingency_table = [
            [success_intervention, fail_intervention],
            [success_control, fail_control],
        ]
        table = np.array(contingency_table)

        if (table.sum(axis=0) == 0).any() or (table.sum(axis=1) == 0).any():
            return None, None, None

        try:
            chi2, p_value, dof, expected = chi2_contingency(table)
            if (expected == 0).any():
                return None, None, None
            return chi2, p_value, dof
        except ValueError:
            return None, None, None

    t_stat, p_value = stats.ttest_ind_from_stats(
        mean1=mean_intervention,
        std1=std_intervention,
        nobs1=n_intervention,
        mean2=mean_control,
        std2=std_control,
        nobs2=n_control,
        equal_var=False,
    )

    # Welch-Satterthwaite approximation for unequal-variance t-test degrees of freedom.
    numerator = (std_intervention**2 / n_intervention + std_control**2 / n_control) ** 2
    denominator = ((std_intervention**2 / n_intervention) ** 2) / (
        n_intervention - 1
    ) + ((std_control**2 / n_control) ** 2) / (n_control - 1)
    df = numerator / denominator
    return t_stat, p_value, df


def calculate_mean_std_result(
    res_list: Sequence[Dict[str, Any]],
) -> Tuple[Optional[float], Optional[float], int]:
    """Compute mean, sample standard deviation, and valid count."""
    values: List[float] = []

    for item in res_list:
        result = item.get("result")
        if result is None:
            continue
        if isinstance(result, (int, float)):
            values.append(result)

    count = len(values)
    if count == 0:
        return None, None, 0

    mean_result = sum(values) / count
    variance = (
        sum((x - mean_result) ** 2 for x in values) / (count - 1) if count > 1 else 0
    )
    std_result = math.sqrt(variance)
    return mean_result, std_result, count


def compute_var_effect_size(n1: int, n2: int, d: float) -> float:
    """Compute the variance of effect size."""
    var_d = (n1 + n2) / (n1 * n2) + (d**2) / (2 * (n1 + n2))
    return var_d


# ============================================================================
# Result analysis
# ============================================================================


def _should_reverse_effect_direction(study_id: str) -> bool:
    """Return whether the study uses reversed effect direction."""
    return study_id in SPECIAL_REVERSED_STUDIES


def analyze_results(
    study_id: str,
    res_control: Sequence[Dict[str, Any]],
    res_intervention: Sequence[Dict[str, Any]],
    exp_data: Dict[str, Any],
) -> None:
    """Analyze LLM and human results and save the output as analysis.json
    within the run-specific result directory.

    Each run uses a separate directory for its study ID and LLM model, so this
    fixed filename does not overwrite analyses from other runs.
    """
    mean_control, std_control, n_control_new = calculate_mean_std_result(res_control)
    mean_intervention, std_intervention, n_intervention_new = calculate_mean_std_result(
        res_intervention
    )

    effect_size_llm = compute_effect_size(
        mean_intervention,
        mean_control,
        std_intervention,
        std_control,
        n_intervention_new,
        n_control_new,
        exp_data["binary_outcome"],
    )

    stat_llm, p_llm, df_llm = p_value_and_stat_from_proportions(
        mean_intervention,
        mean_control,
        std_intervention,
        std_control,
        n_intervention_new,
        n_control_new,
        exp_data["binary_outcome"],
    )

    test_stat_name = "chi2" if exp_data["binary_outcome"] == 1 else "t_stat"
    var_llm = compute_var_effect_size(
        exp_data["n_control"], exp_data["n_intervention"], effect_size_llm
    )
    ci_lower_llm = effect_size_llm - Z_975 * np.sqrt(var_llm)
    ci_upper_llm = effect_size_llm + Z_975 * np.sqrt(var_llm)

    if _should_reverse_effect_direction(study_id):
        # Keep effect signs comparable across studies with reversed coding.
        effect_size_llm = effect_size_llm * -1
        ci_new_lower_llm = -ci_upper_llm
        ci_new_upper_llm = -ci_lower_llm
        ci_upper_llm, ci_lower_llm = ci_new_upper_llm, ci_new_lower_llm

    llm_result = {
        "mean_control": mean_control,
        "mean_intervention": mean_intervention,
        "std_control": std_control,
        "std_intervention": std_intervention,
        "effect_size": effect_size_llm,
        test_stat_name: stat_llm,
        "p_value": p_llm,
        "df": df_llm,
        "variance_d": var_llm,
        "ci_lower": ci_lower_llm,
        "ci_upper": ci_upper_llm,
    }

    try:
        # Prefer recomputing human effect size from raw summary statistics when possible.
        effect_size_human = compute_effect_size(
            exp_data["mean_intervention"],
            exp_data["mean_control"],
            exp_data["sd_intervention"],
            exp_data["sd_control"],
            exp_data["n_intervention"],
            exp_data["n_control"],
            exp_data["binary_outcome"],
        )

        if _should_reverse_effect_direction(study_id):
            effect_size_human = effect_size_human * -1

        if effect_size_human is None or (
            isinstance(effect_size_human, float) and math.isnan(effect_size_human)
        ):
            raise ValueError("Effect Size is NaN or None")
    except Exception:
        effect_size_human = exp_data.get("effect_size", None)

    human_result = {
        "mean_control": exp_data["mean_control"],
        "mean_intervention": exp_data["mean_intervention"],
        "std_control": (
            exp_data.get("sd_control")
            if exp_data.get("sd_control") not in [None, ""]
            else None
        ),
        "std_intervention": (
            exp_data.get("sd_intervention")
            if exp_data.get("sd_intervention") not in [None, ""]
            else None
        ),
        "effect_size": effect_size_human,
        "variance_d": exp_data["variance_d"],
        "ci_lower": exp_data["ci_lower"],
        "ci_upper": exp_data["ci_upper"],
    }

    analysis_result = {
        "human_result": human_result,
        "llm_result": llm_result,
    }

    # Save analysis in the current run directory.
    output_path = Path(ANALYSIS_OUTPUT_FILENAME)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(analysis_result, file, indent=2, ensure_ascii=False)


# ============================================================================
# Graph and interaction utilities
# ============================================================================


def get_all_edges(data: Dict[str, Sequence[int]]) -> List[Tuple[int, int]]:
    """Extract unique undirected edges from the adjacency mapping."""
    edges: List[Tuple[int, int]] = []

    for node, neighbors in data.items():
        for neighbor in neighbors:
            # Sort node IDs so (u, v) and (v, u) are treated as the same edge.
            edge = tuple(sorted((int(node), neighbor)))
            if edge not in edges:
                edges.append(edge)

    return edges


def activate_edges(
    edges: Sequence[Tuple[int, int]],
    activation_rate: float,
) -> List[Tuple[int, int]]:
    """Activate a subset of edges based on the given rate."""
    num_edges_to_activate = int(len(edges) * activation_rate)
    random.seed(RANDOM_SEED)
    # Fixed seed keeps the activated network reproducible across runs.
    activated_edges = random.sample(edges, num_edges_to_activate)
    return activated_edges


def find_results(data: Sequence[Dict[str, Any]], round_num: int) -> Any:
    """Find the stored result for a specific round."""
    for entry in data:
        if entry["round"] == round_num:
            return entry["result"]
    return None


def load_friend_actions(
    activated_pairs: Dict[int, Sequence[int]],
    cha_num: int,
    round_num: int,
    data_store: Dict[int, Sequence[Dict[str, Any]]],
) -> Tuple[List[Any], int]:
    """Load friend actions from the previous round."""
    friend_results: List[Any] = []
    num_activated_friends = 0

    for friend_id in activated_pairs.get(cha_num, []):
        num_activated_friends += 1
        data = data_store.get(friend_id - 1)
        result = find_results(data, round_num - 1)
        friend_results.append(result)

    return friend_results, num_activated_friends
