from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Callable, Tuple
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from agent_data.attribute_to_description import process_agent_descriptions
from agent_data.generate_profile import generate_and_save_population
from utils import get_res


# =========================
# Config
# =========================
DEFAULT_MODEL_NAME = "deepseek-v4-flash"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_NUM_THREADS = 60
DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_NUM_DAYS = 14
DEFAULT_NUM_QUESTIONS_PER_DAY = 3
NUM_AGENTS_PER_GROUP = 50
DEFAULT_SEED = 42
DEFAULT_OUTPUT_DIR = "res_human"

# Supported experimental groups: control plus three nudge types at three exposure levels.
GROUPS = [
    "control",
    "structure_low",
    "information_low",
    "assistant_low",
    "information_high",
    "information_medium",
    "assistant_high",
    "assistant_medium",
    "structure_high",
    "structure_medium",
]

# Shared scenario shown before daily decision tasks.
SCENARIO_TEXT = "You are in a repeated study of brief choices involving effort, rewards, and environmental consequences."

BASE_CHOICE_BLOCK = (
    "Choice: How do you wish to spend the next 30 seconds?\n"
    "Option A. Do a task requiring moderate effort and generate a small donation for reforestation.\n"
    "Option B. Do a task requiring moderate effort and earn a small reward for yourself.\n"
    "Option C. Engage in an entertaining activity requiring almost no effort."
)

NUMBER_TASK_TEXT = "Number task: You will be presented with 60 two-digit numbers arranged in rows. Select all numbers with an even first digit and an odd second digit, working row by row. It is normal not to complete all rows within the time limit. Performance is recorded automatically."
SLIDER_TASK_TEXT = "Slider task: You will be presented with 6 pages of 6 sliders. Adjust each slider to its target as accurately as possible. It is normal not to complete all sliders within the time limit. Performance is recorded automatically."

INFO_NUDGE = "Before making your choice, please note that if you choose the environmentally friendly option and complete the relevant task correctly, 1 penny will be donated to support reforestation, and approximately £0.8 funds the planting of one tree. Across participants in this study, these small contributions can add up to support the planting of many new trees and help restore forest ecosystems."
ASSIS_NUDGE = "Before continuing, you were asked to take a moment setting a personal plan to help you act a bit more sustainably. Today, you plan to choose options that will generate donations as often as you can in the upcoming trial."
STRUC_NUDGE = "The environmentally friendly option A is currently pre-selected as the default. If you do not actively change your choice, the environmentally friendly option will be selected. Choice: keep option A / switch to option B / switch to option C."

MEMORY_SUFFIX = (
    "Memory is context only; reassess today independently instead of copying your last response."
)

ENV_POSTTEST_BLOCK = """
Please indicate how much you agree or disagree with each of the following statements about the environment.

Scale for q1-q23:
1 = Strongly Disagree
2 = Disagree
3 = Neutral
4 = Agree
5 = Strongly Agree

q1. We are approaching the limit of the number of people the Earth can support.
q2. Humans have the right to modify the natural environment to suit their needs.
q3. When humans interfere with nature it often produces disastrous consequences.
q4. Human ingenuity will ensure that we do not make the Earth unlivable.
q5. Humans are seriously abusing the environment.
q6. The Earth has plenty of natural resources if we just learn how to develop them.
q7. Plants and animals have as much right as humans to exist.
q8. The balance of nature is strong enough to cope with the impacts of modern industrial nations.
q9. Despite our special abilities, humans are still subject to the laws of nature.
q10. The so-called "ecological crisis" facing humankind has been greatly exaggerated.
q11. The Earth is like a spaceship with very limited room and resources.
q12. Humans were meant to rule over the rest of nature.
q13. The balance of nature is very delicate and easily upset.
q14. Humans will eventually learn enough about how nature works to be able to control it.
q15. If things continue on their present course, we will soon experience a major ecological catastrophe.
q16. Humans have the right to modify the natural environment to suit their needs.
q17. I believe I am capable of engaging in environmentally friendly behaviors in my daily life.
q18. I do things for the environment because I enjoy improving the quality of the environment.
q19. I do things for the environment because it has become a fundamental part of who I am.
q20. I do things for the environment because I believe it is a sensible thing to do.
q21. I do things for the environment because I would feel guilty if I didn't.
q22. I do things for the environment to avoid being criticized.
q23. I do things for the environment but sometimes I wonder why, since the situation isn't improving.
"""

INFO_NUDGE_POSTTEST_BLOCK = """
Please indicate the extent to which you agree with the following statements about your experience during the experiment.

Scale for n1-n3:
1 = Strongly Disagree
2 = Disagree
3 = Slightly Disagree
4 = Neutral
5 = Slightly Agree
6 = Agree
7 = Strongly Agree

n1. To what extent did you find the environmental information provided during the experiment helpful for encouraging you to engage in environmentally friendly behaviors?
n2. How bothered or annoyed did you feel about being presented with environmental information during the experiment?
n3. To what extent did you feel fatigued or tired of repeatedly receiving environmental information over time?
"""

ASSIS_NUDGE_POSTTEST_BLOCK = """
Please indicate the extent to which you agree with the following statements about your experience during the experiment.

Scale for n1-n3:
1 = Strongly Disagree
2 = Disagree
3 = Slightly Disagree
4 = Neutral
5 = Slightly Agree
6 = Agree
7 = Strongly Agree

n1. To what extent did you find the act of making a commitment helpful for encouraging you to engage in environmentally friendly behaviors?
n2. How bothered or annoyed did you feel about being asked to make a commitment regarding environmental behavior?
n3. To what extent did you feel fatigued or tired of being asked to make commitments over time?
"""

STRUC_NUDGE_POSTTEST_BLOCK = """
Please indicate the extent to which you agree with the following statements about your experience during the experiment.

Scale for n1-n3:
1 = Strongly Disagree
2 = Disagree
3 = Slightly Disagree
4 = Neutral
5 = Slightly Agree
6 = Agree
7 = Strongly Agree

n1. To what extent did you find the default environmental option helpful for encouraging you to engage in environmentally friendly behaviors?
n2. How bothered or annoyed did you feel about the default environmental option presented during the experiment?
n3. To what extent did you feel fatigued or tired of repeatedly encountering the default environmental option over time?
"""

ENV_ITEM_KEYS = [f"q{i}" for i in range(1, 24)]
NUDGE_ITEM_KEYS = ["n1", "n2", "n3"]

# Conservative fallback choices used only when model output cannot be parsed after retries.
DEFAULT_CHOICES = ["C"] * DEFAULT_NUM_QUESTIONS_PER_DAY

DEFAULT_POSTTEST_RESPONSES_CONTROL = {**{f"q{i}": 3 for i in range(1, 24)}}

DEFAULT_POSTTEST_RESPONSES_NUDGE = {
    **{f"q{i}": 3 for i in range(1, 24)},
    "n1": 4,
    "n2": 4,
    "n3": 4,
}


# =========================
# Argument parser
# =========================
def get_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--num_threads", type=int, default=DEFAULT_NUM_THREADS)
    parser.add_argument("--max_attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument("--num_days", type=int, default=DEFAULT_NUM_DAYS)
    parser.add_argument(
        "--two_days",
        action="store_true",
        help="Run only Days 1-2 (equivalent to --num_days 2).",
    )
    parser.add_argument(
        "--questions_per_day", type=int, default=DEFAULT_NUM_QUESTIONS_PER_DAY
    )
    parser.add_argument(
        "--include_task_content",
        action="store_true",
        help="Include the task-content section in daily prompts (disabled by default).",
    )
    parser.add_argument(
        "--posttest",
        action="store_true",
        help="Run the posttest after all daily decisions (disabled by default).",
    )
    parser.add_argument(
        "--group",
        type=str,
        default="all",
        choices=["all", *GROUPS],
        help="Run one group, or run all groups sequentially with 'all'.",
    )
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


# ========================
# Character building
# ========================
def build_all_chara(group: str) -> Tuple[List[str], List[int]]:
    """Load existing profiles, or generate them when no saved agent data exists."""
    is_control = group == "control"
    profile_group = "control" if is_control else "intervention"
    profile_dir = Path("profile")
    agent_data_path = (
        profile_dir / f"agent_data_{profile_group}_{NUM_AGENTS_PER_GROUP}.json"
    )
    character_path = (
        profile_dir / f"character_{profile_group}_{NUM_AGENTS_PER_GROUP}.json"
    )

    if not agent_data_path.exists():
        if character_path.exists():
            raise FileNotFoundError(
                f"Character profiles exist but their agent data is missing: "
                f"{agent_data_path}"
            )
        generate_and_save_population(NUM_AGENTS_PER_GROUP, is_control)

    all_chara = process_agent_descriptions(NUM_AGENTS_PER_GROUP, is_control)

    if all_chara is None:
        raise RuntimeError(f"Failed to build agent profiles for group: {group}")
    if len(all_chara) != NUM_AGENTS_PER_GROUP:
        raise ValueError(
            f"Expected {NUM_AGENTS_PER_GROUP} profiles for {group}, "
            f"found {len(all_chara)}."
        )

    all_ids = list(range(1, NUM_AGENTS_PER_GROUP + 1))
    return all_chara, all_ids


# =========================
# Group helpers
# =========================
def parse_group(group: str) -> tuple[str, str]:
    """Split a group name into nudge type and intensity."""
    if group == "control":
        return "control", "none"

    parts = group.split("_")
    if len(parts) != 2:
        raise ValueError(f"Invalid group name: {group}")

    nudge_type, intensity = parts
    if nudge_type not in {"information", "assistant", "structure"}:
        raise ValueError(f"Invalid nudge type in group: {group}")
    if intensity not in {"high", "medium", "low"}:
        raise ValueError(f"Invalid intensity in group: {group}")

    return nudge_type, intensity


def should_apply_nudge(day: int, group: str) -> bool:
    """Return whether the group receives a nudge on the current day."""
    nudge_type, intensity = parse_group(group)

    if nudge_type == "control":
        return False
    if intensity == "high":
        return True
    if intensity == "medium":
        return day in {1, 4, 7, 10, 13}
    if intensity == "low":
        return day == 1
    return False


def get_nudge_text(group: str) -> str:
    """Return the original intervention text for the selected group."""
    nudge_type, _ = parse_group(group)

    if nudge_type == "control":
        return ""
    if nudge_type == "information":
        return INFO_NUDGE
    if nudge_type == "assistant":
        return ASSIS_NUDGE
    if nudge_type == "structure":
        return STRUC_NUDGE

    raise ValueError(f"Unknown group: {group}")


def get_nudge_status_text(day: int, group: str) -> str:
    """Return the original treatment text on scheduled nudge days."""
    return get_nudge_text(group) if should_apply_nudge(day, group) else ""


def get_nudge_label(group: str) -> tuple[str, str]:
    nudge_type, _ = parse_group(group)

    if nudge_type == "control":
        return "no nudge", "none"
    if nudge_type == "information":
        return "information nudge", INFO_NUDGE
    if nudge_type == "assistant":
        return "assistant nudge", ASSIS_NUDGE
    if nudge_type == "structure":
        return "structure nudge", STRUC_NUDGE

    return "unknown", "none"


def build_posttest_questionnaire(group: str) -> str:
    """Build the posttest questionnaire, adding nudge-specific items when needed."""
    nudge_type, _ = parse_group(group)

    parts = [
        "Rate all items based on your study experience.",
        ENV_POSTTEST_BLOCK,
    ]

    if nudge_type == "information":
        parts.extend(
            [
                INFO_NUDGE_POSTTEST_BLOCK,
            ]
        )
    elif nudge_type == "assistant":
        parts.extend(
            [
                ASSIS_NUDGE_POSTTEST_BLOCK,
            ]
        )
    elif nudge_type == "structure":
        parts.extend(
            [
                STRUC_NUDGE_POSTTEST_BLOCK,
            ]
        )

    if nudge_type == "control":
        parts.append(
            "Reply only with a compact JSON array of 23 integers in q1-q23 order."
        )
    else:
        parts.append(
            "Reply only with a compact JSON array of 26 integers: q1-q23, then n1-n3."
        )

    return "\n".join(parts)


# =========================
# Choice helpers
# =========================
CHOICE_SCORE_MAP = {
    "A": 1.0,
    "B": 0.0,
    "C": -1.0,
}


def sample_daily_tasks(rng: random.Random, questions_per_day: int) -> list[str]:
    """Randomly assign number or slider tasks for one day."""
    return [rng.choice(["number", "slider"]) for _ in range(questions_per_day)]


def summarize_attitude(avg_score: float) -> str:
    """Convert average choice score into a short semantic memory summary."""
    if avg_score >= 0.5:
        return "You hold a relatively strong positive attitude toward the environmentally friendly option and often prefer it."
    if avg_score <= -0.3:
        return "You tend to avoid the environmentally friendly option and show relatively low support for it."
    return "You hold a relatively mixed or neutral attitude toward the environmentally friendly option."


# =========================
# Memory for day decisions
# =========================
def build_day_memory_text(
    data_store: Dict[int, list[Dict[str, Any]]],
    agent_id: int,
    current_day: int,
    group: str,
) -> Optional[str]:
    historical_choices = data_store.get(agent_id, [])
    if current_day == 1:
        return None
    if not historical_choices:
        return None

    memory_text = "\n[Event Memory]\n"

    # Keep only the most recent seven prior days as short-term memory.
    recent_days = [entry for entry in historical_choices if entry["day"] < current_day][
        -7:
    ]
    if recent_days:
        recent_summaries = []
        for entry in recent_days:
            day = entry["day"]
            choices = entry.get("parsed_choices", [])
            choice_text = ",".join(choices) if choices else "unknown"
            if group == "control":
                recent_summaries.append(f"D{day}={choice_text}")
            else:
                nudge_applied = entry.get("nudge_applied", False)
                recent_summaries.append(
                    f"D{day}={choice_text};nudge={'yes' if nudge_applied else 'no'}"
                )
        memory_text += "Recent: " + " | ".join(recent_summaries) + "\n"

    # Map choices to numeric scores to summarize long-term behavioral tendency.
    all_scores = []
    for entry in historical_choices:
        for ch in entry.get("parsed_choices", []):
            if ch in CHOICE_SCORE_MAP:
                all_scores.append(CHOICE_SCORE_MAP[ch])

    if all_scores and current_day >= 7:
        memory_text += "[Semantic Memory]\n"
        overall_avg = mean(all_scores)
        memory_text += summarize_attitude(overall_avg) + "\n"

        memory_text += "[Procedural Memory]\n"
        long_avg = mean(all_scores)
        variance = sum((x - long_avg) ** 2 for x in all_scores) / len(all_scores)
        memory_text += (
            "Repeated choices may have formed a habit. "
            f"Pro-environmental mean={long_avg:.3f} (-1 to 1); "
            f"variance={variance:.3f}.\n"
        )

    memory_text += "\n" + MEMORY_SUFFIX
    return memory_text


# =========================
# Memory for posttest
# =========================
def build_posttest_memory_text(
    data_store: Dict[int, list[Dict[str, Any]]],
    agent_id: int,
    group: str,
) -> str:
    # Posttest memory summarizes the full decision history.
    historical_choices = data_store.get(agent_id, [])
    nudge_name, nudge_text = get_nudge_label(group)
    nudge_type, _ = parse_group(group)

    memory_text = "\n"

    memory_text += "[Scenario]\n"
    memory_text += SCENARIO_TEXT + "\n"

    memory_text += "[Task]\n"
    memory_text += BASE_CHOICE_BLOCK + "\n"

    if nudge_name != "no nudge":
        memory_text += "[Nudge Condition]\n"
        memory_text += f"Assigned group: {group}\n"
        memory_text += f"Nudge type: {nudge_name}\n"
        memory_text += f"Nudge content:\n{nudge_text}\n"
    memory_text += "[All Past Days]\n"
    if not historical_choices:
        memory_text += "None.\n"
    else:
        day_summaries = []
        for entry in sorted(historical_choices, key=lambda x: x["day"]):
            day = entry["day"]
            parsed_choices = entry.get("parsed_choices", [])
            choice_text = ",".join(parsed_choices) if parsed_choices else "unknown"
            if group == "control":
                day_summaries.append(f"D{day}={choice_text}")
            else:
                nudge_applied = entry.get("nudge_applied", False)
                day_summaries.append(
                    f"D{day}={choice_text};nudge={'yes' if nudge_applied else 'no'}"
                )
        memory_text += " | ".join(day_summaries) + "\n"

    all_scores = []
    for entry in historical_choices:
        for ch in entry.get("parsed_choices", []):
            if ch in CHOICE_SCORE_MAP:
                all_scores.append(CHOICE_SCORE_MAP[ch])
    if all_scores:
        long_avg = mean(all_scores)
        variance = sum((x - long_avg) ** 2 for x in all_scores) / len(all_scores)
        memory_text += (
            f"Pro-environmental mean={long_avg:.3f} (-1 to 1); "
            f"variance={variance:.3f}.\n"
        )

    return memory_text


# =========================
# Prompt builders
# =========================
def build_single_question_block(
    q_idx: int,
    task_name: str,
) -> str:
    """Identify the task used for one question without repeating its description."""
    task_label = "number" if task_name == "number" else "slider"
    return f"Q{q_idx}={task_label}"


def build_day_prompt(
    role: str,
    group: str,
    day: int,
    tasks: list[str],
    memory_text: Optional[str],
    include_task_content: bool = False,
) -> tuple[str, str]:
    question_blocks = [
        build_single_question_block(
            q_idx=i + 1,
            task_name=task_name,
        )
        for i, task_name in enumerate(tasks)
    ]

    user_parts = [
        SCENARIO_TEXT,
        f"Today is Day {day} of the study.",
    ]

    nudge_context = get_nudge_status_text(day, group)
    if nudge_context:
        user_parts.append(nudge_context)

    if memory_text:
        user_parts.append(memory_text)

    user_parts.append(BASE_CHOICE_BLOCK)
    user_parts.append(f"Make this choice {len(tasks)} times.")

    if include_task_content:
        task_descriptions = ["Task content:"]
        if "number" in tasks:
            task_descriptions.append(NUMBER_TASK_TEXT)
        if "slider" in tasks:
            task_descriptions.append(SLIDER_TASK_TEXT)
        task_descriptions.append("Assignments: " + "; ".join(question_blocks))
        user_parts.append("\n".join(task_descriptions))

    user_parts.append(
        f"Reply only with {len(tasks)} letters (A/B/C), one per question, "
        "with no separators."
    )

    return role, "\n".join(user_parts)


def build_posttest_prompt(
    role: str,
    group: str,
    memory_text: Optional[str],
) -> tuple[str, str]:
    parts = ["Complete the posttest based on your full study experience."]

    if memory_text:
        parts.append(memory_text)

    parts.append(build_posttest_questionnaire(group))
    return role, "\n".join(parts)


# =========================
# Parsing
# =========================
def parse_choices_from_response(response_text: str, expected_n: int = 5) -> list[str]:
    """Extract A/B/C choices from a model response."""
    compact_response = re.sub(r"\s+", "", response_text).upper()
    if re.fullmatch(rf"[ABC]{{{expected_n}}}", compact_response):
        return list(compact_response)

    lines = [
        line.strip().upper()
        for line in response_text.strip().splitlines()
        if line.strip()
    ]
    parsed = []

    for line in lines:
        if line in {"A", "B", "C"}:
            parsed.append(line)
            continue

        m = re.search(r"\b([ABC])\b", line)
        if m:
            parsed.append(m.group(1))

    if len(parsed) > expected_n:
        parsed = parsed[:expected_n]

    return parsed


def parse_posttest_json(response_text: str) -> Any:
    """Parse a JSON array/object, allowing extra text around it."""
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        m = re.search(r"(?:\[.*\]|\{.*\})", response_text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


def validate_choices(parsed: list[str], expected_n: int) -> bool:
    return len(parsed) == expected_n and all(ch in {"A", "B", "C"} for ch in parsed)


def coerce_int_in_range(value: Any, min_value: int, max_value: int) -> int:
    num = int(round(float(value)))
    num = max(min_value, min(max_value, num))
    return num


def validate_posttest_dict(parsed: Any, group: str) -> dict[str, Any]:
    """Validate an ordered array or legacy object and clamp legal ranges."""
    nudge_type, _ = parse_group(group)
    expected_keys = ENV_ITEM_KEYS + (
        [] if nudge_type == "control" else NUDGE_ITEM_KEYS
    )
    if isinstance(parsed, list):
        if len(parsed) != len(expected_keys):
            raise ValueError(
                f"Expected {len(expected_keys)} posttest values, got {len(parsed)}"
            )
        parsed = dict(zip(expected_keys, parsed))
    if not isinstance(parsed, dict):
        raise TypeError("Posttest response must be a JSON array or object")

    validated: dict[str, Any] = {}

    for key in ENV_ITEM_KEYS:
        if key not in parsed:
            raise ValueError(f"Missing key: {key}")
        validated[key] = coerce_int_in_range(parsed[key], 1, 5)

    if nudge_type != "control":
        for key in NUDGE_ITEM_KEYS:
            if key not in parsed:
                raise ValueError(f"Missing key: {key}")
            validated[key] = coerce_int_in_range(parsed[key], 1, 7)

    return validated


def record_used_default(record: dict[str, Any]) -> bool:
    """Read the fallback marker, treating a missing marker as incomplete."""
    marker = record.get("used_default", record.get("use-default"))
    return marker is not False


def is_complete_day_record(
    record: Any,
    day: int,
    questions_per_day: int,
) -> bool:
    if not isinstance(record, dict):
        return False
    try:
        record_day = int(record.get("day"))
    except (TypeError, ValueError):
        return False
    return (
        record_day == day
        and not record_used_default(record)
        and validate_choices(record.get("parsed_choices", []), questions_per_day)
    )


def is_complete_posttest_record(record: Any, group: str) -> bool:
    if not isinstance(record, dict) or record_used_default(record):
        return False
    try:
        validate_posttest_dict(record.get("items", {}), group)
    except (TypeError, ValueError):
        return False
    return True


def load_json_if_present(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[Warning] Cannot read existing result {path}: {exc}")
        return default


def load_existing_group_results(
    group_dir: Path,
    all_ids: List[int],
    group: str,
    num_days: int,
    questions_per_day: int,
) -> tuple[Dict[int, List[Dict[str, Any]]], Dict[int, Dict[str, Any]]]:
    """Load successful records and discard only target records that need rerunning."""
    data_store: Dict[int, List[Dict[str, Any]]] = {}

    for agent_id in all_ids:
        raw_records = load_json_if_present(group_dir / f"{agent_id}.json", [])
        if not isinstance(raw_records, list):
            print(
                f"[Warning] Expected a list in {group_dir / f'{agent_id}.json'}; "
                "the agent will be resumed from its valid saved records."
            )
            raw_records = []

        retained_records: List[Dict[str, Any]] = []
        # Preserve records outside the requested range. Within the requested range,
        # retain exactly one successful record per day and rerun everything else.
        for record in raw_records:
            if not isinstance(record, dict):
                continue
            try:
                record_day = int(record.get("day"))
            except (TypeError, ValueError):
                continue
            if record_day < 1 or record_day > num_days:
                retained_records.append(record)

        for day in range(1, num_days + 1):
            completed_record = next(
                (
                    record
                    for record in raw_records
                    if is_complete_day_record(record, day, questions_per_day)
                ),
                None,
            )
            if completed_record is not None:
                retained_records.append(completed_record)

        data_store[agent_id] = sorted(
            retained_records,
            key=lambda record: int(record.get("day", 0)),
        )

    raw_posttest = load_json_if_present(group_dir / "posttest.json", {})
    if not isinstance(raw_posttest, dict):
        print(
            f"[Warning] Expected an object in {group_dir / 'posttest.json'}; "
            "invalid posttest data will be rerun when requested."
        )
        raw_posttest = {}

    posttest_store: Dict[int, Dict[str, Any]] = {}
    for agent_id in all_ids:
        record = raw_posttest.get(str(agent_id), raw_posttest.get(agent_id))
        if is_complete_posttest_record(record, group):
            posttest_store[agent_id] = record

    return data_store, posttest_store


def group_results_complete(
    data_store: Dict[int, List[Dict[str, Any]]],
    posttest_store: Dict[int, Dict[str, Any]],
    all_ids: List[int],
    group: str,
    num_days: int,
    questions_per_day: int,
    run_posttest: bool,
) -> bool:
    for agent_id in all_ids:
        records = data_store.get(agent_id, [])
        for day in range(1, num_days + 1):
            if not any(
                is_complete_day_record(record, day, questions_per_day)
                for record in records
            ):
                return False

        if run_posttest and not is_complete_posttest_record(
            posttest_store.get(agent_id), group
        ):
            return False

    return True


# =========================
# Posttest scoring
# =========================
def reverse_5pt(x: float) -> float:
    return 6.0 - x


def round3(x: float) -> float:
    return round(float(x), 3)


def compute_env_attitude_from_items(parsed: dict[str, Any]) -> float:
    """Compute posttest environmental attitude with reverse-coded items."""
    odd_items = [1, 3, 5, 7, 9, 11, 13, 15]
    even_reverse_items = [2, 4, 6, 8, 10, 12, 14]

    vals = []
    for i in odd_items:
        vals.append(float(parsed[f"q{i}"]))
    for i in even_reverse_items:
        vals.append(reverse_5pt(float(parsed[f"q{i}"])))

    return round3(sum(vals) / 15.0)


def compute_env_self_efficacy_from_items(parsed: dict[str, Any]) -> float:
    return round3(float(parsed["q17"]))


def compute_env_motivation_from_items(parsed: dict[str, Any]) -> float:
    """Compute a weighted environmental motivation score from posttest items."""
    q18 = float(parsed["q18"])
    q19 = float(parsed["q19"])
    q20 = float(parsed["q20"])
    q21 = float(parsed["q21"])
    q22 = float(parsed["q22"])
    q23 = float(parsed["q23"])

    score = 2.0 * q18 + 1.0 * q19 + 0.5 * q20 - 0.5 * q21 - 1.0 * q22 - 2.0 * q23
    return round3(score)


def compute_nudge_scores_from_items(
    parsed: dict[str, Any], group: str
) -> dict[str, Optional[float]]:
    nudge_type, _ = parse_group(group)

    if nudge_type == "control":
        return {
            "nudge_acceptance_post": None,
            "nudge_reactance_post": None,
            "nudge_fatigue_post": None,
        }

    return {
        "nudge_acceptance_post": round3(float(parsed["n1"])),
        "nudge_reactance_post": round3(float(parsed["n2"])),
        "nudge_fatigue_post": round3(float(parsed["n3"])),
    }


def compute_posttest_scores(parsed: dict[str, Any], group: str) -> dict[str, Any]:
    result = {
        "env_attitude_post": compute_env_attitude_from_items(parsed),
        "env_self_efficacy_post": compute_env_self_efficacy_from_items(parsed),
        "env_motivation_post": compute_env_motivation_from_items(parsed),
    }
    result.update(compute_nudge_scores_from_items(parsed, group))
    return result


# =========================
# Model call + parse retry
# =========================
def get_res_dict(
    role_prompt: str,
    user_prompt: str,
    model_name: str,
    temperature: float,
) -> dict[str, Any]:
    response = get_res(role_prompt, user_prompt, model_name, temperature)

    if isinstance(response, dict):
        return response

    raise TypeError(
        f"Unsupported response type from get_res: {type(response).__name__}"
    )


def call_and_parse_with_retries(
    role_prompt: str,
    user_prompt: str,
    model_name: str,
    temperature: float,
    max_attempts: int,
    parse_fn: Callable[[str], Any],
    validate_fn: Callable[[Any], Any],
    default_parsed: Any,
    default_output_text: str,
) -> Tuple[dict[str, Any], Any, bool, int]:
    # Retry model calls until parsing and validation both succeed.
    last_response_dict: Optional[dict[str, Any]] = None

    for attempt in range(1, max_attempts + 1):
        try:
            response_dict = get_res_dict(
                role_prompt=role_prompt,
                user_prompt=user_prompt,
                model_name=model_name,
                temperature=temperature,
            )
            last_response_dict = response_dict
            output_text = str(response_dict.get("output", "")).strip()

            parsed = parse_fn(output_text)
            validated = validate_fn(parsed)

            return response_dict, validated, False, attempt

        except Exception as exc:
            continue

    # If all attempts fail, keep the last raw response but use validated defaults.
    fallback_response = last_response_dict or {
        "input": role_prompt + user_prompt,
        "output": default_output_text,
    }

    return fallback_response, default_parsed, True, max_attempts


# =========================
# Core experiment
# =========================
def process_agent_one_day(
    agent_index: int,
    agent_id: int,
    role: str,
    day: int,
    model_name: str,
    temperature: float,
    max_attempts: int,
    group: str,
    questions_per_day: int,
    data_store: Dict[int, List[Dict[str, Any]]],
    seed: int,
    include_task_content: bool,
) -> int:
    # Use a deterministic per-agent/day seed so task sampling is reproducible.
    rng = random.Random(seed + agent_index * 10000 + day)
    tasks = sample_daily_tasks(rng, questions_per_day)
    memory_text = build_day_memory_text(
        data_store=data_store,
        agent_id=agent_id,
        current_day=day,
        group=group,
    )

    role_prompt, user_prompt = build_day_prompt(
        role=role,
        group=group,
        day=day,
        tasks=tasks,
        memory_text=memory_text,
        include_task_content=include_task_content,
    )

    default_choices = ["C"] * questions_per_day
    default_output = "".join(default_choices)

    # Generate and validate all choices for the current day in one response.
    response, parsed_choices, used_default, attempts_used = call_and_parse_with_retries(
        role_prompt=role_prompt,
        user_prompt=user_prompt,
        model_name=model_name,
        temperature=temperature,
        max_attempts=max_attempts,
        parse_fn=lambda text: parse_choices_from_response(
            text, expected_n=questions_per_day
        ),
        validate_fn=lambda parsed: (
            parsed
            if validate_choices(parsed, questions_per_day)
            else (_ for _ in ()).throw(
                ValueError(
                    f"Expected {questions_per_day} parsed choices, got {len(parsed)}"
                )
            )
        ),
        default_parsed=default_choices,
        default_output_text=default_output,
    )

    input_text = response.get("input", "")
    output_text = response.get("output", default_output)
    reason_output = response.get("reason_output", "")

    item = {
        "day": day,
        "tasks": tasks,
        "input": input_text,
        "raw_response": output_text,
        "reason_output": reason_output,
        "parsed_choices": parsed_choices,
        "nudge_applied": should_apply_nudge(day, group),
        "group": group,
        "attempts_used": attempts_used,
        "used_default": used_default,
    }

    if agent_id not in data_store:
        data_store[agent_id] = []
    data_store[agent_id].append(item)
    data_store[agent_id].sort(key=lambda record: int(record.get("day", 0)))
    return agent_id


def process_agent_posttest(
    agent_id: int,
    role: str,
    model_name: str,
    temperature: float,
    max_attempts: int,
    group: str,
    data_store: Dict[int, List[Dict[str, Any]]],
    posttest_store: Dict[int, Dict[str, Any]],
) -> int:
    memory_text = build_posttest_memory_text(
        data_store=data_store,
        agent_id=agent_id,
        group=group,
    )

    role_prompt, user_prompt = build_posttest_prompt(
        role=role,
        group=group,
        memory_text=memory_text,
    )

    nudge_type, _ = parse_group(group)
    default_posttest = (
        DEFAULT_POSTTEST_RESPONSES_CONTROL.copy()
        if nudge_type == "control"
        else DEFAULT_POSTTEST_RESPONSES_NUDGE.copy()
    )
    default_output = json.dumps(
        list(default_posttest.values()),
        ensure_ascii=False,
        separators=(",", ":"),
    )

    # Parse the posttest JSON and fall back to neutral defaults if needed.
    response, parsed_posttest_items, used_default, attempts_used = (
        call_and_parse_with_retries(
            role_prompt=role_prompt,
            user_prompt=user_prompt,
            model_name=model_name,
            temperature=temperature,
            max_attempts=max_attempts,
            parse_fn=parse_posttest_json,
            validate_fn=lambda parsed: validate_posttest_dict(parsed, group),
            default_parsed=default_posttest,
            default_output_text=default_output,
        )
    )

    input_text = response.get("input", "")
    output_text = response.get("output", default_output)

    computed_scores = compute_posttest_scores(parsed_posttest_items, group)

    result = {
        "items": parsed_posttest_items,
        **computed_scores,
        "input": input_text,
        "raw_response": output_text,
        "group": group,
        "attempts_used": attempts_used,
        "used_default": used_default,
    }

    posttest_store[agent_id] = result
    return agent_id


def agent_experiment(
    model_name: str,
    temperature: float,
    num_threads: int,
    all_chara: list[str],
    all_ids: List[int],
    max_attempts: int,
    group: str,
    num_days: int,
    questions_per_day: int,
    seed: int,
    run_posttest: bool = False,
    include_task_content: bool = False,
    data_store: Optional[Dict[int, List[Dict[str, Any]]]] = None,
    posttest_store: Optional[Dict[int, Dict[str, Any]]] = None,
    out_dir: Optional[Path] = None,
) -> tuple[Dict[int, List[Dict[str, Any]]], Dict[int, Dict[str, Any]]]:
    if data_store is None:
        data_store = {}
    if posttest_store is None:
        posttest_store = {}

    if len(all_chara) != len(all_ids):
        raise ValueError("all_chara and all_ids must have the same length.")

    # Preserve both participant IDs and textual profiles during parallel execution.
    agent_records = [
        (agent_index, agent_id, role)
        for agent_index, (agent_id, role) in enumerate(zip(all_ids, all_chara), start=1)
    ]

    # Run daily choices first, then conduct the posttest after all days finish.
    for day in range(1, num_days + 1):
        pending_agent_records = [
            (agent_index, agent_id, role)
            for agent_index, agent_id, role in agent_records
            if not any(
                is_complete_day_record(record, day, questions_per_day)
                for record in data_store.get(agent_id, [])
            )
        ]
        if not pending_agent_records:
            print(f"{group} | Day {day}: already complete, skipping.")
            continue

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(
                    process_agent_one_day,
                    agent_index,
                    agent_id,
                    role,
                    day,
                    model_name,
                    temperature,
                    max_attempts,
                    group,
                    questions_per_day,
                    data_store,
                    seed,
                    include_task_content,
                )
                for agent_index, agent_id, role in pending_agent_records
            ]

            with tqdm(
                total=len(futures), desc=f"{group} | Day {day}", ncols=100
            ) as pbar:
                for future in as_completed(futures):
                    agent_id = future.result()
                    if out_dir is not None:
                        save_agent_results(
                            out_dir,
                            agent_id,
                            data_store.get(agent_id, []),
                        )
                    pbar.update(1)

    if run_posttest:
        pending_posttest_records = [
            (agent_id, role)
            for _, agent_id, role in agent_records
            if all(
                any(
                    is_complete_day_record(record, day, questions_per_day)
                    for record in data_store.get(agent_id, [])
                )
                for day in range(1, num_days + 1)
            )
            and not is_complete_posttest_record(
                posttest_store.get(agent_id), group
            )
        ]
        if not pending_posttest_records:
            print(f"{group} | Posttest: already complete, skipping.")
            return data_store, posttest_store

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(
                    process_agent_posttest,
                    agent_id,
                    role,
                    model_name,
                    temperature,
                    max_attempts,
                    group,
                    data_store,
                    posttest_store,
                )
                for agent_id, role in pending_posttest_records
            ]

            with tqdm(
                total=len(futures), desc=f"{group} | Posttest", ncols=100
            ) as pbar:
                for future in as_completed(futures):
                    future.result()
                    if out_dir is not None:
                        save_posttest_results(out_dir, posttest_store)
                    pbar.update(1)

    return data_store, posttest_store


# =========================
# Save helpers
# =========================
def atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON through a temporary file so interrupted runs keep valid data."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    temporary_path.replace(path)


def save_agent_results(
    out_dir: Path,
    agent_id: int,
    records: List[Dict[str, Any]],
) -> None:
    atomic_write_json(out_dir / f"{agent_id}.json", records)


def save_posttest_results(
    out_dir: Path,
    posttest_store: Dict[int, Dict[str, Any]],
) -> None:
    atomic_write_json(out_dir / "posttest.json", posttest_store)


def save_final_results(
    out_dir: Path,
    data_store: Dict[int, List[Dict[str, Any]]],
    posttest_store: Dict[int, Dict[str, Any]],
    save_posttest: bool = True,
) -> None:
    """Save per-agent daily records and the shared posttest file."""
    for agent_id, records in data_store.items():
        save_agent_results(out_dir, agent_id, records)

    if save_posttest:
        save_posttest_results(out_dir, posttest_store)


# =========================
# Main
# =========================
def main() -> None:
    parser = get_argument_parser()
    args = parser.parse_args()

    model_name = args.model_name
    temperature = args.temperature
    num_threads = args.num_threads
    max_attempts = args.max_attempts
    num_days = 2 if args.two_days else args.num_days
    questions_per_day = args.questions_per_day
    include_task_content = args.include_task_content
    selected_groups = GROUPS if args.group == "all" else [args.group]
    output_dir = Path(args.output_dir).resolve()
    seed = args.seed
    original_cwd = Path.cwd()
    run_posttest = args.posttest and not args.two_days
    expected_agent_ids = list(range(1, NUM_AGENTS_PER_GROUP + 1))

    if args.posttest and args.two_days:
        print("Posttest is disabled when --two_days is used.")

    if num_days <= 0:
        raise ValueError(f"num_days must be positive, got {num_days}")
    if questions_per_day <= 0:
        raise ValueError(
            f"questions_per_day must be positive, got {questions_per_day}"
        )

    try:
        for group in selected_groups:
            group_dir = output_dir / f"{model_name}_res" / group
            group_dir.mkdir(parents=True, exist_ok=True)

            data_store, posttest_store = load_existing_group_results(
                group_dir=group_dir,
                all_ids=expected_agent_ids,
                group=group,
                num_days=num_days,
                questions_per_day=questions_per_day,
            )
            if group_results_complete(
                data_store=data_store,
                posttest_store=posttest_store,
                all_ids=expected_agent_ids,
                group=group,
                num_days=num_days,
                questions_per_day=questions_per_day,
                run_posttest=run_posttest,
            ):
                print(f"All requested results already exist for group: {group}")
                continue

            print(f"Processing group: {group}")
            os.chdir(group_dir)
            try:
                all_chara, all_ids = build_all_chara(group)

                data_store, posttest_store = agent_experiment(
                    model_name=model_name,
                    temperature=temperature,
                    num_threads=num_threads,
                    all_chara=all_chara,
                    all_ids=all_ids,
                    max_attempts=max_attempts,
                    group=group,
                    num_days=num_days,
                    questions_per_day=questions_per_day,
                    seed=seed,
                    run_posttest=run_posttest,
                    include_task_content=include_task_content,
                    data_store=data_store,
                    posttest_store=posttest_store,
                    out_dir=group_dir,
                )

                save_final_results(
                    group_dir,
                    data_store,
                    posttest_store,
                    save_posttest=run_posttest,
                )
            finally:
                os.chdir(original_cwd)

            print(f"Finished group: {group}")
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()
