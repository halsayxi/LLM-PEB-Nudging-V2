from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

# Add the project root to sys.path so shared utility modules can be imported.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils import (
    get_fixed_response_with_traits,
    save_checkpoint,
    get_existing_max_round,
    load_existing_data_store,
)

from agent_data.attribute_to_description import generate_description

from memory import (
    build_trait_output_instruction,
    get_agent_profiles_for_round,
    get_local_profile_dir,
    load_round_characters,
    prepare_local_profile_files,
    save_round_profile_snapshots,
    validate_and_update_agent_traits,
    build_memory_text,
)


DEFAULT_MODEL_NAME = "deepseek-v4-flash"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_NUM_THREADS = 120
DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_NUM_ROUNDS = 30
DEFAULT_START_ID = 0
DEFAULT_NUM_CHARS = 50

CSV_RELATIVE_PATH = Path("..") / ".." / "prompt" / "meta_new_list.csv"
RES_DIR_NAME = "res_long"

def get_profile_base_path(model_name: str) -> Path:
    """Return the matching Study 1 output directory for the selected model."""
    return (
        PROJECT_ROOT
        / "study1"
        / "nudge_replication"
        / "res"
        / f"{model_name}_res"
    )


def get_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--num_threads", type=int, default=DEFAULT_NUM_THREADS)
    parser.add_argument("--max_attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument("--num_rounds", type=int, default=DEFAULT_NUM_ROUNDS)
    parser.add_argument(
        "--two_days",
        action="store_true",
        help="Run only Days 1-2 (equivalent to --num_rounds 2).",
    )
    parser.add_argument("--start_id", type=int, default=DEFAULT_START_ID)
    parser.add_argument("--output_dir", type=str, default=RES_DIR_NAME)
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=["control", "intervention"],
        default=["control", "intervention"],
    )
    parser.add_argument(
        "--num_chars",
        type=int,
        default=DEFAULT_NUM_CHARS,
    )

    return parser


def get_behavior_prompt(
    exp_data: pd.Series,
    round_num: int,
    group: str,
    historical_text: str,
) -> str:
    """
    Build the prompt for one round.

    Day 1 is inherited from the short-term experiment.

    From Day 2 onward, both groups use the control scenario:
        control:
            Day 1  = short-term control
            Day 2+ = control scenario

        intervention:
            Day 1  = short-term intervention
            Day 2+ = control scenario
    """

    nudge_status = ""
    if group == "intervention":
        nudge_status = "You received an intervention nudge on Day 1. No intervention nudge is being applied today.\n"

    prompt = (
        f"{exp_data['control_group_scenario_prompt']}\n"
        f"{nudge_status}"
        f"{exp_data['response_instruction']} {exp_data['response_options']}"
    )
    if historical_text:
        prompt += (
            f"\n{historical_text}"
            "The memory is background context only. Reassess today's scenario "
            "independently. Do not copy the most recent response merely for consistency."
        )
    prompt += "\n" + build_trait_output_instruction()
    return prompt


def process_chara(
    cha_num: int,
    role: str,
    round_num: int,
    model_name: str,
    temperature: float,
    max_attempts: int,
    exp_data: pd.Series,
    data_store: dict[int, list[Any]],
    group: str,
    current_agent_profiles: list[dict[str, Any]],
) -> tuple[int, dict[str, Any] | None]:
    """
    Process a single character for Day 2+.

    Day 1 is loaded directly from the short-term experiment,
    so this function does not handle Day 1.
    """

    historical_text = build_memory_text(
        data_store,
        cha_num,
        round_num,
        exp_data,
        nudge=False,
        procedural_memory_start_day=7,
    )

    exp = get_behavior_prompt(
        exp_data=exp_data,
        round_num=round_num,
        group=group,
        historical_text=historical_text,
    )

    result = get_fixed_response_with_traits(
        role=role,
        exp=exp,
        model_name=model_name,
        temperature=temperature,
        cha_num=cha_num,
        max_attempts=max_attempts,
        is_binary=exp_data["binary_outcome"],
        round_num=round_num,
    )

    result["round"] = round_num

    store_key = cha_num - 1

    previous_agent = current_agent_profiles[store_key]

    updated_agent = validate_and_update_agent_traits(
        previous_agent,
        result,
    )

    data_store.setdefault(store_key, []).append(result)

    return store_key, updated_agent


def load_day1_results(
    study_id: str,
    group: str,
    num_chars: int,
    profile_base_path: Path,
) -> dict[int, list[Any]]:
    """
    Load Day 1 directly from the corresponding short-term result.

    The short-term result order is assumed to be consistent with
    the corresponding agent_data and character files.

    control:
        character_control_{num_chars}.json
        agent_data_control_{num_chars}.json
        control.json

    intervention:
        character_intervention_{num_chars}.json
        agent_data_intervention_{num_chars}.json
        intervention.json
    """

    result_path = (
        profile_base_path
        / str(study_id)
        / f"{group}.json"
    )

    if not result_path.exists():
        raise FileNotFoundError(
            f"Short-term result not found: {result_path}"
        )

    with open(result_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    if len(results) < num_chars:
        raise ValueError(
            f"Study {study_id} {group}: "
            f"short-term result contains only {len(results)} agents, "
            f"but num_chars={num_chars}"
        )

    data_store: dict[int, list[Any]] = {}

    for i in range(num_chars):
        result = copy.deepcopy(results[i])

        # Short-term result becomes Day 1 of the longitudinal experiment.
        result["round"] = 1

        data_store[i] = [result]

    return data_store


def initialize_day1(
    study_id: str,
    group: str,
    num_chars: int,
    profile_base_path: Path,
) -> None:
    """
    Initialize Day 1 directly from the short-term experiment.

    The short-term agent profile has already been copied by
    prepare_local_profile_files().

    No LLM call and no profile update are performed on Day 1.
    """

    data_store = load_day1_results(
        study_id=study_id,
        group=group,
        num_chars=num_chars,
        profile_base_path=profile_base_path,
    )

    # Write Day 1 directly into each agent's longitudinal result file.
    for cha_num, data in data_store.items():
        filename = f"{cha_num + 1}.json"

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=4,
            )


def agent_experiment(
    model_name: str,
    temperature: float,
    num_threads: int,
    max_attempts: int,
    exp_data: pd.Series,
    num_rounds: int,
    group: str,
    study_group_dir: Path,
    effective_n: int,
    start_round: int = 2,
) -> None:
    """
    Run Day 2+ of the longitudinal experiment for all characters.
    """

    data_store = load_existing_data_store(
        study_group_dir
    )

    for round_num in range(
        start_round,
        num_rounds + 1,
    ):
        current_characters = load_round_characters(
            study_group_dir=study_group_dir,
            group=group,
            effective_n=effective_n,
            round_num=round_num,
        )

        current_agent_profiles = get_agent_profiles_for_round(
            study_group_dir=study_group_dir,
            group=group,
            effective_n=effective_n,
            round_num=round_num,
        )

        updated_profiles_for_round = copy.deepcopy(
            current_agent_profiles
        )

        with ThreadPoolExecutor(
            max_workers=num_threads
        ) as executor:

            futures = [
                executor.submit(
                    process_chara,
                    cha_num,
                    role,
                    round_num,
                    model_name,
                    temperature,
                    max_attempts,
                    exp_data,
                    data_store,
                    group,
                    current_agent_profiles,
                )
                for cha_num, role in enumerate(
                    current_characters,
                    start=1,
                )
            ]

            with tqdm(
                total=len(futures),
                desc=f"{group} - Processing day {round_num}",
                ncols=100,
            ) as pbar:

                for future in as_completed(futures):
                    store_key, updated_agent = future.result()

                    if updated_agent is not None:
                        updated_profiles_for_round[
                            store_key
                        ] = updated_agent

                    pbar.update(1)

        # Day 2 generates round_2 profile,
        # Day 3 generates round_3 profile, etc.
        save_round_profile_snapshots(
            study_group_dir=study_group_dir,
            group=group,
            effective_n=effective_n,
            round_num=round_num,
            updated_agents=updated_profiles_for_round,
            generate_description_func=generate_description,
        )

        save_checkpoint(
            round_num,
            num_rounds,
            data_store,
        )


def run_group(
    model_name: str,
    temperature: float,
    num_threads: int,
    max_attempts: int,
    exp_data: pd.Series,
    num_rounds: int,
    study_id: str,
    group: str,
    study_group_dir: Path,
    num_chars: int,
) -> None:
    """
    Run one group independently.

    control:
        Day 1  = short-term control profile + control result
        Day 2+ = control scenario

    intervention:
        Day 1  = short-term intervention profile + intervention result
        Day 2+ = control scenario
    """

    study_group_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    get_local_profile_dir(
        study_group_dir
    )

    existing_max_round = get_existing_max_round(
        study_group_dir
    )

    if existing_max_round >= num_rounds:
        print(
            f"Study_id {study_id} {group}: "
            f"already finished ({existing_max_round} rounds). Skip."
        )
        return

    if existing_max_round > 0:
        print(
            f"Study_id {study_id} {group}: "
            f"resume from round {existing_max_round + 1}"
        )
    else:
        print(
            f"Study_id {study_id} {group}: start from round 1"
        )

    os.chdir(study_group_dir)

    profile_base_path = get_profile_base_path(model_name)

    # Copy the short-term agent profile into the longitudinal directory.
    _, _, prepared_agents = prepare_local_profile_files(
        profile_base_relative_path=profile_base_path,
        study_group_dir=study_group_dir,
        study_id=study_id,
        group=group,
        n_control=num_chars,
        n_intervention=num_chars,
        num_chars=num_chars,
    )

    effective_n = min(
        len(prepared_agents),
        num_chars,
    )

    if effective_n != num_chars:
        raise ValueError(
            f"Study {study_id} {group}: "
            f"expected {num_chars} agents, "
            f"but prepared {effective_n}"
        )

    # Fresh experiment:
    # first copy the short-term result into Day 1.
    if existing_max_round == 0:
        initialize_day1(
            study_id=study_id,
            group=group,
            num_chars=num_chars,
            profile_base_path=profile_base_path,
        )

        existing_max_round = 1

    if num_rounds == 1:
        return

    # Day 2+ or checkpoint resume.
    start_round = max(
        existing_max_round + 1,
        2,
    )

    agent_experiment(
        model_name=model_name,
        temperature=temperature,
        num_threads=num_threads,
        max_attempts=max_attempts,
        exp_data=exp_data,
        num_rounds=num_rounds,
        group=group,
        study_group_dir=study_group_dir,
        effective_n=effective_n,
        start_round=start_round,
    )


def main() -> None:
    parser = get_argument_parser()
    args = parser.parse_args()

    print(args)

    model_name = args.model_name
    temperature = args.temperature
    num_threads = args.num_threads
    max_attempts = args.max_attempts
    num_rounds = 2 if args.two_days else args.num_rounds
    start_id = args.start_id
    num_chars = args.num_chars
    output_dir = args.output_dir
    groups = args.groups

    if num_chars <= 0:
        raise ValueError(
            f"num_chars must be positive, got {num_chars}"
        )
    if num_rounds <= 0:
        raise ValueError(
            f"num_rounds must be positive, got {num_rounds}"
        )

    original_cwd = Path.cwd()

    csv_path = (
        original_cwd
        / CSV_RELATIVE_PATH
    ).resolve()

    try:
        df = pd.read_csv(csv_path)

        if "suitable_for_longitudinal" not in df.columns:
            raise KeyError(
                f"Column 'suitable_for_longitudinal' "
                f"not found in CSV: {csv_path}"
            )

        df = df[
            pd.to_numeric(
                df["suitable_for_longitudinal"],
                errors="coerce",
            )
            == 1
        ].copy()

    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}"
        ) from exc

    except UnicodeDecodeError as exc:
        raise UnicodeDecodeError(
            exc.encoding,
            exc.object,
            exc.start,
            exc.end,
            f"Failed to decode CSV file: {csv_path}. {exc.reason}",
        ) from exc

    except Exception as exc:
        raise RuntimeError(
            f"Failed to load CSV file: {csv_path}"
        ) from exc

    output_root = (
        original_cwd
        / output_dir
        / f"{model_name}_res"
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    for index, row in df.iterrows():

        if index < start_id:
            continue

        study_id = row["study_id"]

        print()
        print("=" * 80)
        print(f"Processing study_id {study_id}...")
        print("=" * 80)

        for group in groups:

            print()
            print(
                f"--- {study_id} | {group} | "
                f"num_chars={num_chars} ---"
            )

            study_group_dir = (
                output_root
                / str(study_id)
                / group
            )

            try:
                os.chdir(original_cwd)

                run_group(
                    model_name=model_name,
                    temperature=temperature,
                    num_threads=num_threads,
                    max_attempts=max_attempts,
                    exp_data=row,
                    num_rounds=num_rounds,
                    study_id=study_id,
                    group=group,
                    study_group_dir=study_group_dir,
                    num_chars=num_chars,
                )

            except Exception as e:
                print(
                    f"Error in study {study_id}, "
                    f"group {group}: {e}"
                )

            finally:
                os.chdir(original_cwd)

    print("All studies processed.")


if __name__ == "__main__":
    main()
