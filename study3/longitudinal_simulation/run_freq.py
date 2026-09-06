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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Allow imports from the project root when this script is run from a subdirectory.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
DEFAULT_FREQUENCY = 10
# DEFAULT_FREQUENCY = 6
DEFAULT_START_ID = 0
DEFAULT_NUM_CHARS = 50

CSV_RELATIVE_PATH = Path("..") / ".." / "prompt" / "meta_new_list.csv"
RES_DIR_NAME = "res_freq"

def get_profile_base_path(model_name: str) -> Path:
    """Return the matching Study 1 output directory for the selected model."""
    return (
        PROJECT_ROOT
        / "study1"
        / "nudge_replication"
        / "res"
        / f"{model_name}_res"
    )

# Studies selected for the scheduled-nudge follow-up simulation.
LONG_TERM_FAILING_STUDIES =  ['Arana_1', 'Bamberg_2', 'Dickerson_2', 'Everett_1', 'Handgraaf_1', 'Kesternich_1', 'Klotz_1', 'Lofgren_1', 'Lofgren_2', 'Marek_1', 'Meng_1', 'Nelson_1', 'Nelson_2', 'Nelson_3', 'Yeomans_1']

def get_argument_parser() -> argparse.ArgumentParser:
    """Build and return the command-line argument parser."""
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
    parser.add_argument("--frequency", type=int, default=DEFAULT_FREQUENCY)
    parser.add_argument("--start_id", type=int, default=DEFAULT_START_ID)
    parser.add_argument(
        "--num_chars",
        type=int,
        default=DEFAULT_NUM_CHARS,
    )
    return parser


def get_behavior_prompt(
    exp_data: pd.Series,
    round_num: int,
    historical_text: str,
    frequency: int,
    group: str = "intervention",
) -> str:
    """Build one round using only the study's original treatment text."""

    # Day 1 inherits the first intervention; apply it again every `frequency` days.
    is_intervention_round = (
        group != "control" and (round_num - 1) % frequency == 0
    )
    scenario_key = (
        "intervention_group_scenario_prompt"
        if is_intervention_round
        else "control_group_scenario_prompt"
    )

    prompt = (
        f"{exp_data[scenario_key]}\n"
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
    frequency: int,
    current_agent_profiles: list[dict[str, Any]],
) -> tuple[int, dict[str, Any] | None]:
    """
    Process a single character for one round.

    Generate a response based on the role, prompt, and memory text.
    """
    # Build memory from previous rounds, aligned with the scheduled nudge frequency.
    # Before the second scheduled nudge, use the same memory construction as long.
    # The frequency-specific memory window begins only after the first cycle.
    memory_frequency = 0 if round_num <= frequency else frequency
    historical_text = build_memory_text(
        data_store,
        cha_num,
        round_num,
        exp_data,
        freq=memory_frequency,
        nudge=False,
        procedural_memory_start_day=7,
    )

    exp = get_behavior_prompt(
        exp_data=exp_data,
        round_num=round_num,
        historical_text=historical_text,
        frequency=frequency,
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

    store_key = cha_num - 1
    # Update the agent profile using validated trait values returned by the model.
    previous_agent = current_agent_profiles[store_key]
    updated_agent = validate_and_update_agent_traits(previous_agent, result)
    data_store.setdefault(store_key, []).append(result)
    return store_key, updated_agent


def load_day1_results(
    study_id: str,
    num_chars: int,
    profile_base_path: Path,
) -> dict[int, list[Any]]:
    """Load Day 1 directly from the short-term intervention result."""
    result_path = profile_base_path / str(study_id) / "intervention.json"

    if not result_path.exists():
        raise FileNotFoundError(f"Short-term result not found: {result_path}")

    with result_path.open("r", encoding="utf-8") as f:
        results = json.load(f)

    if len(results) < num_chars:
        raise ValueError(
            f"Study {study_id} intervention: short-term result contains only "
            f"{len(results)} agents, but num_chars={num_chars}"
        )

    data_store: dict[int, list[Any]] = {}
    for i in range(num_chars):
        result = copy.deepcopy(results[i])
        result["round"] = 1
        data_store[i] = [result]

    return data_store


def initialize_day1(
    study_id: str,
    num_chars: int,
    profile_base_path: Path,
) -> None:
    """Write the short-term intervention result as longitudinal Day 1."""
    data_store = load_day1_results(
        study_id=study_id,
        num_chars=num_chars,
        profile_base_path=profile_base_path,
    )

    for cha_num, data in data_store.items():
        filename = f"{cha_num + 1}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


def agent_experiment(
    model_name: str,
    temperature: float,
    num_threads: int,
    max_attempts: int,
    exp_data: pd.Series,
    num_rounds: int,
    frequency: int,
    study_group_dir: Path,
    effective_n: int,
    start_round: int = 2,
) -> None:
    """Run the multi-round experiment for all characters in one study."""
    # Load inherited Day 1 and any checkpointed later rounds.
    data_store: dict[int, list[Any]] = load_existing_data_store(study_group_dir)

    for round_num in range(start_round, num_rounds + 1):
        current_characters = load_round_characters(
            study_group_dir=study_group_dir,
            group="intervention",
            effective_n=effective_n,
            round_num=round_num,
        )

        current_agent_profiles = get_agent_profiles_for_round(
            study_group_dir=study_group_dir,
            group="intervention",
            effective_n=effective_n,
            round_num=round_num,
        )

        # Start from the previous profile state and replace only agents updated this round.
        updated_profiles_for_round = copy.deepcopy(current_agent_profiles)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
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
                    frequency,
                    current_agent_profiles,
                )
                for cha_num, role in enumerate(current_characters, start=1)
            ]

            with tqdm(
                total=len(futures),
                desc=f"Processing day {round_num}",
                ncols=100,
            ) as pbar:
                for future in as_completed(futures):
                    store_key, updated_agent = future.result()
                    if updated_agent is not None:
                        updated_profiles_for_round[store_key] = updated_agent
                    pbar.update(1)

        # Save round-specific profiles so later prompts reflect updated traits.
        save_round_profile_snapshots(
            study_group_dir=study_group_dir,
            group="intervention",
            effective_n=effective_n,
            round_num=round_num,
            updated_agents=updated_profiles_for_round,
            generate_description_func=generate_description,
        )

        # Periodically persist generated responses to support recovery and partial runs
        save_checkpoint(round_num, num_rounds, data_store)


def main() -> None:
    """Parse arguments, load study data, and run experiments study by study."""
    parser = get_argument_parser()
    args = parser.parse_args()
    print(args)

    model_name = args.model_name
    temperature = args.temperature
    num_threads = args.num_threads
    max_attempts = args.max_attempts
    num_rounds = 2 if args.two_days else args.num_rounds
    frequency = args.frequency
    start_id = args.start_id
    num_chars = args.num_chars
    profile_base_path = get_profile_base_path(model_name)

    if num_chars <= 0:
        raise ValueError(f"num_chars must be positive, got {num_chars}")
    if num_rounds <= 0:
        raise ValueError(f"num_rounds must be positive, got {num_rounds}")
    if frequency <= 0:
        raise ValueError(f"frequency must be positive, got {frequency}")

    original_cwd = Path.cwd()
    csv_path = (original_cwd / CSV_RELATIVE_PATH).resolve()

    try:
        df = pd.read_csv(csv_path)
        # Restrict the run to studies selected for the scheduled-nudge experiment.
        df = df[df["study_id"].isin(LONG_TERM_FAILING_STUDIES)]
        print(f"Number of selected studies: {len(df)}")
        print("Selected study_ids:")
        print(df["study_id"].unique())
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"CSV file not found: {csv_path}") from exc
    except UnicodeDecodeError as exc:
        raise UnicodeDecodeError(
            exc.encoding,
            exc.object,
            exc.start,
            exc.end,
            f"Failed to decode CSV file: {csv_path}. {exc.reason}",
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to load CSV file: {csv_path}") from exc

    output_root = (
        original_cwd / RES_DIR_NAME / f"{model_name}_res" / f"{frequency}_freq"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    for index, row in df.iterrows():
        if index < start_id:
            continue

        study_id = row["study_id"]
        print(f"Processing study_id {study_id}...")

        study_dir = output_root / str(study_id)
        study_dir.mkdir(exist_ok=True)
        get_local_profile_dir(study_dir)

        # Detect completed rounds so the script can skip or resume each study.
        existing_max_round = get_existing_max_round(study_dir)

        if existing_max_round >= num_rounds:
            print(
                f"Study_id {study_id} intervention: already finished "
                f"({existing_max_round} rounds). Skip."
            )
            continue

        if existing_max_round > 0:
            print(
                f"Study_id {study_id} intervention: resume from round "
                f"{existing_max_round + 1}"
            )
        else:
            print(f"Study_id {study_id} intervention: start from round 1")

        try:
             # Each study writes outputs inside its own result directory.
            os.chdir(study_dir)
            # Copy and trim the short-term profiles into the current result directory.
            _, _, prepared_agents = prepare_local_profile_files(
                profile_base_relative_path=profile_base_path,
                study_group_dir=study_dir,
                study_id=study_id,
                group="intervention",
                n_control=num_chars,
                n_intervention=num_chars,
                num_chars=num_chars,
            )
            effective_n = min(len(prepared_agents), num_chars)

            if effective_n != num_chars:
                raise ValueError(
                    f"Study {study_id} intervention: expected {num_chars} agents, "
                    f"but prepared {effective_n}"
                )

            if existing_max_round == 0:
                initialize_day1(
                    study_id=study_id,
                    num_chars=num_chars,
                    profile_base_path=profile_base_path,
                )
                existing_max_round = 1

            if num_rounds == 1:
                continue

            start_round = max(existing_max_round + 1, 2)

            agent_experiment(
                model_name=model_name,
                temperature=temperature,
                num_threads=num_threads,
                max_attempts=max_attempts,
                exp_data=row,
                num_rounds=num_rounds,
                frequency=frequency,
                study_group_dir=study_dir,
                effective_n=effective_n,
                start_round=start_round,
            )
        except Exception as e:
            print(f"Error in study {study_id}: {e}")

    print("All studies processed.")


if __name__ == "__main__":
    main()
