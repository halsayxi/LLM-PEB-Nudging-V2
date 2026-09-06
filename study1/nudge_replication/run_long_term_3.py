import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

# Add the project root to sys.path so utils can be imported reliably
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils import get_fixed_response
from study1.nudge_replication.long_term_common import (
    NUM_AGENTS_PER_GROUP,
    add_model_argument,
    build_group_profiles,
    group_results_complete,
    resolve_model_names,
)


DEFAULT_TEMPERATURE = 0.7
DEFAULT_NUM_THREADS = 30
DEFAULT_MAX_ATTEMPTS = 5

# Phase boundaries used to switch prompt framing across the experiment.
TOTAL_ROUNDS = 24
EARLY_REFERENCE_END_ROUND = 3
WEEK_PHASE_END_ROUND = 8
MONTH_PHASE_END_ROUND = 21
PRICING_PHASE_START_ROUND = 22

OUTPUT_ROOT_DIR = PROJECT_ROOT / "study1" / "nudge_replication" / "res_long_term"
LONG_TERM_SUBDIR = "long_term_3"


def get_argument_parser() -> argparse.ArgumentParser:
    """Create and return the command-line argument parser."""
    parser = argparse.ArgumentParser()
    add_model_argument(parser)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--num_threads", type=int, default=DEFAULT_NUM_THREADS)
    parser.add_argument("--max_attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_ROOT_DIR)
    return parser


def build_experiment_prompt(round_num: int, group: str, last: float | None) -> str:
    """Build the round-level experiment prompt prefix."""
    prompt = "You are participating in an water usage experiment. "

    if round_num <= WEEK_PHASE_END_ROUND:
        prompt += f"This is week {round_num}.\n"
    elif round_num <= MONTH_PHASE_END_ROUND:
        prompt += f"This is month {round_num - 6}.\n"
    else:
        prompt += f"This is quarter {round_num - 16}.\n"

    # Add marginal pricing information only in the final pricing phase.
    if round_num >= PRICING_PHASE_START_ROUND:
        prompt += (
            "We are now using marginal pricing for water consumption. "
            "This means that your water bill increases as you use more:\n"
            "the first 50 liters per person per day are charged at a low base rate, "
            "the next 50 liters at a higher rate, and any usage above 100 liters at the highest rate.\n"
            "If you reduce your daily water use, you can save money while helping the environment.\n"
        )

    # Early rounds use a general reference range instead of personal history.
    if round_num <= EARLY_REFERENCE_END_ROUND:
        prompt += (
            "Please estimate the daily household water usage (in liters per day) "
            "for this period. Give only one number. Do not add any words."
        )
        prompt += (
            "\nFor reference, typical per capita daily water usage is 80-160 liters."
        )
        return prompt

    if WEEK_PHASE_END_ROUND >= round_num >= 4:
        if group == "base":
            prompt += f"Your water usage last period is {last} liters/day.\n"
        elif group == "T1":
            # This branch is handled in process_chara because it requires additional history.
            pass
        elif group == "T2":
            prompt += (
                "The sustainable water usage target is 50-70 liters/day. "
                "Compare your usage with this target: "
            )
            if last is not None and last > 70:
                prompt += "Please note that your current water usage is above the sustainable level.\n"
            else:
                prompt += (
                    "Congratulations! You are approaching a sustainable level "
                    "of water conservation.\n"
                )
        elif group == "T3":
            prompt += "Here are four practical and motivating tips to save water:\n"
            prompt += (
                "1. Wash fruits in a bowl instead of under running water—you can save "
                "a significant amount of water each time.\n"
            )
            prompt += (
                "2. Turn off the tap while soaping your hands or doing laundry—each time "
                "you save 10-30 liters, which adds up quickly!\n"
            )
            prompt += (
                "3. Try shortening your shower time—most sailors finish in under 1 minute. "
                "You'll not only save water but also have more time for other activities!\n"
            )
            prompt += f"Your water usage last period is {last} liters/day.\n"
        else:
            raise ValueError(f"No such group {group}")
    # Retrieve the previous round's usage as feedback for the current prompt.
    elif round_num != 1:
        prompt += f"Your water usage last period is {last} liters/day.\n"

    prompt += (
        "Please estimate the daily household water usage (in liters per day) for this period. "
        "Give only one number. Do not add any words."
    )
    return prompt


def process_chara(
    cha_num: int,
    role: str,
    round_num: int,
    model_name: str,
    temperature: float,
    max_attempts: int,
    data_store: dict,
    group: str,
) -> None:
    """Process one character for one round and store the returned result."""
    role_prompt = (
        "Forget you are an AI model. You are a human participant in a social experiment. "
        + role
    )

    last = None
    if round_num != 1:
        last_list = data_store[cha_num - 1]
        last = next((item for item in last_list if item["round"] == round_num - 1))[
            "result"
        ]

    experiment_prompt = build_experiment_prompt(round_num, group, last)

    if WEEK_PHASE_END_ROUND >= round_num >= 4 and group == "T1":
        last_list = data_store[cha_num - 1]
        last_last = next(
            (item for item in last_list if item["round"] == round_num - 2)
        )["result"]
        filtered_results = [
            item["result"] for item in last_list if item["round"] < round_num - 2
        ]
        average = (
            sum(filtered_results) / len(filtered_results) if filtered_results else None
        )
        experiment_prompt += "Your Water Usage were as follows:\n"
        experiment_prompt += f"Last week: {last} liters/day\n"
        experiment_prompt += f"Week before last: {last_last} liters/day\n"
        experiment_prompt += f"Previous average: {average} liters/day\n"
        experiment_prompt += (
            "Please estimate the daily household water usage (in liters per day) for this period. "
            "Give only one number. Do not add any words."
        )

    response = get_fixed_response(
        role_prompt,
        experiment_prompt,
        model_name,
        temperature,
        cha_num,
        max_attempts,
        False,
        round_num,
    )

    if cha_num - 1 not in data_store:
        data_store[cha_num - 1] = []
    data_store[cha_num - 1].append(response)


def agent_experiment(
    model_name: str,
    temperature: float,
    num_threads: int,
    all_chara,
    max_attempts: int,
    group: str,
    output_dir: Path,
) -> None:
    """Run the experiment for all characters in one group and write JSON outputs."""
    data_store = {}
    all_chara = list(all_chara)

    for round_num in range(1, TOTAL_ROUNDS + 1):
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
                    data_store,
                    group,
                )
                for cha_num, role in enumerate(all_chara, start=1)
            ]

            with tqdm(
                total=len(futures),
                desc=f"Processing round {round_num}",
                ncols=100,
            ) as progress_bar:
                for future in as_completed(futures):
                    future.result()
                    progress_bar.update(1)

    for cha_num, data in data_store.items():
        output_file = output_dir / f"{cha_num + 1}.json"
        with output_file.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)


def run_group_experiment(
    group: str,
    output_root: Path,
    model_name: str,
    temperature: float,
    num_threads: int,
    max_attempts: int,
) -> None:
    """Run one experiment group unless result files already exist."""
    output_dir = output_root / group
    output_dir.mkdir(parents=True, exist_ok=True)

    if group_results_complete(output_dir):
        print("Result already existed.")
        return

    original_cwd = Path.cwd()
    os.chdir(output_dir)
    try:
        all_chara = build_group_profiles(is_control=(group == "base"))

        agent_experiment(
            model_name=model_name,
            temperature=temperature,
            num_threads=num_threads,
            all_chara=all_chara,
            max_attempts=max_attempts,
            group=group,
            output_dir=output_dir,
        )
    finally:
        os.chdir(original_cwd)


def main() -> None:
    """Parse arguments, load input data, and run all experiment groups."""
    parser = get_argument_parser()
    args = parser.parse_args()
    print(args)

    model_names = resolve_model_names(args.model_name)
    temperature = args.temperature
    num_threads = args.num_threads
    max_attempts = args.max_attempts

    result_root = args.output_dir.resolve()
    result_root.mkdir(parents=True, exist_ok=True)

    for model_name in model_names:
        print(f"Running {LONG_TERM_SUBDIR} with model: {model_name}")
        experiment_root = result_root / f"{model_name}_res" / LONG_TERM_SUBDIR
        for group in ("base", "T1", "T2", "T3"):
            run_group_experiment(
                group=group,
                output_root=experiment_root,
                model_name=model_name,
                temperature=temperature,
                num_threads=num_threads,
                max_attempts=max_attempts,
            )

    print(
        f"Completed {len(model_names)} model(s); "
        f"each group contains {NUM_AGENTS_PER_GROUP} agents."
    )


if __name__ == "__main__":
    main()
