import argparse
import calendar
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

# Add the project root so shared utility functions can be imported.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from study1.nudge_replication.long_term_common import (
    NUM_AGENTS_PER_GROUP,
    add_model_argument,
    build_group_profiles,
    group_results_complete,
    resolve_model_names,
)
from utils import get_fixed_response


DEFAULT_TEMPERATURE = 0.7
DEFAULT_NUM_THREADS = 30
DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_NUM_ROUNDS1 = 2
DEFAULT_NUM_ROUNDS2 = 24
DEFAULT_NUM_ROUNDS3 = 12

RESULT_ROOT_DIRNAME = "res_long_term"
LONG_TERM_SUBDIR = "long_term_0"


def get_argument_parser() -> argparse.ArgumentParser:
    """Create and return the CLI argument parser."""
    parser = argparse.ArgumentParser()
    add_model_argument(parser)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--num_threads", type=int, default=DEFAULT_NUM_THREADS)
    parser.add_argument("--max_attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument("--num_rounds1", type=int, default=DEFAULT_NUM_ROUNDS1)
    parser.add_argument("--num_rounds2", type=int, default=DEFAULT_NUM_ROUNDS2)
    parser.add_argument("--num_rounds3", type=int, default=DEFAULT_NUM_ROUNDS3)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PROJECT_ROOT / "study1" / "nudge_replication" / RESULT_ROOT_DIRNAME,
    )
    return parser


def get_month_and_year(round_num: int) -> tuple[str, int]:
    """Convert a round number to its corresponding month name and year index."""
    month_index = (round_num - 1) % 12 + 1
    year_index = (round_num - 1) // 12 + 1
    month_name = calendar.month_name[month_index]
    return month_name, year_index


def process_chara(
    cha_num,
    role,
    round_num,
    model_name,
    temperature,
    max_attempts,
    data_store,
    num_rounds1,
    num_rounds2,
    num_rounds3,
    group,
):
    """Run one round of the experiment for a single character."""
    role = (
        "Forget you are an AI model. You are a human participant in a social experiment. "
        + role
    )

    month_name, year_index = get_month_and_year(round_num)
    round_header = f"{month_name} (Year {year_index})"
    question = (
        f"This is {round_header}. For this month, what is your household electricity "
        f"consumption in kilowatt-hours (kWh)? Please respond with a single number only, "
        f"without any additional explanation or units."
    )

    # Use the first two rounds or the base group as baseline electricity usage.
    if round_num <= num_rounds1 or group == "base":
        exp = (
            "You are participating in a household electricity usage study. " + question
        )
    else:
        # Collect the current agent's recent consumption history.
        user_data = data_store.get(cha_num - 1, [])
        user_last_12 = [entry["result"] for entry in user_data[-12:]]
        months = len(user_last_12)

        # Compute neighbors' average consumption over the same recent months.
        neighbors_sum = [0.0] * months
        neighbors_count = 0

        for other_cha_num, values in data_store.items():
            if other_cha_num == cha_num - 1:
                continue
            other_last_12 = [entry["result"] for entry in values[-months:]]
            neighbors_sum = [
                current_sum + value
                for current_sum, value in zip(neighbors_sum, other_last_12)
            ]
            neighbors_count += 1

        neighbors_avg_12 = [value_sum / neighbors_count for value_sum in neighbors_sum]

        start_round_num = round_num - months
        month_years = [get_month_and_year(start_round_num + i) for i in range(months)]
        month_names = [f"{month} (Year {year})" for month, year in month_years]

        your_last = user_last_12[-1]
        neighbor_last = neighbors_avg_12[-1]
        higher_or_lower = "higher" if your_last > neighbor_last else "lower"

        avg_your = sum(user_last_12) / months
        avg_neighbor = sum(neighbors_avg_12) / months
        avg_diff = avg_your - avg_neighbor
        monthly_cost_extra = avg_diff * 10

        # The intervention group keeps receiving the nudge for an additional period.
        if group == "intervention":
            nudge_limit = num_rounds1 + num_rounds2 + num_rounds3
        else:
            nudge_limit = num_rounds1 + num_rounds2

        if round_num <= nudge_limit:
            prompt = (
                f"Last month ({month_names[-1]}), your household electricity consumption was "
                f"{your_last:.1f} kWh, which is {higher_or_lower} than your neighbors' average "
                f"of {neighbor_last:.1f} kWh.\n\n"
                f"Here is your electricity consumption trend over the past {months} months:\n"
            )

            for month_label, your_value, neighbor_value in zip(
                month_names, user_last_12, neighbors_avg_12
            ):
                prompt += (
                    f"- {month_label}: You used {your_value:.1f} kWh; neighbors' average was "
                    f"{neighbor_value:.1f} kWh.\n"
                )

            if avg_diff > 0:
                prompt += (
                    f"\nOn average, over the past {months} months, you used {avg_diff:.1f} kWh "
                    f"more per month than your neighbors, which costs you approximately "
                    f"${monthly_cost_extra:.2f} extra each month."
                )
            elif avg_diff < 0:
                prompt += (
                    f"\nOn average, over the past {months} months, you used {abs(avg_diff):.1f} "
                    f"kWh less per month than your neighbors, saving you approximately "
                    f"${abs(monthly_cost_extra):.2f} each month."
                )
            else:
                prompt += (
                    f"\nOn average, over the past {months} months, your electricity usage was "
                    f"about the same as your neighbors."
                )

            exp = prompt + "\n" + question
        else:
            user_history = data_store.get(cha_num - 1, [])
            first_month_value = (
                user_history[0]["result"] if len(user_history) > 0 else "N/A"
            )
            second_month_value = (
                user_history[1]["result"] if len(user_history) > 1 else "N/A"
            )

            first_month_name, first_year = get_month_and_year(1)
            second_month_name, second_year = get_month_and_year(2)
            last_month_name = month_names[-1]

            prompt = (
                f"You used to receive a monthly household energy report, but these reports are "
                f"no longer being sent.\n\n"
                f"For your reference, here is a record of your electricity consumption:\n"
                f"- {first_month_name} (Year {first_year}): {first_month_value} kWh\n"
                f"- {second_month_name} (Year {second_year}): {second_month_value} kWh\n\n"
                f"Last month ({last_month_name}), your household electricity consumption was "
                f"{your_last:.1f} kWh.\n"
            )
            exp = prompt + question

    res = get_fixed_response(
        role,
        exp,
        model_name,
        temperature,
        cha_num,
        max_attempts,
        False,
        round_num,
    )

    if cha_num - 1 not in data_store:
        data_store[cha_num - 1] = []
    data_store[cha_num - 1].append(res)


def agent_experiment(
    model_name,
    temperature,
    num_threads,
    all_chara,
    max_attempts,
    num_rounds1,
    num_rounds2,
    num_rounds3,
    group,
):
    """Run the full experiment for one group and write per-character JSON outputs."""
    data_store = {}
    all_chara = list(all_chara)
    total_rounds = num_rounds1 + num_rounds2 + num_rounds3

    for round_num in range(1, total_rounds + 1):
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
                    num_rounds1,
                    num_rounds2,
                    num_rounds3,
                    group,
                )
                for cha_num, role in enumerate(all_chara, start=1)
            ]

            with tqdm(
                total=len(futures),
                desc=f"Processing month {round_num}",
                ncols=100,
            ) as pbar:
                for future in as_completed(futures):
                    future.result()
                    pbar.update(1)

    for cha_num, data in data_store.items():
        filename = f"{cha_num + 1}.json"
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)


def run_group_experiment(
    output_dir: Path,
    group_name: str,
    model_name: str,
    temperature: float,
    num_threads: int,
    max_attempts: int,
    num_rounds1: int,
    num_rounds2: int,
    num_rounds3: int,
) -> None:
    """Run one group if its output directory does not already contain JSON results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    original_cwd = Path.cwd()

    try:
        os.chdir(output_dir)

        if group_results_complete(Path.cwd()):
            print(f"Result already existed for group '{group_name}'.")
            return

        all_chara = build_group_profiles(is_control=(group_name == "base"))

        agent_experiment(
            model_name,
            temperature,
            num_threads,
            all_chara,
            max_attempts,
            num_rounds1,
            num_rounds2,
            num_rounds3,
            group=group_name,
        )
    finally:
        os.chdir(original_cwd)


def main():
    """Parse arguments, load character data, and run experiments for all groups."""
    parser = get_argument_parser()
    args = parser.parse_args()
    print(args)

    model_names = resolve_model_names(args.model_name)
    temperature = args.temperature
    num_threads = args.num_threads
    max_attempts = args.max_attempts
    num_rounds1 = args.num_rounds1
    num_rounds2 = args.num_rounds2
    num_rounds3 = args.num_rounds3

    result_root = args.output_dir.resolve()
    result_root.mkdir(parents=True, exist_ok=True)

    for model_name in model_names:
        print(f"Running {LONG_TERM_SUBDIR} with model: {model_name}")
        experiment_root = result_root / f"{model_name}_res" / LONG_TERM_SUBDIR
        for group_name in ("control", "intervention", "base"):
            run_group_experiment(
                output_dir=experiment_root / group_name,
                group_name=group_name,
                model_name=model_name,
                temperature=temperature,
                num_threads=num_threads,
                max_attempts=max_attempts,
                num_rounds1=num_rounds1,
                num_rounds2=num_rounds2,
                num_rounds3=num_rounds3,
            )

    print(
        f"Completed {len(model_names)} model(s); "
        f"each group contains {NUM_AGENTS_PER_GROUP} agents."
    )


if __name__ == "__main__":
    main()
