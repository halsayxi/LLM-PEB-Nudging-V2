import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

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
from utils import get_res


DEFAULT_TEMPERATURE = 0.7
DEFAULT_NUM_THREADS = 30
DEFAULT_MAX_ATTEMPTS = 5

TOTAL_WEEKS = 29
RECENT_WEEKS_WINDOW = 7
EXPECTED_OUTPUT_COUNT = 7

RESULT_ROOT_DIRNAME = "res_long_term"
LONG_TERM_SUBDIR = "long_term_2"

GROUP_BASELINE = "baseline"
GROUP_NUDGE = "nudge"
GROUP_BOOST = "boost"

ROLE_PREFIX = (
    "Forget you are an AI model. You are a human participant in a social experiment."
)

OUTPUT_FIELD_NAMES = [
    "electricity_kWh",
    "warm_water_m3",
    "feedback_engagement_sec",
    "information_engagement_count",
    "energy_competence",
    "self_efficacy",
    "social_comparison",
]


def get_argument_parser():
    """Create and return the command-line argument parser."""
    parser = argparse.ArgumentParser()
    add_model_argument(parser)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--num_threads", type=int, default=DEFAULT_NUM_THREADS)
    parser.add_argument("--max_attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PROJECT_ROOT / "study1" / "nudge_replication" / RESULT_ROOT_DIRNAME,
    )
    return parser


def get_fixed_response(
    role,
    exp,
    model_name,
    temperature,
    cha_num,
    max_attempts,
    round_num=None,
):
    """Call the model repeatedly until a valid 7-value numeric response is parsed.

    If parsing fails after all attempts, the returned structure keeps the original
    schema and fills numeric fields with None.
    """
    attempt = 0
    chara_res = {}
    output = ""

    while attempt < max_attempts:
        chara_res = get_res(role, exp, model_name, temperature)
        output = chara_res.get("output", "")

        if output:
            output = output.strip()
            try:
                # The model must return exactly seven comma-separated numeric values.
                values = [float(item.strip()) for item in output.split(",")]
                if len(values) == EXPECTED_OUTPUT_COUNT:
                    return {
                        "round": round_num,
                        "input": chara_res.get("input", ""),
                        "output": output,
                        OUTPUT_FIELD_NAMES[0]: values[0],
                        OUTPUT_FIELD_NAMES[1]: values[1],
                        OUTPUT_FIELD_NAMES[2]: values[2],
                        OUTPUT_FIELD_NAMES[3]: values[3],
                        OUTPUT_FIELD_NAMES[4]: values[4],
                        OUTPUT_FIELD_NAMES[5]: values[5],
                        OUTPUT_FIELD_NAMES[6]: values[6],
                        "reason_output": chara_res.get("reason_output", ""),
                    }
            except Exception:
                pass

        if attempt > 0:
            print(output)
            print(f"⚠️ Attempt {attempt + 1} failed for chara {cha_num}, retrying...")

        attempt += 1

    print(f"Failed to parse output after {max_attempts} attempts.")
    return {
        "round": round_num,
        "input": chara_res.get("input", ""),
        "output": output,
        OUTPUT_FIELD_NAMES[0]: None,
        OUTPUT_FIELD_NAMES[1]: None,
        OUTPUT_FIELD_NAMES[2]: None,
        OUTPUT_FIELD_NAMES[3]: None,
        OUTPUT_FIELD_NAMES[4]: None,
        OUTPUT_FIELD_NAMES[5]: None,
        OUTPUT_FIELD_NAMES[6]: None,
        "reason_output": chara_res.get("reason_output", ""),
    }


def process_chara(
    cha_num,
    role,
    round_num,
    model_name,
    temperature,
    max_attempts,
    data_store,
    group,
):
    """Process one character for one experimental week."""
    role = f"{ROLE_PREFIX} {role}"
    exp = f"You are participating in an experiment. This is week {round_num}.\n"

    if round_num != 1:
        # Later weeks use the participant's previous records as memory.
        cha_data = data_store[cha_num - 1]
        total_electricity = sum(week_data["electricity_kWh"] for week_data in cha_data)
        total_warm_water = sum(week_data["warm_water_m3"] for week_data in cha_data)

        cha_data_sorted = sorted(cha_data, key=lambda item: item["round"])
        cha_recent_weeks = cha_data_sorted[-RECENT_WEEKS_WINDOW:]

        # Approximate peer comparison by averaging other participants' records for the same weeks.
        other_participants = [
            data_store[index]
            for index in range(len(data_store))
            if index != cha_num - 1
        ]

        friends_avg_weeks = []
        for week_data in cha_recent_weeks:
            week_num = week_data["round"]
            week_vals = []

            for participant_data in other_participants:
                week_entry = next(
                    (entry for entry in participant_data if entry["round"] == week_num),
                    None,
                )
                if week_entry:
                    week_vals.append(week_entry)

            if week_vals:
                avg_electricity = sum(
                    item["electricity_kWh"] for item in week_vals
                ) / len(week_vals)
                avg_water = sum(item["warm_water_m3"] for item in week_vals) / len(
                    week_vals
                )
            else:
                avg_electricity, avg_water = 0, 0

            friends_avg_weeks.append(
                {
                    "round": week_num,
                    "electricity_kWh": avg_electricity,
                    "warm_water_m3": avg_water,
                }
            )

    # Each group receives a different intervention framing.
    if group == GROUP_BASELINE:
        exp += "A web link library contains various energy-saving tips for your reference.\n"
        if round_num != 1:
            if round_num < 13:
                exp += (
                    f"As of this week, your cumulative electricity usage is "
                    f"{total_electricity} kWh.\n"
                )
            else:
                exp += (
                    f"As of this week, your cumulative electricity usage is "
                    f"{total_electricity} kWh, and your cumulative hot water usage is "
                    f"{total_warm_water} m³.\n"
                )

    elif group == GROUP_BOOST:
        exp += (
            "Next to each device in your daily life (e.g., oven, fridge), there is a QR code "
            "you can scan to view energy-saving tips specific to that device. "
            "It is very convenient. You can scan the QR codes with your phone to receive and "
            "store the tips.\n"
        )
        if round_num != 1:
            if round_num < 13:
                exp += (
                    f"As of this week, your cumulative electricity usage is "
                    f"{total_electricity} kWh.\n"
                )
            else:
                exp += (
                    f"As of this week, your cumulative electricity usage is "
                    f"{total_electricity} kWh, and your cumulative hot water usage is "
                    f"{total_warm_water} m³.\n"
                )

    elif group == GROUP_NUDGE:
        # The nudge group sees social-comparison feedback and a 10% saving goal.
        exp += "A web link library contains various energy-saving tips for your reference.\n"
        if round_num != 1:
            if round_num < 13:
                exp += (
                    "Over the past several weeks, your and your friends' weekly electricity "
                    "usage were as follows:\n"
                )
                for index in range(len(cha_recent_weeks)):
                    week = cha_recent_weeks[index]["round"]
                    your_electricity = cha_recent_weeks[index]["electricity_kWh"]
                    friend_electricity = friends_avg_weeks[index]["electricity_kWh"]
                    exp += (
                        f"Week {week}: You ({your_electricity:.2f} kWh), "
                        f"Friends ({friend_electricity:.2f} kWh)\n"
                    )
            else:
                exp += (
                    "Over the past several weeks, your and your friends' weekly electricity "
                    "and hot water usage were as follows:\n"
                )
                for index in range(len(cha_recent_weeks)):
                    week = cha_recent_weeks[index]["round"]
                    your_electricity = cha_recent_weeks[index]["electricity_kWh"]
                    your_water = cha_recent_weeks[index]["warm_water_m3"]
                    friend_electricity = friends_avg_weeks[index]["electricity_kWh"]
                    friend_water = friends_avg_weeks[index]["warm_water_m3"]
                    exp += (
                        f"Week {week}: You ({your_electricity:.2f} kWh / {your_water:.2f} m³), "
                        f"Friends ({friend_electricity:.2f} kWh / {friend_water:.2f} m³)\n"
                    )
        exp += "You have set an energy-saving goal for yourself (10%).\n"

    else:
        raise ValueError(f"No such group {group}")

    # Enforce a compact numeric format so results can be parsed automatically.
    exp += (
        "Please answer the following questions (separate your answers with commas, in the "
        "order listed below):\n"
        "1. Electricity used this week (kWh)\n"
        "2. Hot water used this week (m³)\n"
        "3. Total time spent checking your energy dashboard this week (seconds)\n"
        "4. Number of times you accessed energy-saving tips this week\n"
        "5. Rate your energy-saving competence (0-10)\n"
        "6. Rate your confidence in your energy-saving skills (0-10)\n"
        "7. Number of times you compared your energy usage to your peers this week\n"
        "**IMPORTANT:** Output only 7 numeric values, separated by commas, on a single line. "
        "Do not include any text, explanations, or other characters."
    )

    response = get_fixed_response(
        role=role,
        exp=exp,
        model_name=model_name,
        temperature=temperature,
        cha_num=cha_num,
        max_attempts=max_attempts,
        round_num=round_num,
    )

    if cha_num - 1 not in data_store:
        data_store[cha_num - 1] = []
    data_store[cha_num - 1].append(response)


def agent_experiment(
    model_name,
    temperature,
    num_threads,
    all_chara,
    max_attempts,
    group,
):
    """Run the full multi-week experiment for all characters in one group."""
    data_store = {}
    all_chara = list(all_chara)

    for round_num in range(1, TOTAL_WEEKS + 1):
        # Process all agents for the same week in parallel.
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
                desc=f"Processing week {round_num}",
                ncols=100,
            ) as progress_bar:
                for future in as_completed(futures):
                    future.result()
                    progress_bar.update(1)

    # Save one JSON trajectory per agent after all weeks are completed.
    for cha_num, data in data_store.items():
        filename = f"{cha_num + 1}.json"
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)


def run_group_experiment(
    group_name,
    output_dir,
    model_name,
    temperature,
    num_threads,
    max_attempts,
):
    """Run one group experiment inside its output directory if results do not already exist."""
    group_dir = Path(output_dir)
    group_dir.mkdir(parents=True, exist_ok=True)

    original_cwd = Path.cwd()
    try:
        os.chdir(group_dir)

        if group_results_complete(Path.cwd()):
            print("Result already existed.")
        else:
            all_chara = build_group_profiles(
                is_control=(group_name == GROUP_BASELINE)
            )
            agent_experiment(
                model_name=model_name,
                temperature=temperature,
                num_threads=num_threads,
                all_chara=all_chara,
                max_attempts=max_attempts,
                group=group_name,
            )
    finally:
        os.chdir(original_cwd)


def main():
    """Parse arguments, load inputs, and run all experiment groups."""
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
        for group_name in (GROUP_BASELINE, GROUP_NUDGE, GROUP_BOOST):
            run_group_experiment(
                group_name=group_name,
                output_dir=experiment_root / group_name,
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
