import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from agent_data.attribute_to_description import process_agent_descriptions
from agent_data.generate_profile import generate_and_save_population
from utils import analyze_results, get_fixed_response

DEFAULT_MODEL_NAME = "gpt-3.5-turbo-0125"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_NUM_THREADS = 120
DEFAULT_MAX_ATTEMPTS = 5

# Metadata table containing study prompts, sample sizes, and outcome settings.
PROMPT_FILE_PATH = PROJECT_ROOT / "prompt" / "meta_new_list.csv"
RESULTS_ROOT_DIR = "res"
CONTROL_RESULTS_FILENAME = "control.json"
INTERVENTION_RESULTS_FILENAME = "intervention.json"

DEFAULT_NUM_CHARS = 50


def get_argument_parser() -> argparse.ArgumentParser:
    """Create and return the command-line argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--num_threads", type=int, default=DEFAULT_NUM_THREADS)
    parser.add_argument("--max_attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument("--output_dir", type=str, default=RESULTS_ROOT_DIR)
    parser.add_argument(
        "--num_chars",
        type=int,
        default=DEFAULT_NUM_CHARS,
        help="Number of characters to process (-1 means use all characters without limit)",
    )
    return parser


def process_chara(
    cha_num: int,
    role: str,
    model_name: str,
    temperature: float,
    max_attempts: int,
    exp_data,
    control: bool,
):
    """Run one agent for one experiment condition and return its response."""
    # Select the scenario prompt according to the experimental condition.
    prompt_key = (
        "control_group_scenario_prompt"
        if control
        else "intervention_group_scenario_prompt"
    )

    # Restrict the model output to a numeric response for downstream parsing.
    experiment_prompt = (
        f"{exp_data[prompt_key]}\n"
        f"{exp_data['response_instruction']} "
        f"{exp_data['response_options']} "
        "Only output a number."
    )

    result = get_fixed_response(
        role,
        experiment_prompt,
        model_name,
        temperature,
        cha_num,
        max_attempts,
        exp_data["binary_outcome"],
    )

    return cha_num, result


def run_group(
    group_name: str,
    chara_list,
    num_threads: int,
    model_name: str,
    temperature: float,
    max_attempts: int,
    exp_data,
    control_flag: bool,
):
    """Run all agents in one group and save the results to a JSON file."""
    print(f"Processing {group_name} group...")

    # Preallocate the result list so agent order is preserved.
    results = [None] * len(chara_list)

    # Run agents in parallel to speed up independent response generation.
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(
                process_chara,
                cha_num,
                role,
                model_name,
                temperature,
                max_attempts,
                exp_data,
                control_flag,
            )
            for cha_num, role in enumerate(chara_list, start=1)
        ]

        with tqdm(total=len(futures), desc=f"{group_name} group", ncols=100) as pbar:
            for future in as_completed(futures):
                cha_num, result = future.result()

                # Save each result back to its original agent position.
                results[cha_num - 1] = result

                pbar.update(1)

    output_path = Path(f"{group_name}.json")
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    return results

def agent_experiment(
    model_name: str,
    temperature: float,
    num_threads: int,
    control_chara,
    intervention_chara,
    max_attempts: int,
    exp_data,
) -> None:
    """Run control and intervention groups, then analyze the experiment results."""
    res_control = run_group(
        "control",
        control_chara,
        num_threads,
        model_name,
        temperature,
        max_attempts,
        exp_data,
        control_flag=True,
    )

    res_intervention = run_group(
        "intervention",
        intervention_chara,
        num_threads,
        model_name,
        temperature,
        max_attempts,
        exp_data,
        control_flag=False,
    )

    analyze_results(exp_data["study_id"], res_control, res_intervention, exp_data)


def main() -> None:
    """Run the full experiment pipeline for all studies listed in the CSV file."""
    parser = get_argument_parser()
    args = parser.parse_args()
    print(args)

    model_name = args.model_name
    temperature = args.temperature
    num_threads = args.num_threads
    max_attempts = args.max_attempts
    output_dir = args.output_dir
    num_chars = args.num_chars

    try:
        df = pd.read_csv(PROMPT_FILE_PATH)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"CSV file not found: {PROMPT_FILE_PATH}") from exc
    except UnicodeDecodeError as exc:
        raise UnicodeDecodeError(
            exc.encoding,
            exc.object,
            exc.start,
            exc.end,
            f"Failed to decode CSV file: {PROMPT_FILE_PATH}. {exc.reason}",
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to read CSV file: {PROMPT_FILE_PATH}") from exc

    original_cwd = Path.cwd()
    results_dir = original_cwd / output_dir / f"{model_name}_res"
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Each study writes outputs inside its own result directory.
        os.chdir(results_dir)

        for _, row in df.iterrows():
            study_id = row["study_id"]
            print(f"Processing study_id {study_id}...")

            n_control = int(row["n_control"])
            n_intervention = int(row["n_intervention"])
            # Cap very large studies to keep the social simulation computationally manageable.
            if n_control + n_intervention > 1000:
                n_control = 500
                n_intervention = 500

            study_dir = Path(study_id)
            study_dir.mkdir(exist_ok=True)
            os.chdir(study_dir)
            try:
                control_path = Path(CONTROL_RESULTS_FILENAME)
                intervention_path = Path(INTERVENTION_RESULTS_FILENAME)

                # Reuse completed group outputs and only rerun the analysis step.
                if control_path.exists() and intervention_path.exists():
                    with control_path.open("r", encoding="utf-8") as file:
                        res_control = json.load(file)
                    with intervention_path.open("r", encoding="utf-8") as file:
                        res_intervention = json.load(file)

                    print(f"Study_id {study_id}: Results already existed.")
                    analyze_results(study_id, res_control, res_intervention, row)
                    continue

                # Generate separate virtual populations for control and intervention groups.
                if num_chars == -1:
                    generate_and_save_population(n_control, 1)
                    generate_and_save_population(n_intervention, 0)
                    control_chara = process_agent_descriptions(n_control, 1)
                    intervention_chara = process_agent_descriptions(n_intervention, 0)
                else:
                    generate_and_save_population(num_chars, 1)
                    generate_and_save_population(num_chars, 0)
                    control_chara = process_agent_descriptions(num_chars, 1)
                    intervention_chara = process_agent_descriptions(num_chars, 0)

                agent_experiment(
                    model_name,
                    temperature,
                    num_threads,
                    control_chara,
                    intervention_chara,
                    max_attempts,
                    row,
                )
            finally:
                os.chdir("..")
    finally:
        os.chdir(original_cwd)

    print("All studies processed.")


if __name__ == "__main__":
    main()
