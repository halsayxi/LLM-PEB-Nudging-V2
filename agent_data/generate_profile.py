from __future__ import annotations

import gzip
import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

DEMO_ENV_SUBDIR = Path("IPF") / "demographic_env_data"
SVO_SENSITIVITY_SUBDIR = Path("IPF") / "svo_sensitivity_data"

DEMO_ENV_JOINT_PROBS_FILENAME = "joint_probs_flat.csv.gz"
DEMO_ENV_CELL_VALUES_FILENAME = "cell_values.json"
SVO_SENSITIVITY_JOINT_PROBS_FILENAME = "joint_probs.csv"
SVO_SENSITIVITY_CELL_VALUES_FILENAME = "cell_values.json"

PROFILE_DIRNAME = "profile"
DEFAULT_OUTPUT_FILES = [
    "long_term_0/agent_data_base.json",
    "long_term_0/agent_data_control.json",
    "long_term_0/agent_data_intervention.json",
]

WAIT_TIMEOUT_SECONDS = 2
WAIT_INTERVAL_SECONDS = 0.1
DEFAULT_NUM_SAMPLES = 250

TARGET_FIELDS = [
    "envSelfEfficacy_pre",
    "envAttitude_pre",
    "envMotivation_pre",
    "Age",
    "Sex",
    "Ethnicity",
    "Occupation",
    "Income",
    "Education",
]


def load_demo_env_data(base_dir: Path) -> dict[str, Any]:
    """Load demographic and environmental joint distribution data."""
    data_dir = base_dir / DEMO_ENV_SUBDIR
    joint_probs_path = data_dir / DEMO_ENV_JOINT_PROBS_FILENAME
    cell_values_path = data_dir / DEMO_ENV_CELL_VALUES_FILENAME

    if not joint_probs_path.exists():
        raise FileNotFoundError(f"Not found: {joint_probs_path}")
    if not cell_values_path.exists():
        raise FileNotFoundError(f"Not found: {cell_values_path}")

    cells: list[dict[str, str]] = []
    probs: list[float] = []

    # The demographic/environment probability table is stored as gzip-compressed CSV.
    with gzip.open(joint_probs_path, "rt", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prob = float(row["prob"])
            probs.append(prob)
            cells.append(row)

    # Normalize probabilities.
    probs_array = np.array(probs, dtype=float)
    probs_sum = probs_array.sum()
    if probs_sum == 0:
        raise ValueError(f"Probability sum is zero in {joint_probs_path}")
    probs_array = probs_array / probs_sum

    with open(cell_values_path, "r", encoding="utf-8") as f:
        cell_values = json.load(f)

    return {
        "cells": cells,
        "probs": probs_array,
        "cell_values": cell_values,
    }


def load_svo_sensitivity_data(base_dir: Path) -> dict[str, Any]:
    """Load SVO and sensitivity-score joint distribution data."""
    data_dir = base_dir / SVO_SENSITIVITY_SUBDIR
    joint_probs_path = data_dir / SVO_SENSITIVITY_JOINT_PROBS_FILENAME
    cell_values_path = data_dir / SVO_SENSITIVITY_CELL_VALUES_FILENAME

    if not joint_probs_path.exists():
        raise FileNotFoundError(f"Not found: {joint_probs_path}")
    if not cell_values_path.exists():
        raise FileNotFoundError(f"Not found: {cell_values_path}")

    pairs: list[tuple[str, str]] = []
    probs: list[float] = []

    with open(joint_probs_path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Empty CSV file: {joint_probs_path}")

    header = rows[0]
    categories = header[1:]

    # Flatten the SVO × sensitivity table into sampled pairs with probabilities.
    for row in rows[1:]:
        svo = row[0]
        for index, probability in enumerate(row[1:]):
            pairs.append((svo, categories[index]))
            probs.append(float(probability))

    probs_array = np.array(probs, dtype=float)
    probs_sum = probs_array.sum()
    if probs_sum == 0:
        raise ValueError(f"Probability sum is zero in {joint_probs_path}")
    probs_array = probs_array / probs_sum

    with open(cell_values_path, "r", encoding="utf-8") as f:
        cell_values = json.load(f)

    return {
        "pairs": pairs,
        "probs": probs_array,
        "cell_values": cell_values,
    }


def parse_numeric_if_possible(value: Any) -> Any:
    """Convert a value to int/float when possible; otherwise return it unchanged."""
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return value

    text = str(value).strip()
    if text == "" or text.lower() == "none":
        return None

    # Preserve categorical strings while converting numeric-looking values.
    try:
        number = float(text)
        if number.is_integer():
            return int(number)
        return number
    except Exception:
        return value


def build_demo_env_key_from_cell_row(cell_row: dict[str, Any]) -> tuple[str, list[str]]:
    """Build the lookup key from all *_cat columns in a demographic/environment row."""
    category_columns = [key for key in cell_row.keys() if key.endswith("_cat")]
    key = "|||".join(str(cell_row[column]) for column in category_columns)
    return key, category_columns


def sample_demo_env(demo_env_data: dict[str, Any]) -> dict[str, Any]:
    """Sample one demographic/environment profile from the joint distribution."""
    # First sample one IPF cell, then sample concrete values from that cell.
    index = np.random.choice(len(demo_env_data["cells"]), p=demo_env_data["probs"])
    cell_row = demo_env_data["cells"][index]

    key, _ = build_demo_env_key_from_cell_row(cell_row)
    cell_values = demo_env_data["cell_values"].get(key, {})

    result: dict[str, Any] = {}

    for field in TARGET_FIELDS:
        values = cell_values.get(field, [])
        if values is not None and len(values) > 0:
            sampled_value = random.choice(values)
            result[field] = parse_numeric_if_possible(sampled_value)
        else:
            category_field = f"{field}_cat"
            if category_field in cell_row:
                result[field] = parse_numeric_if_possible(cell_row[category_field])
            else:
                result[field] = None

    return result


def sample_svo_sensitivity(svo_data: dict[str, Any]) -> dict[str, Any]:
    """Sample one SVO and sensitivity score pair from the joint distribution."""
    # Sample the joint SVO/sensitivity category before drawing a concrete score.
    index = np.random.choice(len(svo_data["pairs"]), p=svo_data["probs"])
    svo, category = svo_data["pairs"][index]

    key = f"{svo}|||{category}"
    values = svo_data["cell_values"].get(key, [])

    if len(values) == 0:
        sensitivity_score = None
    else:
        sensitivity_score = int(round(random.choice(values)))

    # Stored sensitivity values are scaled by 10, so convert back to the original score range.
    return {
        "svo": svo,
        "sensitivity_score": (
            sensitivity_score / 10 if sensitivity_score is not None else None
        ),
    }


def generate_person(
    person_id: int, demo_env_data: dict[str, Any], svo_data: dict[str, Any]
) -> dict[str, Any]:
    """Generate one person record."""
    person = {"id": person_id}
    person.update(sample_demo_env(demo_env_data))
    person.update(sample_svo_sensitivity(svo_data))
    return person


def generate_person_without_svo(
    person_id: int,
    demo_env_data: dict[str, Any],
) -> dict[str, Any]:
    """Generate one person record without svo and sensitivity_score."""
    person = {"id": person_id}
    person.update(sample_demo_env(demo_env_data))
    return person


def generate_population_with_ids(
    num_samples: int,
    demo_env_data: dict[str, Any],
    svo_data: dict[str, Any],
) -> list[dict[str, Any]]:
    """Generate a population with sequential IDs starting from 1."""
    population = []
    for person_id in range(1, num_samples + 1):
        population.append(generate_person(person_id, demo_env_data, svo_data))
    return population


def generate_population_with_ids_without_svo(
    num_samples: int,
    demo_env_data: dict[str, Any],
) -> list[dict[str, Any]]:
    """Generate a population with sequential IDs, excluding svo and sensitivity_score."""
    population = []
    for person_id in range(1, num_samples + 1):
        population.append(
            generate_person_without_svo(
                person_id=person_id,
                demo_env_data=demo_env_data,
            )
        )
    return population


def save_population_to_file(
    population: list[dict[str, Any]], filename: str | Path
) -> None:
    """Save population data to a JSON file."""
    output_path = Path(filename)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(population, file, indent=4, ensure_ascii=False)


def wait_for_file(path: str | Path, timeout: float = WAIT_TIMEOUT_SECONDS) -> None:
    """Wait until a file exists or raise a timeout error."""
    target_path = Path(path)
    start_time = time.time()

    # This protects downstream code that immediately reads the generated JSON file.
    while not target_path.exists():
        if time.time() - start_time > timeout:
            raise TimeoutError(f"File {target_path} not created in time.")
        time.sleep(WAIT_INTERVAL_SECONDS)


def generate_and_save_population(num_agents: int, control: bool) -> None:
    """Generate population data and save it to the profile directory."""
    base_dir = Path(__file__).resolve().parent

    demo_env_data = load_demo_env_data(base_dir)
    svo_data = load_svo_sensitivity_data(base_dir)

    output_dir = Path.cwd()
    output_dir = output_dir / PROFILE_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)

    # Keep separate filenames for control and intervention populations.
    if control:
        file_path = output_dir / f"agent_data_control_{num_agents}.json"
    else:
        file_path = output_dir / f"agent_data_intervention_{num_agents}.json"

    population_data = generate_population_with_ids(
        num_samples=num_agents,
        demo_env_data=demo_env_data,
        svo_data=svo_data,
    )

    save_population_to_file(population_data, file_path)
    wait_for_file(file_path)
    print(f"Data has been saved to {file_path} file.")


def generate_and_save_population_without_svo(num_agents: int) -> None:
    """Generate population data without svo/sensitivity_score and save it."""
    base_dir = Path(__file__).resolve().parent

    demo_env_data = load_demo_env_data(base_dir)

    output_dir = Path.cwd() / PROFILE_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)

    file_path = output_dir / f"agent_data_{num_agents}.json"

    population_data = generate_population_with_ids_without_svo(
        num_samples=num_agents,
        demo_env_data=demo_env_data,
    )

    save_population_to_file(population_data, file_path)
    wait_for_file(file_path)
    print(f"Data has been saved to {file_path} file.")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent

    demo_env_data = load_demo_env_data(base_dir)
    svo_data = load_svo_sensitivity_data(base_dir)

    # Standalone execution generates default population files for long-term experiments.
    for relative_file_path in DEFAULT_OUTPUT_FILES:
        population_data = generate_population_with_ids(
            num_samples=DEFAULT_NUM_SAMPLES,
            demo_env_data=demo_env_data,
            svo_data=svo_data,
        )

        output_path = base_dir / relative_file_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(population_data, file, indent=4, ensure_ascii=False)

        print(f"Saved: {output_path}")
