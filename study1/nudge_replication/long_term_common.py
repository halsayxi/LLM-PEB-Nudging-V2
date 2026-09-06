"""Shared configuration and profile helpers for Study 1 long-term runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from agent_data.attribute_to_description import process_agent_descriptions
from agent_data.generate_profile import generate_and_save_population


MODEL_NAMES = [
    "gpt-3.5-turbo-0125",
    "deepseek-v3",
    "claude-haiku-4-5-20251001",
    "gpt-4o-2024-11-20",
    "deepseek-v4-flash",
    "gpt-5.4",
]

NUM_AGENTS_PER_GROUP = 50


def add_model_argument(parser: argparse.ArgumentParser) -> None:
    """Add a model argument that supports one, several, or all configured models."""
    parser.add_argument(
        "--model_name",
        nargs="+",
        default=["all"],
        help=(
            "Model name(s) to run. The default, 'all', runs all six configured "
            "models. A single model name remains supported."
        ),
    )


def resolve_model_names(values: list[str]) -> list[str]:
    """Validate and normalize the model names supplied on the command line."""
    if not values or values == ["all"]:
        return MODEL_NAMES.copy()
    if "all" in values:
        raise ValueError("Use --model_name all by itself, or list explicit models.")

    unknown = [value for value in values if value not in MODEL_NAMES]
    if unknown:
        raise ValueError(
            f"Unsupported model(s): {unknown}. Supported models: {MODEL_NAMES}"
        )
    return list(dict.fromkeys(values))


def build_group_profiles(is_control: bool) -> list[str]:
    """Generate exactly 50 profiles in the current result directory."""
    generate_and_save_population(NUM_AGENTS_PER_GROUP, is_control)
    profiles = process_agent_descriptions(NUM_AGENTS_PER_GROUP, is_control)
    if profiles is None:
        raise RuntimeError("Failed to generate agent profile descriptions.")
    if len(profiles) != NUM_AGENTS_PER_GROUP:
        raise ValueError(
            f"Expected {NUM_AGENTS_PER_GROUP} profiles, found {len(profiles)}."
        )
    return profiles


def group_results_complete(output_dir: Path) -> bool:
    """Return whether a group directory contains all 50 per-agent results."""
    if not output_dir.exists():
        return False
    result_files = [
        path
        for path in output_dir.glob("*.json")
        if path.is_file() and path.stem.isdigit()
    ]
    return len(result_files) == NUM_AGENTS_PER_GROUP
