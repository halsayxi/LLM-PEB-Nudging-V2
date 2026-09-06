import json
import time
from pathlib import Path
from typing import Any

# Template used to convert structured agent attributes into a natural-language profile.
DESCRIPTION_TEMPLATE = (
    "You are a {sex} aged {age}. "
    "Your ethnicity is {ethnicity}. "
    "You are {occupation_phrase} and your education level is {education_phrase}. "
    "Your income level is {income_phrase}. "
    "Your social value orientation (SVO) is {svo_phrase}. "
    "Your environmental self-efficacy is {env_selfeff_desc} (score: {env_selfeff_score}/5), "
    "your environmental attitude is {env_attitude_desc} (score: {env_attitude_score}/5), "
    "your environmental motivation is {env_motivation_desc} (score: {env_motivation_score}, range: -14 to 14), "
    "and your emotional sensitivity is {sensitivity_desc} (score: {sensitivity_score}/5)."
)


PROFILE_DIR = Path("profile")
WAIT_TIMEOUT_SECONDS = 2
WAIT_INTERVAL_SECONDS = 0.1

# Code-to-text mappings keep generated profiles readable and consistent.
OCCUPATION_MAPPING = {
    1: "a student",
    2: "working in the government or public sector",
    3: "working in the private sector",
    4: "a freelancer or self-employed",
    5: "in another type of occupation",
}

INCOME_MAPPING = {
    1: "less than £10,000",
    2: "£10,001–£20,000",
    3: "£20,001–£30,000",
    4: "£30,001–£40,000",
    5: "£40,001–£60,000",
    6: "above £60,000",
    7: "prefer not to say",
}

EDUCATION_MAPPING = {
    1: "primary school or below",
    2: "middle school",
    3: "high school",
    4: "an associate degree",
    5: "a bachelor’s degree",
    6: "a master’s degree",
    7: "a doctorate",
}

SVO_MAPPING = {
    "proself": "proself",
    "prosocial": "prosocial",
    "none": "none",
}


def map_occupation(value: Any) -> str:
    """Map occupation code to a human-readable phrase."""
    try:
        occupation_code = int(value)
        return OCCUPATION_MAPPING.get(
            occupation_code, f"working in occupation category {occupation_code}"
        )
    except Exception:
        return str(value).lower() if value is not None else "in an unknown occupation"


def map_income(value: Any) -> str:
    """Map income code to a human-readable phrase."""
    try:
        income_code = int(value)
        return INCOME_MAPPING.get(income_code, f"in income category {income_code}")
    except Exception:
        return str(value) if value is not None else "unknown"


def map_education(value: Any) -> str:
    """Map education code to a human-readable phrase."""
    try:
        education_code = int(value)
        return EDUCATION_MAPPING.get(
            education_code, f"education category {education_code}"
        )
    except Exception:
        return str(value).lower() if value is not None else "unknown"


def map_svo(value: Any) -> str:
    """Normalize and map SVO value."""
    if value is None:
        return "unknown"
    normalized_value = str(value).strip().lower()
    return SVO_MAPPING.get(normalized_value, normalized_value)


def normalize_sex(value: Any) -> str:
    """Normalize sex value for display."""
    if value is None:
        return "unknown"
    return str(value).strip().lower()


def normalize_ethnicity(value: Any) -> str:
    """Normalize ethnicity value for display."""
    if value is None:
        return "unknown"
    return str(value).strip()


def describe_1to5_score(score: Any) -> str:
    """Convert a 1-to-5 style score into a descriptive label."""
    try:
        numeric_score = float(score)
    except Exception:
        return "unknown"

    if numeric_score < 2:
        return "low"
    if numeric_score < 3:
        return "relatively low"
    if numeric_score < 4:
        return "moderate"
    if numeric_score < 4.5:
        return "high"
    return "very high"


def describe_env_motivation(score: Any) -> str:
    """Convert environmental motivation score into a descriptive label."""
    try:
        numeric_score = float(score)
    except Exception:
        return "unknown"

    if numeric_score < -8.4:
        return "very low"
    if numeric_score < -2.8:
        return "low"
    if numeric_score < 2.8:
        return "moderate"
    if numeric_score < 8.4:
        return "high"
    return "very high"


def generate_description(agent: dict[str, Any]) -> str:
    """Generate a natural-language profile description for one agent."""
    age = agent.get("Age", "unknown")
    # Normalize categorical fields before inserting them into the profile template.
    sex = normalize_sex(agent.get("Sex", "unknown"))
    ethnicity = normalize_ethnicity(agent.get("Ethnicity", "unknown"))

    occupation_phrase = map_occupation(agent.get("Occupation"))
    income_phrase = map_income(agent.get("Income"))
    education_phrase = map_education(agent.get("Education"))
    svo_phrase = map_svo(agent.get("svo"))

    env_selfeff_score = agent.get("envSelfEfficacy_pre", "unknown")
    env_attitude_score = agent.get("envAttitude_pre", "unknown")
    env_motivation_score = agent.get("envMotivation_pre", "unknown")
    sensitivity_score = agent.get("sensitivity_score", "unknown")

    # Convert numeric psychological scores into coarse descriptive labels.
    env_selfeff_desc = describe_1to5_score(env_selfeff_score)
    env_attitude_desc = describe_1to5_score(env_attitude_score)
    sensitivity_desc = describe_1to5_score(sensitivity_score)
    env_motivation_desc = describe_env_motivation(env_motivation_score)

    return DESCRIPTION_TEMPLATE.format(
        sex=sex,
        age=age,
        ethnicity=ethnicity,
        occupation_phrase=occupation_phrase,
        education_phrase=education_phrase,
        income_phrase=income_phrase,
        svo_phrase=svo_phrase,
        env_selfeff_desc=env_selfeff_desc,
        env_selfeff_score=env_selfeff_score,
        env_attitude_desc=env_attitude_desc,
        env_attitude_score=env_attitude_score,
        env_motivation_desc=env_motivation_desc,
        env_motivation_score=env_motivation_score,
        sensitivity_desc=sensitivity_desc,
        sensitivity_score=sensitivity_score,
    )


def wait_for_file(path: str | Path, timeout: float = WAIT_TIMEOUT_SECONDS) -> None:
    """Wait for a file to appear within the given timeout."""
    target_path = Path(path)
    start_time = time.time()

    while not target_path.exists():
        if time.time() - start_time > timeout:
            raise TimeoutError(f"File {target_path} not created in time.")
        time.sleep(WAIT_INTERVAL_SECONDS)


def process_agent_descriptions(num_agents: int, control: bool) -> list[str] | None:
    """Generate and save profile descriptions for control or intervention agents."""
    if control:
        input_file = f"agent_data_control_{num_agents}.json"
        output_file = f"character_control_{num_agents}.json"
    else:
        input_file = f"agent_data_intervention_{num_agents}.json"
        output_file = f"character_intervention_{num_agents}.json"

    input_path = PROFILE_DIR / input_file
    output_path = PROFILE_DIR / output_file

    if not PROFILE_DIR.exists():
        print(f"Folder {PROFILE_DIR} does not exist.")
        return None

    if not input_path.exists():
        print(f"{input_file} does not exist.")
        return None

    if output_path.exists():
        # Reuse existing descriptions to avoid regenerating the same profile file.
        print(f"{output_file} has existed.")
        try:
            with output_path.open("r", encoding="utf-8") as file:
                character_data = json.load(file)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"Failed to read existing output file: {output_path}. Error: {exc}")
            return None
        return list(character_data.values())

    try:
        with input_path.open("r", encoding="utf-8") as file:
            agents = json.load(file)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Failed to read input file: {input_path}. Error: {exc}")
        return None

    character_data: dict[str, str] = {}
    # Store descriptions with 1-based string IDs to match the agent numbering convention.
    for idx, agent in enumerate(agents, start=1):
        description = generate_description(agent)
        print(f"Processed Agent {idx}:")
        print(description)
        print("-" * 50)
        character_data[str(idx)] = description

    try:
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(character_data, file, ensure_ascii=False, indent=4)
    except OSError as exc:
        print(f"Failed to write output file: {output_path}. Error: {exc}")
        return None

    wait_for_file(output_path)
    print(f"All agent descriptions have been saved to '{output_file}'.")
    return list(character_data.values())


if __name__ == "__main__":
    file_pairs = [
        ("long_term_3/agent_data_base.json", "long_term_3/character_base.json"),
        ("long_term_3/agent_data_T1.json", "long_term_3/character_T1.json"),
        ("long_term_3/agent_data_T2.json", "long_term_3/character_T2.json"),
        ("long_term_3/agent_data_T3.json", "long_term_3/character_T3.json"),
    ]

    # Batch-generate character files for the predefined long-term experiment inputs.
    for input_file_name, output_file_name in file_pairs:
        input_path = Path(input_file_name)
        output_path = Path(output_file_name)

        if not input_path.exists():
            print(f"Input file not found: {input_path}")
            continue

        try:
            with input_path.open("r", encoding="utf-8") as file:
                agents = json.load(file)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"Failed to read input file: {input_path}. Error: {exc}")
            continue

        character_data: dict[str, str] = {}
        for idx, agent in enumerate(agents, start=1):
            description = generate_description(agent)
            character_data[str(idx)] = description

        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with output_path.open("w", encoding="utf-8") as file:
                json.dump(character_data, file, ensure_ascii=False, indent=4)
        except OSError as exc:
            print(f"Failed to write output file: {output_path}. Error: {exc}")
            continue

        print(f"Saved: {output_path}")
