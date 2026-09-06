# Understanding the Persistence of Pro-environmental Nudges through an Integrated LLM-Human Approach


## Environment Setup
To set up the environment and dependencies, run the following commands:

### Python Environment
```bash
conda create -n ProEnvironment python=3.10
conda activate ProEnvironment
pip install -r requirements.txt
```

## API Configuration
We use environment variables to securely manage API credentials.

**OpenAI-compatible endpoint:**
```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_API_BASE="your-base-url"  # Optional for the default OpenAI endpoint
```

**DeepSeek endpoint:**
```bash
export DEEPSEEK_API_KEY="your-api-key"
export DEEPSEEK_API_BASE="your-base-url"
```

## Running the Experiments
Run the three main experiments from the repository root with:

```bash
./run.sh
```

`run.sh` uses `deepseek-v4-flash` by default. Override the model shared by all three studies with, for example, `MODEL_NAME=gpt-4o-2024-11-20 ./run.sh`. It runs the main experiment for each study. Study 3 reuses the matching Study 1 Day 1 results, so keep the same model name when running those studies separately.

### Study 1: Benchmarking LLMs against human responses to pro-environmental nudges

```bash
cd study1/nudge_replication
python run.py --model_name deepseek-v4-flash
```

#### Parameters
| **Parameter**    | **Type** | **Description** |
|------------------|----------|-----------------|
| `--model_name`   | `str`   | The name of the model used in the experiment. |
| `--temperature`  | `float` | The sampling temperature for response generation. |
| `--num_threads`  | `int`   | The number of threads used for parallel execution. |
| `--max_attempts` | `int`   | The maximum number of retry attempts for each response generation. |
| `--output_dir`   | `str`   | The directory used to save experiment results. |
| `--num_chars`    | `int`   | The number of agents to process. `-1` uses the sample size recorded for each experiment. |

> Shared parameters (for example, `--model_name`) are documented only once in Study 1 for brevity.

#### Result Analysis
```bash
cd study1/result_analysis
python result_analysis.py
```

#### Long-term Experiments

Run the three long-term field-experiment replications with:

```bash
cd study1/nudge_replication
python run_long_term_0.py --model_name deepseek-v4-flash
python run_long_term_2.py --model_name deepseek-v4-flash
python run_long_term_3.py --model_name deepseek-v4-flash
```

Analyze result with:
```bash
cd study1/result_analysis
python result_analysis_long.py
```

### Study 2: Human behavioural experiment tests simulated predictions of decay and persistence

```bash
cd study2/human_experiments
python run_human.py --model_name deepseek-v4-flash
```

#### Parameters
| **Parameter**  | **Type** | **Description** |
| -------------- | -------- | --------------- |
| `--num_days` | `int` | The number of simulation days. |
| `--questions_per_day` | `int` | The number of questions/tasks asked per agent each day. |
| `--group` | `str` | One experimental group, or `all` to run all groups. |
| `--seed` | `int` | The random seed used for reproducibility. |

#### Result Analysis
```bash
cd study2/result_analysis
python result_analysis.py
```

### Study 3: Longitudinal simulations reveal temporal decay of nudge effects

```bash
cd study3/longitudinal_simulation
python run_long.py --model_name deepseek-v4-flash
python run_freq.py --model_name deepseek-v4-flash
python run_freq.py --model_name deepseek-v4-flash --frequency 6
```

Study 3 loads the matching Study 1 profiles and Day 1 responses from `study1/nudge_replication/res/<model_name>_res/`.


#### Parameters
| **Parameter**  | **Type** | **Description** |
| -------------- | -------- | --------------- |
| `--num_rounds` | `int` | Number of simulation rounds (days). |
| `--start_id`   | `int` | Row index at which to start the experiments. |
| `--groups` | `str` | Run `control`, `intervention`, or both groups. |
| `--frequency` | `int` | Intervention interval in days for `run_freq.py`. |

#### Result Analysis
```bash
cd study3/result_analysis
python result_analysis.py
```

## Data

Each `study*/data/` directory contains the final de-identified data retained for that study and a corresponding `data_dictionary*.xlsx`. Each `study*/result_analysis/` directory contains the analysis code and final generated `.txt` report.
