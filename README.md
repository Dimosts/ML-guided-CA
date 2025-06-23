# ML-guided-CA

This repository contains the implementation and experiments for the paper "ML-guided Interactive Constraint Acquisition", which is under review in the Journal of Artificial Intelligence Research (JAIR).


## Running the Experiments

The repository includes experiments for four research questions (Q1-Q4). Below are the instructions for running each experiment.

### Q1: Pure Classification Experiment

To run the pure classification experiment of Q1, execute:

```bash
python classification_experiment.py
```

### Q2-Q3: ML-guided Query Generation

To run experiments investigating the use of ML classifiers to guide top-level query generation (using probabilities or direct class labels):

```bash
python run_experiments.py classifiers ./configs/classifier_exp/qgen.json --unravel
```

**Arguments:**
- `classifiers`: Specifies that this experiment focuses on guiding interactive CA
- `./configs/classifier_exp/qgen.json`: Configuration file containing all combinations of benchmarks/classifiers/feature representations/objective functions for query generation guidance
- `--unravel`: Unravels the configuration file to generate all experiment combinations

### Q4: ML-guided All Layers

To run experiments using ML classifiers to guide all layers of interactive CA:

#### Guiding All Layers Together (Guide All)

```bash
python run_experiments.py classifiers ./configs/classifier_exp/guide_all.json --unravel
```

**Arguments:**
- `classifiers`: Specifies that this experiment focuses on guiding interactive CA
- `./configs/classifier_exp/guide_all.json`: Configuration file for guiding all layers
- `--unravel`: Unravels the configuration file

#### FindScope Experiments

```bash
python run_experiments.py findscope ./configs/findscope_exp/all.json --unravel
```

**Arguments:**
- `findscope`: Specifies that this experiment focuses on FindScope
- `./configs/findscope_exp/all.json`: Configuration file including all combinations of benchmarks/findscope functions
- `--unravel`: Unravels the configuration file

#### FindC Experiments

```bash
python run_experiments.py findc ./configs/findc_exp/all.json --unravel
```

**Arguments:**
- `findc`: Specifies that this experiment focuses on FindC
- `./configs/findc_exp/all.json`: Configuration file including all combinations of benchmarks/findc functions
- `--unravel`: Unravels the configuration file

## Processing and Plotting Results

After running the experiments, you can process and plot the results using a three-step workflow:

### Step 1: Process and Aggregate Raw Results

The `process_*` scripts aggregate raw experimental results from CSV files into averaged results with standard errors:

#### For Classifier Experiments:
```bash
python process_classifier_results.py <results_folder>
```

#### For FindScope Experiments:
```bash
python process_findscope_results.py <results_folder>
```

#### For FindC Experiments:
```bash
python process_findc_results.py <results_folder>
```

#### For Baseline Experiments:
```bash
python process_baseline_results.py <results_folder>
```

#### For Hyperparameter Tuning:
```bash
python process_tuning_results.py <results_folder>
```

**What these scripts do:**
- Read raw CSV files from experiment results
- Calculate means and standard errors across runs
- Output aggregated CSV files (e.g., `*_averaged_results.csv`, `*_stderr_results.csv`)

### Step 2: Parse Aggregated Results to JSON

The `parse_*` scripts convert the aggregated CSV files into JSON format optimized for plotting:

#### For Q1 (Classification Experiment):
```bash
python experiments/processing/parse_classification_results.py
```

#### For Q2-Q3 (Query Generation):
```bash
python experiments/processing/parse_averaged_qgen_results.py
```

#### For Q4 (Guide All):
```bash
python experiments/processing/parse_averaged_guide_all_results.py
```

#### For Q4 (FindScope):
```bash
python experiments/processing/parse_averaged_findscope_results.py
```

#### For Q4 (FindC):
```bash
python experiments/processing/parse_averaged_findc_results.py
```

#### For Baseline Results:
```bash
python experiments/processing/parse_averaged_baseline_results.py
```

**What these scripts do:**
- Read the aggregated CSV files from Step 1
- Convert data to JSON format optimized for plotting
- Save results to `experiments/parsed_results/` directory

### Step 3: Generate Plots

After processing and parsing the results, generate plots using the following scripts:

#### For Q1 (Classification Results):
```bash
python experiments/plotting/plot_q1_results.py
```

#### For Q2 (Query Generation Results):
```bash
python experiments/plotting/plot_q2_results.py
```

#### For Q3 (Probability vs Class Results):
```bash
python experiments/plotting/plot_q3_results.py
```

#### For Q3 (Probability vs Probability Previous):
```bash
python experiments/plotting/plot_q3_proba_vs_proba_prev_results.py
```

#### For Q4 (All Layers Results):
```bash
python experiments/plotting/plot_q4_all_results.py
```

#### For Q4 (FindScope Results):
```bash
python experiments/plotting/plot_q4_findscope_results.py
```

#### For Q4 (FindC Results):
```bash
python experiments/plotting/plot_q4_findc_results.py
```

### Complete Workflow Example

For a typical experiment, the complete workflow would be:

```bash
# 1. Run experiments
python run_experiments.py classifiers ./configs/classifier_exp/qgen.json --unravel

# 2. Process raw results
python process_classifier_results.py results/

# 3. Parse to JSON
python experiments/processing/parse_averaged_qgen_results.py

# 4. Generate plots
python experiments/plotting/plot_q2_results.py
```

### Output Locations

- **Raw experiment results:**
  - For Q1 (classification_experiment.py): `results/` directory (unless otherwise specified)
  - For Q2-Q3 (run_experiments.py classifiers ... with guide='qgen'): `results_qgen/` directory (with subdirectories per benchmark)
  - For Q4, guide all layers experiment (run_experiments.py classifiers ... with guide='all'): `results_all/` directory (with subdirectories per benchmark)
  - For FindScope experiment: `findscope_results/` directory (with subdirectories per benchmark)
  - For FindC experiment: `findc_results/` directory (with subdirectories per benchmark)
  - For tuning.py: `tuning_results/` directory
- **Aggregated results:**
  - All `process_*` scripts output to `experiments/aggregated_results/` directory
  - `process_tuning_results.py` outputs to `experiments/aggregated_tuning_results/` directory
- **Parsed results:** JSON files in `experiments/parsed_results/` directory
- **Plots:** Generated in `experiments/plots/` directory, organized by experiment type
- **Pre-computed aggregated results:** Available in `experiments/aggregated_results/` directory
- **Tuning results:** Available in `experiments/aggregated_tuning_results/` directory

## Results

The repository includes pre-computed results from our experiments:

- **Aggregated results**: Available in the `experiments/aggregated_results/` folder
- **Parsed results**: Available in the `experiments/parsed_results/` folder as JSON files
- **Plots**: Both the ones presented in the paper and additional visualizations are available in the `experiments/plots/` folder
- **Tuning results**: Available in the `experiments/aggregated_tuning_results/` folder

## Dependencies

Install the required dependencies using:

```bash
pip install -r requirements.txt
```