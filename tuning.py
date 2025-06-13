import pycona as ca
from experiments.utils.experiment_utils import BENCHMARKS, FEATURES
from experiments.runners.classifier_experiment import construct_benchmark, construct_feature_representation
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import argparse
from runexp import Runner
from dataclasses import dataclass
import json
import csv
import os

# Define valid options
valid_benchmarks = ['jobshop']
valid_features = FEATURES
valid_classifiers = ['mlp', 'logistic_regression', 'svm']

@dataclass
class TuneConfig:
    benchmark: str
    feature: str
    classifier: str

def create_dataset(benchmark: str, features: str) -> Tuple[list, list]:
    instance, oracle = construct_benchmark(benchmark)
    instance.construct_bias()
    datasetX = []
    datasetY = []

    feature_representation = construct_feature_representation(features)
    feature_representation.instance = instance
    
    for c in instance.bias:
        datasetX.append(feature_representation.featurize_constraint(c))
        if c in set(oracle.constraints):
            datasetY.append(1)
        else:
            datasetY.append(0)

    return datasetX, datasetY

def get_classifier_param_grid(classifier_name: str) -> Dict[str, Any]:
    if classifier_name == 'mlp':
        return {
            'hidden_layer_sizes': [(8,), (16,), (32,), (64,), (128,), (8, 8), (16, 16), (32, 32), (64, 64)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01, 0.1],
            'learning_rate_init': [0.001, 0.01, 0.1]
        }
    elif classifier_name == 'logistic_regression':
        return {
            'C': [0.1, 1.0, 10.0, 100.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'max_iter': [1000, 2000, 3000, 4000, 5000],
        }
    elif classifier_name == 'svm':
        return {
            'C': [0.1, 1.0, 10.0, 100.0],
            'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
            'gamma': ['scale', 'auto', 0.1, 0.01, 0.001, 0.0001],
            'probability': [True],
            'class_weight': ['balanced', None]
        }
    else:
        raise ValueError(f"Unsupported classifier: {classifier_name}")

def get_classifier(classifier_name: str):
    if classifier_name == 'mlp':
        return MLPClassifier(random_state=42)
    elif classifier_name == 'logistic_regression':
        return LogisticRegression(random_state=42)
    elif classifier_name == 'svm':
        return SVC(random_state=42)
    else:
        raise ValueError(f"Unsupported classifier: {classifier_name}")

def write_results(results, output_file, n_top=None):
    """
    Write the results of GridSearchCV to a CSV file.
    
    Args:
        results: GridSearchCV results dictionary
        output_file: Path to output CSV file
        n_top: Number of top results to write (default: all)
    """
    if n_top is None:
        n_top = len(results["rank_test_balanced_accuracy"])
    
    # Get all parameter names from the first result
    params = list(results["params"][0].keys())
    
    # Define all metrics we want to save
    metrics = [
        'rank_test_balanced_accuracy',
        'mean_test_balanced_accuracy',
        'std_test_balanced_accuracy',
        'mean_test_f1',
        'std_test_f1',
        'mean_test_f1_weighted',
        'std_test_f1_weighted',
        'mean_test_accuracy',
        'std_test_accuracy',
        'mean_fit_time',
        'std_fit_time'
    ]
    
    # Create fieldnames for CSV
    fieldnames = metrics + params
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results["rank_test_balanced_accuracy"] == i)
            for candidate in candidates:                
                # Prepare row for CSV
                row = {
                    metric: results[metric][candidate] for metric in metrics
                }
                row.update(results["params"][candidate])
                
                writer.writerow(row)

def tune_classifier(benchmark: str, features: str, classifier_name: str) -> Dict[str, Any]:
    # Create dataset
    X, y = create_dataset(benchmark, features)
    X = np.array(X)
    y = np.array(y)
    
    # Get classifier and parameter grid
    classifier = get_classifier(classifier_name)
    param_grid = get_classifier_param_grid(classifier_name)
    
    # Create GridSearchCV object
    grid_search = GridSearchCV(
        estimator=classifier,
        param_grid=param_grid,
        scoring={
            'f1': 'f1',
            'f1_weighted': 'f1_weighted',
            'balanced_accuracy': 'balanced_accuracy',
            'accuracy': 'accuracy'
        },
        refit='balanced_accuracy',  # Use balanced accuracy for selecting best model
        cv=5,
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )
    
    # Perform grid search
    grid_search.fit(X, y)
    
    return grid_search

def tune(config: TuneConfig):
    print(f"\nTuning {config.classifier} on {config.benchmark} with {config.feature} features...")
    
    try:
        grid_search = tune_classifier(config.benchmark, config.feature, config.classifier)
        
        # Create tuning_results directory if it doesn't exist
        os.makedirs('tuning_results', exist_ok=True)
        
        # Write results to CSV in tuning_results directory
        output_file = os.path.join('tuning_results', f'tuning_results_{config.feature}_{config.classifier}_{config.benchmark}.csv')
        write_results(grid_search.cv_results_, output_file)
        print(f"\nResults saved to {output_file}")
        
    except Exception as e:
        print(f"Error during tuning: {str(e)}")

def TuneParser():

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Configuration file, should be json-formatted")
    parser.add_argument("--output", type=str, default="results_dir", help="Directory to output results of experiments")
    parser.add_argument("-s", "--save_results", action="store_true", help="Whether to save results of experiments")
    parser.add_argument("-u", "--unravel", action="store_true", help="Whether to unravel config file to run experiments in a batch (will unravel lists in configuration file to separate configs)")
    parser.add_argument("--parallel", action="store_true", help="Wheter to run experiments in paralell, only useful if `--unravel` is True")
    parser.add_argument("--num-workers", action="store", type=int, help=f"Number of threads to use for parallelization (=nb of experiments running in parallel")
    parser.add_argument("--memory_limit", action="store", type=int, default=-1, help="Memory limit in MB to use by each experiment, only works on Linux.")
    return parser

class TunerRunner(Runner):
    def description(self, config):
        """Return a description of the current experiment"""
        return f"Running {config['classifier']} on {config['benchmark']} with {config['feature']} features"

    def make_kwargs(self, config):
        """Convert config dictionary to ExperimentConfig object"""
        
        # Create ExperimentConfig object with defaults for optional parameters
        experiment_config = TuneConfig(
            benchmark=config['benchmark'],
            feature=config.get('feature', 'rel_dim_block'),
            classifier=config.get('classifier', 'decision_tree'),
        )
        
        return {'config': experiment_config}


if __name__ == "__main__":
    parser = TuneParser()
    args = parser.parse_args()
    
    # Load config file
    with open(args.config, "r") as f:
        config = json.loads(f.read())
    
    runner = TunerRunner(
        func=tune,
        output=args.output,
        memory_limit=args.memory_limit,
        printlog=True,
        save_results=args.save_results
    )
        
    if args.unravel is True:
        runner.run_batch(config, parallel=args.parallel, num_workers=args.num_workers)
    else:
        runner.run_one(config)


    








