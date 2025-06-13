import argparse
import json
import os
from datetime import datetime
from experiments.runners.classifier_experiment import run_classifiers_experiment, ClassifierConfig, Algorithm
from experiments.runners.findscope_experiment import run_findscope_experiment, FindScopeConfig
from experiments.runners.findc_experiments import run_findc_experiment, FindCConfig
from experiments.runners.baseline_experiment import run_baseline_experiment, BaselineConfig

def myparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str, help="Experiment to run") #classifiers, findscope, findc, baseline
    parser.add_argument("config", type=str, help="Configuration file, should be json-formatted")
    parser.add_argument("--output", type=str, default="results_dir", help="Directory to output results of experiments")
    parser.add_argument("-s", "--save_results", action="store_true", help="Whether to save results of experiments")
    parser.add_argument("-u", "--unravel", action="store_true", help="Whether to unravel config file to run experiments in a batch (will unravel lists in configuration file to separate configs)")
    return parser

def unravel_config(config):
    """Unravel a configuration dictionary into a list of configurations.
    If any value in the config is a list, create separate configs for each value."""
    if not isinstance(config, dict):
        return [config]
    
    # Find all list values in the config
    list_keys = [k for k, v in config.items() if isinstance(v, list)]
    
    if not list_keys:
        return [config]
    
    # Get the first list key
    key = list_keys[0]
    values = config[key]
    
    # Create new configs for each value
    new_configs = []
    for value in values:
        new_config = config.copy()
        new_config[key] = value
        new_configs.extend(unravel_config(new_config))
    
    return new_configs

def run_experiment(experiment_type, config, output_dir, save_results):
    """Run a single experiment based on the configuration"""
    print(f"Running experiment: {experiment_type}")
    print(f"Configuration: {json.dumps(config, indent=2)}")
    
    if experiment_type == "classifiers":
        algorithm = Algorithm(config['algorithm'])
        experiment_config = ClassifierConfig(
            benchmark=config['benchmark'],
            algorithm=algorithm,
            objective=config.get('objective', 'obj_proba'),
            classifier=config.get('classifier', 'decision_tree'),
            features=config.get('features', 'rel_dim_block'),
            guide=config.get('guide', 'qgen'),
            n_runs=config.get('n_runs', 25),
            verbose=config.get('verbose', 0)
        )
        results = run_classifiers_experiment(config=experiment_config)
    
    elif experiment_type == "findscope":
        algorithm = Algorithm(config['algorithm'])
        experiment_config = FindScopeConfig(
            benchmark=config['benchmark'],
            algorithm=algorithm,
            objective=config.get('objective', 'max_viol'),
            findscope=config.get('findscope', 'findscope2'),
            features=config.get('features', 'rel_dim_block'),
            n_runs=config.get('n_runs', 25),
            verbose=config.get('verbose', 0)
        )
        results = run_findscope_experiment(config=experiment_config)
    
    elif experiment_type == "findc":
        algorithm = Algorithm(config['algorithm'])
        experiment_config = FindCConfig(
            benchmark=config['benchmark'],
            algorithm=algorithm,
            objective=config.get('objective', 'max_viol'),
            findc=config.get('findc', 'findc'),
            features=config.get('features', 'rel_dim_block'),
            n_runs=config.get('n_runs', 25),
            verbose=config.get('verbose', 0)
        )
        results = run_findc_experiment(config=experiment_config)
    
    elif experiment_type == "baseline":
        algorithm = Algorithm(config['algorithm'])
        experiment_config = BaselineConfig(
            benchmark=config['benchmark'],
            algorithm=algorithm,
            n_runs=config.get('n_runs', 25),
            verbose=config.get('verbose', 0)
        )
        results = run_baseline_experiment(config=experiment_config)
    
    else:
        raise ValueError(f"Invalid experiment type: {experiment_type}")

    if save_results:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{experiment_type}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Save results
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {filepath}")
    
    return results

if __name__ == "__main__":
    parser = myparser()
    args = parser.parse_args()
    
    # Load config file
    with open(args.config, "r") as f:
        config = json.loads(f.read())
    
    if args.unravel:
        # Unravel the configuration into a list of configs
        configs = unravel_config(config)
        print(f"Unraveled {len(configs)} configurations")
        
        # Run experiments for each configuration
        for i, single_config in enumerate(configs, 1):
            print(f"\nRunning configuration {i}/{len(configs)}")
            results = run_experiment(
                experiment_type=args.experiment,
                config=single_config,
                output_dir=args.output,
                save_results=args.save_results
            )
    else:
        # Run single experiment
        results = run_experiment(
            experiment_type=args.experiment,
            config=config,
            output_dir=args.output,
            save_results=args.save_results
        )
    
    print("\nAll experiments completed successfully!") 