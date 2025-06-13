import pycona as ca
from dataclasses import dataclass

import os
import argparse

from ..utils.experiment_utils import BENCHMARKS, ALGORITHMS, OBJECTIVES, FEATURES, FINDC, Algorithm, ExperimentConfig # experiment constants
from ..utils.experiment_utils import construct_benchmark, construct_classifier, construct_feature_representation, create_objective, create_algorithm, create_findc # experiment functions

@dataclass
class FindCConfig(ExperimentConfig):

    def validate(self):
        super().validate()
        if self.findc not in FINDC:
            print(f"Invalid findc: {self.findc}")
            return False
        if self.classifier != 'decision_tree':
            print(f"Findc requires a decision tree classifier")
            return False
        return True
        
def _save_findc_results(ga, config: ExperimentConfig):
    # Construct filename components
    filename = f"{config.benchmark}_{config.algorithm.value}_{config.objective}_{config.features}_{config.findc}.csv"

    # Create results directory
    results_dir = os.path.join("findc_results", config.benchmark)
    os.makedirs(results_dir, exist_ok=True)

    # Write results to file in results directory
    ga.env.metrics.write_to_file(os.path.join(results_dir, filename))


def run_findc_experiment(config: FindCConfig) -> None:
    if not config.validate():
        print(f"Invalid configuration: {config}")
        return
    instance, oracle = construct_benchmark(config.benchmark)
    objective = create_objective(config)
    classifier = construct_classifier(config)
    findc = create_findc(config)
    feature_representation = construct_feature_representation(config.features)
    
    for _ in range(config.n_runs):
        env = ca.ProbaActiveCAEnv(qgen=ca.PQGen(objective_function=objective), feature_representation=feature_representation, classifier=classifier, findc=findc)
        ga = create_algorithm(config, env)
        ga.learn(instance, oracle, verbose=config.verbose)

        if config.verbose > 0:
            print(ga.env.metrics.statistics)
        _save_findc_results(ga, config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run constraint acquisition experiments')
    parser.add_argument('-b', '--benchmark', 
                       choices=BENCHMARKS,
                       required=True,
                       help='The benchmark to run')
    parser.add_argument('-a', '--algorithm', choices=ALGORITHMS,
                        help='The algorithm to use')
    parser.add_argument('-o', '--objective', choices=OBJECTIVES,
                        help='The objective to use')
    parser.add_argument('-fc', '--findc', choices=FINDC,
                        help='The findc to use')
    parser.add_argument('-f', '--features', choices=FEATURES, default='rel_dim_block',
                        help='The features to use')
    parser.add_argument('-n', '--n_runs', type=int, default=25,
                        help='Number of runs to perform (default: 25)')
    parser.add_argument('-v', '--verbose', type=int, default=0,
                        help='Verbose level (default: 0)')

    args = parser.parse_args()

    config = FindCConfig(
        benchmark=args.benchmark,
        algorithm=Algorithm(args.algorithm),
        objective=args.objective,
        findc=args.findc,
        features=args.features,
        n_runs=args.n_runs,
        verbose=args.verbose
    )
    
    run_findc_experiment(config)


