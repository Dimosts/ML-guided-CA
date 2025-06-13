import pycona as ca
from dataclasses import dataclass
import os
import argparse

from ..utils.experiment_utils import BENCHMARKS, ALGORITHMS, CLASSIFIERS, OBJECTIVES, FEATURES, FINDSCOPE, Algorithm, ExperimentConfig # experiment constants
from ..utils.experiment_utils import construct_benchmark, construct_classifier, construct_feature_representation, create_objective, create_algorithm, create_findscope, create_findc # experiment functions
from pycona.find_scope.findscope_obj import split_proba, split_half, cb_split_proba, cb_split_half_vars
from pycona.find_constraint.findc_obj import findc_obj_splithalf, findc_obj_proba


@dataclass
class BaselineConfig(ExperimentConfig):
    def validate(self):
        return super().validate()

def _save_baseline_results(ga, config: BaselineConfig):
    # Construct filename components
    filename = f"{config.benchmark}_{config.algorithm.value}.csv"

    # Create results directory
    dir = "baseline_results"

    results_dir = os.path.join(dir, config.benchmark)
    os.makedirs(results_dir, exist_ok=True)

    # Write results to file in results directory
    ga.env.metrics.write_to_file(os.path.join(results_dir, filename))


def run_baseline_experiment(config: BaselineConfig) -> None:
    if not config.validate():
        print(f"Invalid configuration: {config}")
        return
    instance, oracle = construct_benchmark(config.benchmark)
    from pycona.query_generation.qgen_obj import obj_max_viol

    for _ in range(config.n_runs):
        env = ca.ActiveCAEnv(qgen=ca.PQGen(objective_function=obj_max_viol))
        ga = create_algorithm(config, env)        
        ga.learn(instance, oracle, verbose=config.verbose)

        if config.verbose > 0:
            print(ga.env.metrics.statistics)
        _save_baseline_results(ga, config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run constraint acquisition experiments')
    parser.add_argument('-b', '--benchmark', 
                       choices=BENCHMARKS,
                       required=True,
                       help='The benchmark to run')
    parser.add_argument('-a', '--algorithm', choices=ALGORITHMS,
                        help='The algorithm to use')
    parser.add_argument('-n', '--n_runs', type=int, default=25,
                        help='Number of runs to perform (default: 25)')
    parser.add_argument('-v', '--verbose', type=int, default=0,
                        help='Verbose level (default: 0)')

    args = parser.parse_args()

    config = BaselineConfig(
        benchmark=args.benchmark,
        algorithm=Algorithm(args.algorithm),
        n_runs=args.n_runs,
        verbose=args.verbose
    )
    
    run_baseline_experiment(config)


