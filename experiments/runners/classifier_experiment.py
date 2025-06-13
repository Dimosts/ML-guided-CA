import pycona as ca
from dataclasses import dataclass
import os
import argparse

from ..utils.experiment_utils import BENCHMARKS, ALGORITHMS, CLASSIFIERS, OBJECTIVES, FEATURES, FINDSCOPE, Algorithm, ExperimentConfig # experiment constants
from ..utils.experiment_utils import construct_benchmark, construct_classifier, construct_feature_representation, create_objective, create_algorithm, create_findscope, create_findc # experiment functions
from pycona.find_scope.findscope_obj import split_proba, split_half, cb_split_proba, cb_split_half_vars
from pycona.find_constraint.findc_obj import findc_obj_splithalf, findc_obj_proba


@dataclass
class ClassifierConfig(ExperimentConfig):
    guide: str = 'qgen'

    def validate(self):
        if self.guide != 'all' and self.guide != 'qgen':
            print(f"Invalid guide: {self.guide}")
            return False
        if self.objective == 'max_viol':
            if self.features != 'none' or self.classifier != 'none':
                print(f"Max violation objective is not supported for {self.features} features and {self.classifier} classifier")
                return False
        return super().validate()

def _save_classifier_results(ga, config: ClassifierConfig):
    # Construct filename components
    if config.guide != 'all':
        filename = f"{config.benchmark}_{config.algorithm.value}_{config.objective}_{config.features}_{config.classifier}_{config.guide}.csv"
    else:
        filename = f"{config.benchmark}_{config.algorithm.value}_{config.objective}_{config.features}_{config.classifier}_{config.guide}.csv"

    # Create results directory
    if config.guide == 'all':
        dir = "results_all"
    else:
        dir = "results_qgen"

    results_dir = os.path.join(dir, config.benchmark)
    os.makedirs(results_dir, exist_ok=True)

    # Write results to file in results directory
    ga.env.metrics.write_to_file(os.path.join(results_dir, filename))


def run_classifiers_experiment(config: ClassifierConfig) -> None:
    if not config.validate():
        print(f"Invalid configuration: {config}")
        return
    instance, oracle = construct_benchmark(config.benchmark)
    objective = create_objective(config)

    if config.objective != 'max_viol':
        classifier = construct_classifier(config)
        feature_representation = construct_feature_representation(config.features)
        
    else:
        classifier = None
        feature_representation = None

    for _ in range(config.n_runs):
        if config.objective != 'max_viol':
            if config.guide == 'qgen':
                env = ca.ProbaActiveCAEnv(qgen=ca.PQGen(objective_function=objective), feature_representation=feature_representation, classifier=classifier, find_scope=ca.FindScope2(split_func=split_half), findc=ca.FindC(findc_obj=findc_obj_splithalf))
            else:
                env = ca.ProbaActiveCAEnv(qgen=ca.PQGen(objective_function=objective), feature_representation=feature_representation, classifier=classifier, find_scope=ca.CBFindScope(split_func=cb_split_proba), 
                                      findc=ca.FindC(findc_obj=findc_obj_proba))
        else:
            if config.guide == 'qgen':
                env = ca.ActiveCAEnv(qgen=ca.PQGen(objective_function=objective), feature_representation=feature_representation)
            else:
                env = ca.ActiveCAEnv(qgen=ca.PQGen(objective_function=objective), find_scope=ca.FindScope2(split_func=split_half), findc=ca.FindC(findc_obj=findc_obj_splithalf))

        ga = create_algorithm(config, env)
        
        ga.learn(instance, oracle, verbose=config.verbose)

        if config.verbose > 0:
            print(ga.env.metrics.statistics)
        _save_classifier_results(ga, config)

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
    parser.add_argument('-c', '--classifier', choices=CLASSIFIERS,
                        help='The classifier to use')
    parser.add_argument('-f', '--features', choices=FEATURES, default='rel_dim_block',
                        help='The features to use')
    parser.add_argument('-n', '--n_runs', type=int, default=25,
                        help='Number of runs to perform (default: 25)')
    parser.add_argument('-v', '--verbose', type=int, default=0,
                        help='Verbose level (default: 0)')

    args = parser.parse_args()

    config = ClassifierConfig(
        benchmark=args.benchmark,
        algorithm=Algorithm(args.algorithm),
        objective=args.objective,
        classifier=args.classifier,
        features=args.features,
        n_runs=args.n_runs,
        verbose=args.verbose
    )
    
    run_classifiers_experiment(config)


