from enum import Enum
from typing import Tuple
from dataclasses import dataclass

import pycona as ca
from pycona.benchmarks import construct_murder_problem, construct_nurse_rostering, construct_sudoku, construct_examtt_simple, \
    construct_jsudoku, construct_gtsudoku, construct_nqueens, construct_job_shop_scheduling_problem, construct_random495
from pycona.query_generation.qgen_obj import obj_class, obj_proba, obj_proba2, obj_proba3, obj_max_viol
from pycona.predictor.feature_representation import FeaturesRelDimBlock, FeaturesRelDim, FeaturesSimpleRel, FeatureRepresentation
from pycona.predictor.predictor import CountsPredictor

from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.ensemble._forest import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


BENCHMARKS = ['sudoku', 'jsudoku', 'gtsudoku', 'exam_timetable', 'nurse_rostering', 'jobshop', 'random']
ALGORITHMS = ['growacq', 'quacq']
CLASSIFIERS = ['random_forest', 'decision_tree', 'svm', 'logistic_regression', 'mlp', 'naive_bayes', 'counts', 'none']
OBJECTIVES = ['obj_class', 'obj_proba', 'obj_proba2', 'obj_proba3', 'max_viol']
FEATURES = ['rel_dim_block', 'rel_dim', 'simple_rel', 'none']
FINDSCOPE = ['findscope2', 'guided-findscope2', 'cb-findscope', 'cb-findscope-half']
FINDC = ['findc', 'guided-findc']

class Algorithm(Enum):
    GROWACQ = 'growacq'
    QUACQ = 'quacq'

# Benchmark constants
SUDOKU_SIZE = 9
SUDOKU_BLOCK_SIZE = 3

# Experiment constants
N_RUNS = 10

@dataclass
class ExperimentConfig:
    benchmark: str
    algorithm: Algorithm
    objective: str = 'obj_proba'
    classifier: str = 'decision_tree'
    features: str = 'rel_dim_block'
    findscope: str = 'findscope2'
    findc: str = 'findc'
    n_runs: int = N_RUNS
    verbose: int = 0

    def validate(self):
        if self.n_runs <= 0:
            print(f"Invalid n_runs: must be positive")
            return False #ValueError("n_runs must be positive")
        if self.benchmark not in BENCHMARKS:
            print(f"Invalid benchmark: {self.benchmark}")
            return False #ValueError(f"Invalid benchmark: {self.benchmark}")
        if self.algorithm.value not in ALGORITHMS:
            print(f"Invalid algorithm: {self.algorithm}")
            return False #ValueError(f"Invalid algorithm: {self.algorithm}")
        if self.objective not in OBJECTIVES:
            print(f"Invalid objective: {self.objective}")
            return False #ValueError(f"Invalid objective: {self.objective}")
        if self.classifier not in CLASSIFIERS:
            print(f"Invalid classifier: {self.classifier}")
            return False #ValueError(f"Invalid classifier: {self.classifier}")
        elif self.classifier == 'none':
            if self.features != 'none' or self.objective != 'max_viol':
                print(f"None classifier is not supported for {self.features} features and {self.objective} objective")
                return False
        if self.features not in FEATURES:
            print(f"Invalid features: {self.features}")
            return False #ValueError(f"Invalid features: {self.features}")
        elif self.features == 'none':
            if self.classifier != 'none' or self.objective != 'max_viol':
                print(f"None features are not supported for {self.classifier} classifier and {self.objective} objective")
                return False
        elif self.features != 'simple_rel':
            if self.classifier == 'counts':
                print(f"Counts classifier is not supported for {self.features} features")
                return False
        if self.findscope not in FINDSCOPE:
            print(f"Invalid findscope: {self.findscope}")
            return False
        if self.verbose not in [0, 1, 2, 3, 4]:
            print(f"Invalid verbose: {self.verbose}")
            return False
        return True

def construct_feature_representation(features: str) -> FeatureRepresentation:
    if features == 'rel_dim_block':
        return FeaturesRelDimBlock()
    elif features == 'rel_dim':
        return FeaturesRelDim()
    elif features == 'simple_rel':
        return FeaturesSimpleRel()

def construct_benchmark(benchmark: str) -> Tuple[any, any]:
    """Constructs the benchmark instance and returns (instance, oracle)"""
    
    if benchmark == 'sudoku':
        instance, oracle = construct_sudoku(SUDOKU_BLOCK_SIZE, SUDOKU_BLOCK_SIZE, SUDOKU_SIZE)
    elif benchmark == 'nurse_rostering':
        instance, oracle = construct_nurse_rostering(3, 7, 18, 5)
    elif benchmark == 'exam_timetable':
        instance, oracle = construct_examtt_simple(9,5,6,10)
    elif benchmark == 'murder':
        instance, oracle = construct_murder_problem()
    elif benchmark == 'jsudoku':
        instance, oracle = construct_jsudoku()
    elif benchmark == 'gtsudoku':
        instance, oracle = construct_gtsudoku()
    elif benchmark == 'nqueens':
        instance, oracle = construct_nqueens()
    elif benchmark == 'jobshop':
        instance, oracle = construct_job_shop_scheduling_problem(20, 3, 50)
    elif benchmark == 'random':
        instance, oracle = construct_random495()
    else:
        raise ValueError(f"Invalid benchmark: {benchmark}")

    #instance.construct_bias()
    #print("Bias size: ", len(instance.bias))
    #print("vars size: ", len(instance.X))
    #input("Press Enter to continue...")
    return instance, oracle

def construct_classifier(config: ExperimentConfig) -> any:
    classifier = None

    if config.classifier == "random_forest":
        classifier = RandomForestClassifier()
    elif config.classifier == "mlp":
        if config.features == 'rel_dim_block':
            classifier = MLPClassifier(hidden_layer_sizes=tuple([8]), activation='relu', alpha=0.01, solver='adam',
                                   random_state=1, learning_rate_init=0.1)
        elif config.features == 'rel_dim':
            classifier = MLPClassifier(hidden_layer_sizes=tuple([8,8]), activation='tanh', alpha=0.0001, solver='adam',
                                   random_state=1, learning_rate_init=0.01)
        elif config.features == 'simple_rel':
            classifier = MLPClassifier(hidden_layer_sizes=tuple([8]), activation='relu', alpha=0.01, solver='adam',
                                   random_state=1, learning_rate_init=0.1)
        else:
            raise ValueError(f"Invalid features for MLP classifier: {config.features}")
    elif config.classifier == "naive_bayes":
        classifier = GaussianNB()
    elif config.classifier == "svm":
        if config.features == 'rel_dim_block':
            classifier = SVC(kernel='linear', C=0.1, class_weight='balanced', gamma=0.1, probability=True)
        elif config.features == 'rel_dim':
            classifier = SVC(kernel='rbf', C=100, class_weight='balanced', gamma=0.1, probability=True)
        elif config.features == 'simple_rel':
            classifier = SVC(kernel='linear', C=0.1, class_weight='balanced', gamma=0.1, probability=True)
        else:
            raise ValueError(f"Invalid features for SVM classifier: {config.features}")
    elif config.classifier == "decision_tree":
        classifier = DecisionTreeClassifier()
    elif config.classifier == "logistic_regression":
        classifier = LogisticRegression(C=100, penalty='l2', max_iter=2000, solver='liblinear')
    elif config.classifier == "knn":
        classifier = KNeighborsClassifier()
    elif config.classifier == "xgboost":
        classifier = XGBClassifier()
    elif config.classifier == "counts":
        classifier = CountsPredictor()
    else:
        raise ValueError(f"Invalid classifier: {config.classifier}")

    return classifier

def create_algorithm(config: ExperimentConfig, env) -> any:
    """Creates the appropriate learning algorithm based on configuration"""
    if config.algorithm == Algorithm.GROWACQ:
        ga = ca.GrowAcq(env)
    elif config.algorithm == Algorithm.QUACQ:
        ga = ca.QuAcq(env)
    else:
        raise ValueError(f"Invalid algorithm: {config.algorithm}")
    
    return ga

def create_objective(config: ExperimentConfig):
    if config.objective == 'obj_class':
        objective = obj_class
    elif config.objective == 'obj_proba':
        objective = obj_proba
    elif config.objective == 'obj_proba2':
        objective = obj_proba2
    elif config.objective == 'obj_proba3':
        objective = obj_proba3
    elif config.objective == 'max_viol':
        objective = obj_max_viol
    else:
        raise ValueError(f"Invalid objective: {config.objective}")

    return objective

def create_findscope(config: ExperimentConfig):
    from pycona.find_scope.findscope_obj import split_proba, split_half, cb_split_proba, cb_split_half_vars
    if config.findscope == 'findscope2':
        return ca.FindScope2(split_func=split_half)
    elif config.findscope == 'guided-findscope2':
        return ca.FindScope2(split_func=split_proba)
    elif config.findscope == 'cb-findscope':
        return ca.CBFindScope(split_func=cb_split_proba)
    elif config.findscope == 'cb-findscope-half':
        return ca.CBFindScope(split_func=cb_split_half_vars)
    else:
        raise ValueError(f"Invalid findscope: {config.findscope}")

def create_findc(config: ExperimentConfig):
    from pycona.find_constraint.findc_obj import findc_obj_splithalf, findc_obj_proba
    if config.findc == 'findc':
        return ca.FindC(findc_obj=findc_obj_splithalf)
    elif config.findc == 'guided-findc':
        return ca.FindC(findc_obj=findc_obj_proba)
    else:
        raise ValueError(f"Invalid findc: {config.findc}")



