import pycona as ca
from experiments.utils.experiment_utils import BENCHMARKS, FEATURES, CLASSIFIERS, construct_classifier
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
from dataclasses import dataclass
import json
import csv
import os
# Perform cross validation once
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

# Define valid options
valid_benchmarks = ['sudoku', 'exam_timetable', 'nurse_rostering']
valid_features = ['rel_dim_block', 'rel_dim', 'simple_rel']

valid_classifiers = ['random_forest', 'decision_tree', 'logistic_regression', 'mlp', 'naive_bayes', 'svm']

@dataclass
class ClassificationConfig:
    benchmark: str
    features: str
    classifier: str

def create_dataset(config: ClassificationConfig) -> Tuple[list, list]:
    instance, oracle = construct_benchmark(config.benchmark)
    instance.construct_bias()
    datasetX = []
    datasetY = []

    feature_representation = construct_feature_representation(config.features)
    feature_representation.instance = instance
    
    for c in instance.bias:
        datasetX.append(feature_representation.featurize_constraint(c))
        if c in set(oracle.constraints):
            datasetY.append(1)
        else:
            datasetY.append(0)

    return datasetX, datasetY


def write_results(results, filename: str, is_partial: bool = False, benchmark: str = None, feature: str = None, classifier: str = None):
    """
    Write classification results to a CSV file.
    
    Args:
        results: Either a dictionary of metrics (total results) or a list of lists (partial results)
        filename: Path to the output CSV file
        is_partial: Whether these are partial results (True) or total results (False)
        benchmark: Name of the benchmark
        feature: Name of the feature representation
        classifier: Name of the classifier
    """
    if is_partial:
        # For partial results, write each row with percentage and metrics
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Benchmark', 'Feature', 'Classifier', 'Percentage', 'Accuracy', 'Balanced Accuracy', 'F1 Score'])
            percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            for p, metrics in zip(percentages, results):
                writer.writerow([benchmark, feature, classifier, p] + metrics)
    else:
        # For total results, write single row with all metrics
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Benchmark', 'Feature', 'Classifier', 'Accuracy', 'Balanced Accuracy', 'F1 Score'])
            writer.writerow([benchmark, feature, classifier, results['accuracy'], results['balanced_accuracy'], results['f1']])


def classify(config: ClassificationConfig) -> Dict[str, Any]:
    # Create dataset
    X, y = create_dataset(config)
    X = np.array(X)
    y = np.array(y)
    
    # Get classifier and parameter grid
    classifier = construct_classifier(config)
    
    # Get predictions for all folds
    y_pred = cross_val_predict(classifier, X, y, cv=10)
    
    # Calculate all metrics from the same predictions
    accuracy = accuracy_score(y, y_pred)
    balanced_accuracy = balanced_accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
    # Print results
    print("\nCross-validation results:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.3f}")
    print(f"F1 Score: {f1:.3f}")
    
    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'f1': f1
    }


def classification_partial(config: ClassificationConfig) -> Dict[str, Any]:

    X, y = create_dataset(config)
    X = np.array(X)
    y = np.array(y)

    X_len = len(X)

    train_X = []
    train_Y = []
    test_X = []
    test_Y = []

    for _, p in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):

        train_X.append(X[0:int(X_len*p)])
        train_Y.append(y[0:int(X_len*p)])
        test_X.append(X[int(X_len*p):])
        test_Y.append(y[int(X_len * p):])

    classifier = construct_classifier(config)
    
    results = []
    for p in range(len(train_X)):

        try:
            classifier.fit(train_X[p],train_Y[p])
            y_pred = classifier.predict(test_X[p])            
        except:
            y_pred = [0] * len(test_X[p])
            
        acc = accuracy_score(test_Y[p],y_pred)
        balanced_acc = balanced_accuracy_score(test_Y[p],y_pred)
        f1 = f1_score(test_Y[p],y_pred)

        results.append([acc, balanced_acc, f1])

    return results  
        

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    print("Starting classification...")

    for benchmark in valid_benchmarks:
        print(f"Benchmark: {benchmark}")
        for feature in valid_features:
            print(f"Feature: {feature}")
            for classifier in valid_classifiers:
                print(f"Classifier: {classifier}")

                config = ClassificationConfig(benchmark=benchmark, features=feature, classifier=classifier)

                # Get results
                print("Classifying...")
                results = classify(config)
                print("Partial classification...")
                results_partial = classification_partial(config)
                
                # Create filenames
                total_filename = f'results/{benchmark}_{feature}_{classifier}_total.csv'
                partial_filename = f'results/{benchmark}_{feature}_{classifier}_partial.csv'
                
                # Write results to separate files
                write_results(results, total_filename, is_partial=False, benchmark=benchmark, feature=feature, classifier=classifier)
                write_results(results_partial, partial_filename, is_partial=True, benchmark=benchmark, feature=feature, classifier=classifier)
                
                print(f"Results written to {total_filename} and {partial_filename}\n\n")
                





