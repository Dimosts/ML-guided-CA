import pandas as pd
import os
from typing import Dict, List
import glob
import re
from experiments.utils.experiment_utils import BENCHMARKS, FEATURES, CLASSIFIERS

def load_tuning_results() -> Dict[str, List[pd.DataFrame]]:
    """
    Load all tuning result CSV files and group them by feature and classifier.
    Returns a dictionary with keys in format 'feature_classifier' and values as list of DataFrames.
    """
    results_dir = 'tuning_results'
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Directory {results_dir} not found")

    # Get all CSV files
    csv_files = glob.glob(os.path.join(results_dir, 'tuning_results_*.csv'))
    
    # Group files by feature and classifier
    grouped_results = {}
    for file in csv_files:
        # Extract feature and classifier from filename
        # Format: tuning_results_feature_classifier_benchmark.csv
        filename = os.path.basename(file).replace('tuning_results_', '').replace('.csv', '')
        
        # Try to match against known features and classifiers
        found_match = False
        for feature in FEATURES:
            for classifier in CLASSIFIERS:
                # Check if filename starts with this feature and classifier combination
                prefix = f"{feature}_{classifier}_"
                if filename.startswith(prefix):
                    benchmark = filename[len(prefix):]
                    if benchmark in BENCHMARKS:
                        key = f"{feature}_{classifier}"
                        if key not in grouped_results:
                            grouped_results[key] = []
                        df = pd.read_csv(file)
                        grouped_results[key].append(df)
                        found_match = True
                        break
            if found_match:
                break
        
        if not found_match:
            print(f"Warning: Could not match filename {filename} to known features/classifiers/benchmarks, skipping...")
    
    return grouped_results

def aggregate_results(grouped_results: Dict[str, List[pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
    """
    Aggregate results for each feature-classifier combination across all benchmarks.
    """
    aggregated = {}
    
    for key, dfs in grouped_results.items():
        # Combine all DataFrames
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Group by all parameter columns
        param_cols = [col for col in combined_df.columns if col not in [
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
        ]]
        
        # Calculate mean of all metrics for each parameter combination
        aggregated_df = combined_df.groupby(param_cols).agg({
            'mean_test_balanced_accuracy': 'mean',
            'std_test_balanced_accuracy': 'mean',
            'mean_test_f1': 'mean',
            'std_test_f1': 'mean',
            'mean_test_f1_weighted': 'mean',
            'std_test_f1_weighted': 'mean',
            'mean_test_accuracy': 'mean',
            'std_test_accuracy': 'mean',
            'mean_fit_time': 'max',
            'std_fit_time': 'max'
        }).reset_index()
        
        # Sort by mean balanced accuracy
        aggregated_df = aggregated_df.sort_values('mean_test_balanced_accuracy', ascending=False)
        
        aggregated[key] = aggregated_df
    
    return aggregated

def save_aggregated_results(aggregated_results: Dict[str, pd.DataFrame]):
    """
    Save aggregated results to CSV files.
    """
    output_dir = 'experiments/aggregated_tuning_results'
    os.makedirs(output_dir, exist_ok=True)
    
    for key, df in aggregated_results.items():
        output_file = os.path.join(output_dir, f'aggregated_results_{key}.csv')
        df.to_csv(output_file, index=False)
        print(f"Saved aggregated results to {output_file}")

def main():
    try:
        # Load all tuning results
        print("Loading tuning results...")
        grouped_results = load_tuning_results()
        
        # Aggregate results
        print("Aggregating results...")
        aggregated_results = aggregate_results(grouped_results)
        
        # Save results
        print("Saving aggregated results...")
        save_aggregated_results(aggregated_results)
        
        print("Processing complete!")
        
    except Exception as e:
        print(f"Error processing results: {str(e)}")

if __name__ == "__main__":
    main() 