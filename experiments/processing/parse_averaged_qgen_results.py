import pandas as pd
import json
from typing import Dict, Any
import os

def parse_averaged_classifier_results(csv_file: str) -> Dict[str, Any]:
    """
    Parse the averaged results CSV file into a nested dictionary structure.
    Structure: {algorithm: {benchmark: {objective: {features: {classifier: metrics}}}}}
    
    Args:
        csv_file: Path to the averaged results CSV file
        
    Returns:
        Nested dictionary containing the parsed results
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Initialize the nested dictionary
    results = {}
    
    # Get all unique values for each category
    algorithms = df['Algorithm'].unique()
    benchmarks = df['Benchmark'].unique()
    
    # Get the metric columns (all columns except the category columns)
    metric_columns = [col for col in df.columns if col not in 
                     ['Algorithm', 'Benchmark', 'Objective', 'Features', 'Classifier', 'Rows Used']]
    
    # Build the nested dictionary
    for algorithm in algorithms:
        results[algorithm] = {}
        algorithm_df = df[df['Algorithm'] == algorithm]
        
        for benchmark in benchmarks:
            results[algorithm][benchmark] = {}
            benchmark_df = algorithm_df[algorithm_df['Benchmark'] == benchmark]
            
            # Get valid objective-feature combinations for this benchmark
            valid_combinations = benchmark_df[['Objective', 'Features']].drop_duplicates()
            
            for _, row in valid_combinations.iterrows():
                objective = row['Objective']
                feature = row['Features']
                
                if objective not in results[algorithm][benchmark]:
                    results[algorithm][benchmark][objective] = {}
                
                results[algorithm][benchmark][objective][feature] = {}
                
                # Get data for this specific objective-feature combination
                combination_df = benchmark_df[
                    (benchmark_df['Objective'] == objective) & 
                    (benchmark_df['Features'] == feature)
                ]
                
                # Get unique classifiers for this combination
                classifiers = combination_df['Classifier'].unique()
                
                for classifier in classifiers:
                    classifier_df = combination_df[combination_df['Classifier'] == classifier]
                    
                    if not classifier_df.empty:
                        # Get the metrics for this combination
                        metrics = classifier_df[metric_columns].iloc[0].to_dict()
                        results[algorithm][benchmark][objective][feature][classifier] = metrics
    
    return results

def save_results_to_json(results: Dict[str, Any], output_file: str):
    """
    Save the parsed results to a JSON file.
    
    Args:
        results: The nested dictionary of results
        output_file: Path to save the JSON file
    """
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

def main():
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Parse averaged results into a nested dictionary structure')
    parser.add_argument('--input', type=str, default='experiments/aggregated_results/results_qgen_averaged_results.csv',
                      help='Path to the averaged results CSV file (default: experiments/aggregated_results/results_qgen_averaged_results.csv)')
    parser.add_argument('--output', type=str, default='experiments/parsed_results/parsed_qgen_results.json',
                      help='Path to save the JSON output (default: experiments/parsed_results/parsed_qgen_results.json)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Parse the results
    results = parse_averaged_classifier_results(args.input)
    
    # Save to JSON
    save_results_to_json(results, args.output)
    
    print(f"Results have been parsed and saved to {args.output}")
    
    # Print some statistics
    print("\nStatistics:")
    print(f"Number of algorithms: {len(results)}")
    for algorithm, benchmarks in results.items():
        print(f"\nAlgorithm: {algorithm}")
        print(f"Number of benchmarks: {len(benchmarks)}")
        for benchmark, objectives in benchmarks.items():
            print(f"  Benchmark: {benchmark}")
            print(f"  Number of objectives: {len(objectives)}")
            for objective, features in objectives.items():
                print(f"    Objective: {objective}")
                print(f"    Number of features: {len(features)}")
                for feature, classifiers in features.items():
                    print(f"      Features: {feature}")
                    print(f"      Number of classifiers: {len(classifiers)}")

if __name__ == "__main__":
    main() 