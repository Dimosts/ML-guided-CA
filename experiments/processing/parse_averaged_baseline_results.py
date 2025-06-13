import pandas as pd
import json
from typing import Dict, Any
import os

def parse_averaged_baseline_results(csv_file: str) -> Dict[str, Any]:
    """
    Parse the averaged baseline results CSV file into a nested dictionary structure.
    Structure: {algorithm: {benchmark: metrics}}
    
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
                     ['Algorithm', 'Benchmark', 'Rows Used']]
    
    # Build the nested dictionary
    for algorithm in algorithms:
        results[algorithm] = {}
        algorithm_df = df[df['Algorithm'] == algorithm]
        
        for benchmark in benchmarks:
            benchmark_df = algorithm_df[algorithm_df['Benchmark'] == benchmark]
            
            if not benchmark_df.empty:
                # Get the metrics for this algorithm-benchmark combination
                metrics = benchmark_df[metric_columns].iloc[0].to_dict()
                results[algorithm][benchmark] = metrics
    
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
    
    parser = argparse.ArgumentParser(description='Parse averaged baseline results into a nested dictionary structure')
    parser.add_argument('--input', type=str, default='experiments/aggregated_results/baseline_results_averaged_results.csv',
                      help='Path to the averaged results CSV file (default: experiments/aggregated_results/baseline_results_averaged_results.csv)')
    parser.add_argument('--output', type=str, default='experiments/parsed_results/parsed_baseline_results.json',
                      help='Path to save the JSON output (default: experiments/parsed_results/parsed_baseline_results.json)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Parse the results
    results = parse_averaged_baseline_results(args.input)
    
    # Save to JSON
    save_results_to_json(results, args.output)
    
    print(f"Results have been parsed and saved to {args.output}")
    
    # Print some statistics
    print("\nStatistics:")
    print(f"Number of algorithms: {len(results)}")
    for algorithm, benchmarks in results.items():
        print(f"\nAlgorithm: {algorithm}")
        print(f"Number of benchmarks: {len(benchmarks)}")
        for benchmark, metrics in benchmarks.items():
            print(f"  Benchmark: {benchmark}")
            print(f"  Number of metrics: {len(metrics)}")

if __name__ == "__main__":
    main() 