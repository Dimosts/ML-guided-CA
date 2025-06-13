import pandas as pd
import glob
import os
import numpy as np
import argparse
from experiments.utils.experiment_utils import BENCHMARKS, ALGORITHMS

def parse_filename(filename):
    """Parse filename in format benchmark_algorithm.csv"""
    
    # Remove .csv extension
    name = filename.replace('.csv', '')
    
    # Known valid values for each component
    valid_benchmarks = BENCHMARKS
    valid_algorithms = ALGORITHMS
    
    # Try to match each component
    for benchmark in valid_benchmarks:
        if name.startswith(benchmark + '_'):
            algorithm = name[len(benchmark)+1:]
            if algorithm in valid_algorithms:
                return benchmark, algorithm
    
    return None, None

def process_baseline_results(results_folder):
    # Create output directory if it doesn't exist
    output_dir = "experiments/aggregated_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the folder name to use as output prefix
    normalized_path = results_folder.rstrip('/')  # Remove trailing slashes first
    output_prefix = os.path.basename(normalized_path) + "_"
    
    # Get all benchmark subdirectories in the results directory
    benchmark_dirs = [d for d in glob.glob(f"{results_folder}/*") if os.path.isdir(d)]
    
    if not benchmark_dirs:
        print(f"No benchmark directories found in {results_folder}/")
        return
    
    # Dictionary to store aggregated results
    aggregated_means = {}
    aggregated_stderr = {}
    algorithm_info = []
    benchmark_info = []
    rows_used_info = []
    
    for benchmark_dir in benchmark_dirs:
        # Get all CSV files in the benchmark directory
        csv_files = glob.glob(os.path.join(benchmark_dir, "*.csv"))
        benchmark = os.path.basename(benchmark_dir)
        
        for file in csv_files:
            try:
                # Read the CSV file
                df = pd.read_csv(file)
                
                # Take only the last 8 rows if there are more than 8 rows
                original_row_count = len(df)
                if len(df) > 8:
                    df = df.tail(8)
                    rows_used_msg = f"Last 8 of {original_row_count}"
                else:
                    rows_used_msg = f"All {original_row_count}"
                
                # Calculate means for all numeric columns
                means = df.mean(numeric_only=True)
                
                # Calculate standard errors for all numeric columns
                stderr = df.std(numeric_only=True) / np.sqrt(len(df))
                
                # Parse filename to get components
                filename = os.path.basename(file)
                benchmark, algorithm = parse_filename(filename)
                
                if None in (benchmark, algorithm):
                    print(f"Warning: Could not parse filename {filename}. Skipping.")
                    continue
                
                # Store results
                aggregated_means[filename] = means
                aggregated_stderr[filename] = stderr
                algorithm_info.append(algorithm)
                benchmark_info.append(benchmark)
                rows_used_info.append(rows_used_msg)
                
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
    
    # Create DataFrames from the aggregated results
    means_df = pd.DataFrame.from_dict(aggregated_means, orient='index')
    stderr_df = pd.DataFrame.from_dict(aggregated_stderr, orient='index')
    
    # Add metadata columns to both DataFrames
    for i, df in enumerate([means_df, stderr_df]):
        df['Algorithm'] = algorithm_info
        df['Benchmark'] = benchmark_info
        df['Rows Used'] = rows_used_info
    
        # Reorder columns to put metadata first
        cols = ['Algorithm', 'Benchmark', 'Rows Used'] + \
               [col for col in df.columns if col not in ['Algorithm', 'Benchmark', 'Rows Used']]
        if i == 0:
            means_df = df[cols]
        else:
            stderr_df = df[cols]
    
    # Save to CSV files
    means_output_file = os.path.join(output_dir, f"{output_prefix}averaged_results.csv")
    stderr_output_file = os.path.join(output_dir, f"{output_prefix}stderr_results.csv")
    
    means_df.to_csv(means_output_file)
    stderr_df.to_csv(stderr_output_file)
    
    print(f"Processed {results_folder}/ directory")
    print(f"Average results have been saved to {means_output_file}")
    print(f"Standard error results have been saved to {stderr_output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process baseline experiment results from CSV files.')
    parser.add_argument('folder', type=str, 
                        help='Path to the folder containing baseline results', default='baseline_results')
    args = parser.parse_args()
    
    process_baseline_results(args.folder) 