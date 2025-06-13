import pandas as pd
import os
import glob
import json
from typing import Dict, List

def load_all_results() -> Dict[str, pd.DataFrame]:
    """
    Load all result files from the results directory.
    Returns a dictionary with 'total' and 'partial' DataFrames.
    """
    # Get all CSV files
    total_files = glob.glob('results/*_total.csv')
    partial_files = glob.glob('results/*_partial.csv')
    
    # Load total results
    total_dfs = []
    for file in total_files:
        df = pd.read_csv(file)
        total_dfs.append(df)
    total_df = pd.concat(total_dfs, ignore_index=True)
    
    # Load partial results
    partial_dfs = []
    for file in partial_files:
        df = pd.read_csv(file)
        partial_dfs.append(df)
    partial_df = pd.concat(partial_dfs, ignore_index=True)
    
    return {
        'total': total_df,
        'partial': partial_df
    }

def aggregate_total_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate total results by classifier and feature, averaging across benchmarks.
    """
    # First group by benchmark, classifier, and feature to get per-benchmark results
    benchmark_results = df.groupby(['Benchmark', 'Classifier', 'Feature']).agg({
        'Accuracy': 'mean',
        'Balanced Accuracy': 'mean',
        'F1 Score': 'mean'
    }).reset_index()
    
    # Then group by classifier and feature to get averages across benchmarks
    agg_df = benchmark_results.groupby(['Classifier', 'Feature']).agg({
        'Accuracy': 'mean',
        'Balanced Accuracy': 'mean',
        'F1 Score': 'mean'
    }).reset_index()
    
    # Calculate standard deviation across benchmarks
    std_df = benchmark_results.groupby(['Classifier', 'Feature']).agg({
        'Accuracy': 'std',
        'Balanced Accuracy': 'std',
        'F1 Score': 'std'
    }).reset_index()
    
    # Rename columns to indicate they are standard deviations
    std_df.columns = ['Classifier', 'Feature', 'Accuracy_std', 'Balanced_Accuracy_std', 'F1_Score_std']
    
    # Merge means and standard deviations
    final_df = pd.merge(agg_df, std_df, on=['Classifier', 'Feature'])
    
    return final_df

def aggregate_partial_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate partial results by classifier, feature, and percentage.
    """
    # First group by benchmark, classifier, feature, and percentage
    benchmark_results = df.groupby(['Benchmark', 'Classifier', 'Feature', 'Percentage']).agg({
        'Accuracy': 'mean',
        'Balanced Accuracy': 'mean',
        'F1 Score': 'mean'
    }).reset_index()
    
    # Then group by classifier, feature, and percentage to get averages across benchmarks
    agg_df = benchmark_results.groupby(['Classifier', 'Feature', 'Percentage']).agg({
        'Accuracy': 'mean',
        'Balanced Accuracy': 'mean',
        'F1 Score': 'mean'
    }).reset_index()
    
    return agg_df

def save_aggregated_results(total_df: pd.DataFrame, partial_df: pd.DataFrame):
    """
    Save aggregated results to JSON files in experiments/parsed_results directory.
    """
    os.makedirs('experiments/parsed_results', exist_ok=True)
    
    # Convert DataFrames to dictionaries and save as JSON
    total_dict = total_df.to_dict(orient='records')
    partial_dict = partial_df.to_dict(orient='records')
    
    with open('experiments/parsed_results/parsed_classification_total_results.json', 'w') as f:
        json.dump(total_dict, f, indent=4)
    with open('experiments/parsed_results/parsed_classification_partial_results.json', 'w') as f:
        json.dump(partial_dict, f, indent=4)

def main():
    print("Loading results...")
    results = load_all_results()
    
    print("Aggregating total results...")
    total_agg = aggregate_total_results(results['total'])
    
    print("Aggregating partial results...")
    partial_agg = aggregate_partial_results(results['partial'])
    
    print("Saving aggregated results...")
    save_aggregated_results(total_agg, partial_agg)
    
    print("Done! Results saved in 'experiments/parsed_results' directory.")

if __name__ == "__main__":
    main() 