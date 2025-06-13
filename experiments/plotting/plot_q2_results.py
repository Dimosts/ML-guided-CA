import json
import matplotlib.pyplot as plt
import numpy as np
import os
from utils import ml_dict, feature_dict, colors

# Set font sizes
plt.rcParams.update({
    'font.size': 25,
    'axes.titlesize': 30,
    'axes.labelsize': 23,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 16,
    'figure.titlesize': 30
})

# Load the data
with open('experiments/parsed_results/parsed_baseline_results.json', 'r') as f:
    baseline_data = json.load(f)

with open('experiments/parsed_results/parsed_qgen_results.json', 'r') as f:
    classifier_data = json.load(f)

# Define classifiers and their display names using ml_dict
classifiers = {
    'logistic_regression': ml_dict['logistic_regression'],
    'random_forest': ml_dict['random_forest'],
    'naive_bayes': ml_dict['naive_bayes'],
    'decision_tree': ml_dict['decision_tree'],
    'mlp': ml_dict['mlp'],
    'svm': ml_dict['svm']
}

# Assign colors from utils.py
classifier_colors = {
    'logistic_regression': colors[0],
    'random_forest': colors[1],
    'naive_bayes': colors[2],
    'decision_tree': colors[3],
    'mlp': colors[4],
    'svm': colors[5]
}

# Create output directory if it doesn't exist
os.makedirs('experiments/plots/q2', exist_ok=True)

# Create a separate legend figure
plt.figure(figsize=(12, 1))
handles = []
labels = []
# Add baseline line to legend
handles.append(plt.Line2D([0], [0], color='black', linestyle='--', linewidth=4, label='Not guided'))
labels.append('Not guided')
# Add classifier bars to legend
for classifier, classifier_name in classifiers.items():
    handles.append(plt.Rectangle((0, 0), 1, 1, color=classifier_colors[classifier], label=classifier_name))
    labels.append(classifier_name)
plt.legend(handles, labels, loc='center', ncol=len(classifiers) + 1)
plt.axis('off')
plt.savefig('experiments/plots/q2/legend.png', bbox_inches='tight', dpi=300)
plt.close()

# Define fixed order for feature subtypes
feature_order = ['simple_rel', 'rel_dim', 'rel_dim_block']

# Dictionary to store all percentage values for averaging
all_percentages = {metric: {classifier: {subtype: [] for subtype in feature_order} 
                           for classifier in classifiers.keys()} 
                  for metric in ['tot_q', 'max_t']}

# For each benchmark
for benchmark in baseline_data['growacq'].keys():
    for metric, metric_name in [('tot_q', '# of Queries'), ('max_t', 'Time (s)')]:
        # Original value plot
        plt.figure(figsize=(15, 4))
        baseline_value = baseline_data['growacq'][benchmark][metric]
        plt.axhline(y=baseline_value, color='black', linestyle='--', linewidth=4, label='Not guided', alpha=1.0)

        # Only use subtypes under obj_proba2
        feature_family = 'obj_proba2'
        if feature_family not in classifier_data['growacq'][benchmark]:
            print(f"Warning: {feature_family} not found for {benchmark}")
            plt.close()
            continue
        feature_data = classifier_data['growacq'][benchmark][feature_family]
        
        # Print available feature subtypes for debugging
        print(f"Available feature subtypes for {benchmark}: {list(feature_data.keys())}")
        
        feature_subtypes = [subtype for subtype in feature_order if subtype in feature_data.keys()]

        n_subtypes = len(feature_subtypes)
        n_classifiers = len(classifiers)
        bar_width = 0.8 / n_classifiers
        x = np.arange(n_subtypes)

        # For each classifier, collect values for each feature subtype
        for i, (classifier, classifier_name) in enumerate(classifiers.items()):
            values = []
            for subtype in feature_subtypes:
                if classifier in feature_data[subtype]:
                    val = feature_data[subtype][classifier][metric]
                else:
                    val = 0
                values.append(val)
            plt.bar(x + i * bar_width - 0.4 + bar_width/2, values, bar_width, label=classifier_name, color=classifier_colors[classifier], edgecolor='black', linewidth=1.5)

        plt.xticks(x, [feature_dict[subtype] for subtype in feature_subtypes], rotation=0)
        plt.ylabel(metric_name)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(f'experiments/plots/q2/{benchmark}_{metric}_objproba2.png', bbox_inches='tight', dpi=300)
        plt.close()

        # Percentage plot
        plt.figure(figsize=(15, 4))
        plt.axhline(y=100, color='black', linestyle='--', linewidth=4, label='Baseline', alpha=1.0)

        # For each classifier, collect percentage values for each feature subtype
        for i, (classifier, classifier_name) in enumerate(classifiers.items()):
            percentage_values = []
            for subtype in feature_subtypes:
                if classifier in feature_data[subtype]:
                    val = feature_data[subtype][classifier][metric]
                    # Calculate percentage of baseline
                    percentage = (val / baseline_value) * 100
                    # Store for averaging
                    all_percentages[metric][classifier][subtype].append(percentage)
                else:
                    percentage = 0
                percentage_values.append(percentage)
            bars = plt.bar(x + i * bar_width - 0.4 + bar_width/2, percentage_values, bar_width, 
                   label=classifier_name, color=classifier_colors[classifier], 
                   edgecolor='black', linewidth=1.5)
            
            # Add value labels above bars
            for bar in bars:
                height = bar.get_height()
                # Round to whole numbers for both metrics
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.0f}%',
                        ha='center', va='bottom', fontsize=12)

        plt.xticks(x, [feature_dict[subtype] for subtype in feature_subtypes], rotation=0)
        plt.ylabel(f'% of Baseline')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(f'experiments/plots/q2/{benchmark}_{metric}_objproba2_percentage.png', bbox_inches='tight', dpi=300)
        plt.close()

# Create average percentage plots
for metric, metric_name in [('tot_q', '# of Queries'), ('max_t', 'Time (s)')]:
    plt.figure(figsize=(15, 4))
    plt.axhline(y=100, color='black', linestyle='--', linewidth=4, label='Baseline', alpha=1.0)

    n_subtypes = len(feature_order)
    n_classifiers = len(classifiers)
    bar_width = 0.8 / n_classifiers
    x = np.arange(n_subtypes)

    # For each classifier, plot average percentages
    for i, (classifier, classifier_name) in enumerate(classifiers.items()):
        avg_percentages = []
        for subtype in feature_order:
            values = all_percentages[metric][classifier][subtype]
            if values:  # Only calculate average if we have values
                avg = np.mean(values)
            else:
                avg = 0
            avg_percentages.append(avg)
        
        bars = plt.bar(x + i * bar_width - 0.4 + bar_width/2, avg_percentages, bar_width,
               label=classifier_name, color=classifier_colors[classifier],
               edgecolor='black', linewidth=1.5)
        
        # Add value labels above bars
        for bar in bars:
            height = bar.get_height()
            # Round to whole numbers for both metrics
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}%',
                    ha='center', va='bottom', fontsize=12)

    plt.xticks(x, [feature_dict[subtype] for subtype in feature_order], rotation=0)
    plt.ylabel(metric_name)
    #plt.title(f'Average {metric_name} Across All Benchmarks')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(f'experiments/plots/q2/average_{metric}_objproba2_percentage.png', bbox_inches='tight', dpi=300)
    plt.close() 