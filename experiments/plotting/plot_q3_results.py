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
os.makedirs('experiments/plots/q3', exist_ok=True)

# Create a separate legend figure
plt.figure(figsize=(12, 1))
handles = []
labels = []
# Add classifier bars to legend
for classifier, classifier_name in classifiers.items():
    handles.append(plt.Rectangle((0, 0), 1, 1, color=classifier_colors[classifier], label=classifier_name))
    labels.append(classifier_name)
plt.legend(handles, labels, loc='center', ncol=len(classifiers))
plt.axis('off')
plt.savefig('experiments/plots/q3/legend.png', bbox_inches='tight', dpi=300)
plt.close()

# Dictionary to store percentages for averaging
percentages = {feature: {classifier: {'proba': [], 'class': []} for classifier in classifiers} 
              for feature in ['simple_rel', 'rel_dim', 'rel_dim_block']}

# For each benchmark
for benchmark in baseline_data['growacq'].keys():
    baseline_value = baseline_data['growacq'][benchmark]['tot_q']
    
    # For each feature subtype
    for feature in ['simple_rel', 'rel_dim', 'rel_dim_block']:
        plt.figure(figsize=(14, 4))
        
        n_classifiers = len(classifiers)
        bar_width = 0.8 / n_classifiers
        x = np.arange(2)  # Two positions for proba and class

        # First pass to get max value for y-axis limit
        max_val = 0
        for classifier in classifiers:
            # Get proba value
            if ('obj_proba2' in classifier_data['growacq'][benchmark] and 
                feature in classifier_data['growacq'][benchmark]['obj_proba2'] and 
                classifier in classifier_data['growacq'][benchmark]['obj_proba2'][feature]):
                val = classifier_data['growacq'][benchmark]['obj_proba2'][feature][classifier]['tot_q']
                max_val = max(max_val, val)
            
            # Get class value
            if ('obj_class' in classifier_data['growacq'][benchmark] and 
                feature in classifier_data['growacq'][benchmark]['obj_class'] and 
                classifier in classifier_data['growacq'][benchmark]['obj_class'][feature]):
                val = classifier_data['growacq'][benchmark]['obj_class'][feature][classifier]['tot_q']
                max_val = max(max_val, val)

        # Plot results for each classifier
        for i, (classifier, classifier_name) in enumerate(classifiers.items()):
            values = []
            # Get proba value
            proba_val = 0
            if ('obj_proba2' in classifier_data['growacq'][benchmark] and 
                feature in classifier_data['growacq'][benchmark]['obj_proba2'] and 
                classifier in classifier_data['growacq'][benchmark]['obj_proba2'][feature]):
                proba_val = classifier_data['growacq'][benchmark]['obj_proba2'][feature][classifier]['tot_q']
                # Calculate and store percentage for later use
                if baseline_value > 0:
                    percentages[feature][classifier]['proba'].append((proba_val / baseline_value) * 100)
            
            # Get class value
            class_val = 0
            if ('obj_class' in classifier_data['growacq'][benchmark] and 
                feature in classifier_data['growacq'][benchmark]['obj_class'] and 
                classifier in classifier_data['growacq'][benchmark]['obj_class'][feature]):
                class_val = classifier_data['growacq'][benchmark]['obj_class'][feature][classifier]['tot_q']
                # Calculate and store percentage for later use
                if baseline_value > 0:
                    percentages[feature][classifier]['class'].append((class_val / baseline_value) * 100)
            
            bars = plt.bar(x + i * bar_width - 0.4 + bar_width/2, [proba_val, class_val], bar_width, 
                    label=classifier_name, color=classifier_colors[classifier], edgecolor='black', linewidth=1.5)
            
            # Add numbers above bars
            for bar in bars:
                height = bar.get_height()
                if height > 0:  # Only add text if there's a value
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height)}',
                            ha='center', va='bottom', fontsize=16)

        plt.xticks(x, ['proba', 'class'], rotation=0)
        plt.ylabel('# of Queries')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)

        # Set y-axis limit with padding for the numbers
        plt.ylim(0, max_val * 1.15)  # Add 15% padding at the top

        #plt.title(f'{benchmark.replace("_", " ").title()} - {feature_dict[feature]}')
        plt.tight_layout()
        plt.savefig(f'experiments/plots/q3/{benchmark}_{feature}_queries.png', bbox_inches='tight', dpi=300)
        plt.close()

        # Add percentage plot for this benchmark and feature
        plt.figure(figsize=(14, 4))
        
        n_classifiers = len(classifiers)
        bar_width = 0.8 / n_classifiers
        x = np.arange(2)  # Two positions for proba and class

        # First pass to get max value for y-axis limit
        max_val = 0
        for classifier in classifiers:
            # Get proba value
            if ('obj_proba2' in classifier_data['growacq'][benchmark] and 
                feature in classifier_data['growacq'][benchmark]['obj_proba2'] and 
                classifier in classifier_data['growacq'][benchmark]['obj_proba2'][feature]):
                val = (classifier_data['growacq'][benchmark]['obj_proba2'][feature][classifier]['tot_q'] / baseline_value) * 100
                max_val = max(max_val, val)
            
            # Get class value
            if ('obj_class' in classifier_data['growacq'][benchmark] and 
                feature in classifier_data['growacq'][benchmark]['obj_class'] and 
                classifier in classifier_data['growacq'][benchmark]['obj_class'][feature]):
                val = (classifier_data['growacq'][benchmark]['obj_class'][feature][classifier]['tot_q'] / baseline_value) * 100
                max_val = max(max_val, val)

        # Plot percentage results for each classifier
        for i, (classifier, classifier_name) in enumerate(classifiers.items()):
            # Get proba value
            proba_val = 0
            if ('obj_proba2' in classifier_data['growacq'][benchmark] and 
                feature in classifier_data['growacq'][benchmark]['obj_proba2'] and 
                classifier in classifier_data['growacq'][benchmark]['obj_proba2'][feature]):
                proba_val = classifier_data['growacq'][benchmark]['obj_proba2'][feature][classifier]['tot_q']
                # Store for averaging
                if baseline_value > 0:
                    percentages[feature][classifier]['proba'].append((proba_val / baseline_value) * 100)
            
            # Get class value
            class_val = 0
            if ('obj_class' in classifier_data['growacq'][benchmark] and 
                feature in classifier_data['growacq'][benchmark]['obj_class'] and 
                classifier in classifier_data['growacq'][benchmark]['obj_class'][feature]):
                class_val = classifier_data['growacq'][benchmark]['obj_class'][feature][classifier]['tot_q']
                # Store for averaging
                if baseline_value > 0:
                    percentages[feature][classifier]['class'].append((class_val / baseline_value) * 100)
            
            # Calculate percentages for display
            proba_percentage = (proba_val / baseline_value) * 100 if baseline_value > 0 else 0
            class_percentage = (class_val / baseline_value) * 100 if baseline_value > 0 else 0
            
            bars = plt.bar(x + i * bar_width - 0.4 + bar_width/2, [proba_percentage, class_percentage], bar_width, 
                    label=classifier_name, color=classifier_colors[classifier], edgecolor='black', linewidth=1.5)
            
            # Add numbers above bars
            for bar in bars:
                height = bar.get_height()
                if height > 0:  # Only add text if there's a value
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.0f}%',
                            ha='center', va='bottom', fontsize=16)

        plt.xticks(x, ['proba', 'class'], rotation=0)
        plt.ylabel('% of Baseline')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)

        # Set y-axis limit with padding for the numbers
        plt.ylim(0, max_val * 1.15)  # Add 15% padding at the top

        plt.tight_layout()
        plt.savefig(f'experiments/plots/q3/{benchmark}_{feature}_queries_percentage.png', bbox_inches='tight', dpi=300)
        plt.close()

# Create average figure for each feature
for feature in ['simple_rel', 'rel_dim', 'rel_dim_block']:
    plt.figure(figsize=(14, 4))
    
    n_classifiers = len(classifiers)
    bar_width = 0.8 / n_classifiers
    x = np.arange(2)  # Two positions for proba and class

    # First pass to get max value for y-axis limit
    max_val = 0
    for classifier in classifiers:
        avg_proba = np.mean(percentages[feature][classifier]['proba']) if percentages[feature][classifier]['proba'] else 0
        avg_class = np.mean(percentages[feature][classifier]['class']) if percentages[feature][classifier]['class'] else 0
        max_val = max(max_val, avg_proba, avg_class)

    # Plot average results for each classifier
    for i, (classifier, classifier_name) in enumerate(classifiers.items()):
        # Calculate averages
        avg_proba = np.mean(percentages[feature][classifier]['proba']) if percentages[feature][classifier]['proba'] else 0
        avg_class = np.mean(percentages[feature][classifier]['class']) if percentages[feature][classifier]['class'] else 0
        
        bars = plt.bar(x + i * bar_width - 0.4 + bar_width/2, [avg_proba, avg_class], bar_width, 
                label=classifier_name, color=classifier_colors[classifier], edgecolor='black', linewidth=1.5)
        
        # Add numbers above bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only add text if there's a value
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.0f}%',
                        ha='center', va='bottom', fontsize=16)

    plt.xticks(x, ['proba', 'class'], rotation=0)
    plt.ylabel('# of Queries')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Set y-axis limit with padding for the numbers
    plt.ylim(0, max_val * 1.15)  # Add 15% padding at the top

    #plt.title(f'Average Across All Benchmarks - {feature_dict[feature]}')
    plt.tight_layout()
    plt.savefig(f'experiments/plots/q3/average_{feature}_queries.png', bbox_inches='tight', dpi=300)
    plt.close() 