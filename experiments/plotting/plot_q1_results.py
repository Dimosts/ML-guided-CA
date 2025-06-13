import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from utils import ml_dict, feature_dict, classifier_colors, colors, linestyles, markers
from matplotlib.ticker import FormatStrFormatter

plt.rcParams.update({
    'font.size': 25,
    'axes.titlesize': 30,
    'axes.labelsize': 25,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'figure.titlesize': 30
})

def load_results():
    """
    Load results from JSON files.
    """
    with open('experiments/parsed_results/parsed_classification_total_results.json', 'r') as f:
        total_df = pd.DataFrame(json.load(f))
    with open('experiments/parsed_results/parsed_classification_partial_results.json', 'r') as f:
        partial_df = pd.DataFrame(json.load(f))
    return total_df, partial_df

def create_plots(total_df: pd.DataFrame, partial_df: pd.DataFrame):
    """
    Create plots for the aggregated results.
    """
    os.makedirs('experiments/plots/q1', exist_ok=True)
    
    # Set style and font sizes
    plt.style.use('bmh')  # Using a clean, modern style
    plt.rcParams.update({
        'font.size': 25,
        'axes.labelsize': 25,
        'axes.titlesize': 30,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20,
        'legend.title_fontsize': 20
    })
    
    metrics = ['Accuracy', 'Balanced Accuracy', 'F1 Score']
    feature_order = ['rel_dim_block', 'rel_dim', 'simple_rel']
    # Only keep features present in the data
    features_in_data = [f for f in feature_order if f in total_df['Feature'].unique()]
    
    # Create and save legends separately
    # For total results (features)
    plt.figure(figsize=(2, 4))  # Back to vertical layout
    handles = []
    labels = []
    for i, feature in enumerate(features_in_data):
        patch = Patch(facecolor=colors[i % len(colors)], edgecolor='black', label=feature_dict[feature])
        handles.append(patch)
        labels.append(feature_dict[feature])
    plt.legend(handles, labels, title='Features', loc='center')
    plt.axis('off')
    plt.savefig('experiments/plots/q1/legend_features.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # For partial results (classifiers)
    plt.figure(figsize=(2, 6))  # Back to vertical layout
    handles = []
    labels = []
    for j, classifier in enumerate(partial_df['Classifier'].unique()):
        handle = Line2D(
            [0], [0],
            color=classifier_colors[classifier],
            linestyle=linestyles[j % len(linestyles)],
            marker=markers[j % len(markers)],
            linewidth=2,
            markersize=8,
            label=ml_dict[classifier]
        )
        handles.append(handle)
        labels.append(ml_dict[classifier])
    plt.legend(handles, labels, title='Classifiers', loc='center')
    plt.axis('off')
    plt.savefig('experiments/plots/q1/legend_classifiers.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # 1. Bar plots for total results - one figure per metric (no legend)
    for metric in metrics:
        plt.figure(figsize=(5, 4))
        # Create a copy of the dataframe with mapped names
        plot_df = total_df.copy()
        plot_df['Classifier'] = plot_df['Classifier'].map(ml_dict)
        plot_df['Feature'] = plot_df['Feature'].map(feature_dict)
        
        ax = sns.barplot(data=plot_df, x='Classifier', y=metric, hue='Feature', 
                        palette=colors[:len(features_in_data)], 
                        hue_order=[feature_dict[f] for f in features_in_data])
        
        # Add error bars for standard deviation
        for j, feature in enumerate(features_in_data):
            feature_data = plot_df[plot_df['Feature'] == feature_dict[feature]]
            std_col = f'{metric}_std'
            if std_col in feature_data.columns:
                plt.errorbar(x=range(len(feature_data)), 
                           y=feature_data[metric],
                           yerr=feature_data[std_col],
                           fmt='none', color='black', capsize=5)
        
        plt.xticks(rotation=0)  # Make labels horizontal
        plt.xlabel('')
        plt.ylabel(metric, fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Format y-axis to show 2 decimal places
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        
        # Remove legend
        ax.get_legend().remove()
        
        plt.tight_layout()
        plt.savefig(f'experiments/plots/q1/total_{metric.lower().replace(" ", "_")}.png', bbox_inches='tight', dpi=300)
        plt.close()
    
    # 2. Line plots for partial results - one figure per feature and metric (no legend)
    for i, feature in enumerate(features_in_data):
        feature_data = partial_df[partial_df['Feature'] == feature].copy()
        feature_data['Classifier'] = feature_data['Classifier'].map(ml_dict)
        
        for metric in metrics:
            plt.figure(figsize=(5, 4))
            
            # Reapply font settings for each plot
            plt.rcParams.update({
                'font.size': 20,
                'axes.labelsize': 20,
                'axes.titlesize': 20,
                'xtick.labelsize': 16,
                'ytick.labelsize': 16,
                'legend.fontsize': 20,
                'legend.title_fontsize': 20
            })
            
            for j, classifier in enumerate(feature_data['Classifier'].unique()):
                classifier_data = feature_data[feature_data['Classifier'] == classifier]
                original_classifier = next(k for k, v in ml_dict.items() if v == classifier)
                plt.plot(classifier_data['Percentage'], classifier_data[metric], 
                        color=classifier_colors[original_classifier],
                        linestyle=linestyles[j % len(linestyles)],
                        marker=markers[j % len(markers)],
                        linewidth=2,
                        markersize=8)
            
            plt.xlabel('Training Data Percentage', fontsize=20)
            plt.ylabel(metric, fontsize=20)
            plt.grid(True, alpha=0.3)
            
            # Set y-axis based on actual data values
            min_val = feature_data[metric].min()
            max_val = feature_data[metric].max()
            margin = (max_val - min_val) * 0.1  # 10% margin
            plt.ylim(max(0, min_val - margin), min(1, max_val + margin))
            
            # Format y-axis to show 2 decimal places
            plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            
            # Explicitly set tick label sizes
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            
            plt.tight_layout()
            plt.savefig(f'experiments/plots/q1/partial_{feature_dict[feature]}_{metric.lower().replace(" ", "_")}.png', 
                       bbox_inches='tight', dpi=300)
            plt.close()

def main():
    print("Loading results from JSON...")
    total_df, partial_df = load_results()
    
    print("Creating plots...")
    create_plots(total_df, partial_df)
    
    print("Done! Plots saved in 'experiments/plots/q1' directory.")

if __name__ == "__main__":
    main() 