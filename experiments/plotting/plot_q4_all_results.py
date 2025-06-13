import json
import matplotlib.pyplot as plt
import numpy as np
import os
from utils import colors

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

# Benchmark name mapping
benchmark_names = {
    'sudoku': 'Sudoku',
    'gtsudoku': 'GTSudoku',
    'jsudoku': 'JSudoku',
    'random': 'Random',    
    'exam_timetable': 'ET',
    'nurse_rostering': 'NR',
    'jobshop': 'JS'
}

# Load the data
with open('experiments/parsed_results/parsed_baseline_results.json', 'r') as f:
    baseline_data = json.load(f)

with open('experiments/parsed_results/parsed_qgen_results.json', 'r') as f:
    classifier_data = json.load(f)

with open('experiments/parsed_results/parsed_guide_all_results.json', 'r') as f:
    classifier_guide_all_data = json.load(f)

with open('experiments/parsed_results/parsed_findc_results.json', 'r') as f:
    findc_data = json.load(f)

with open('experiments/parsed_results/parsed_findscope_results.json', 'r') as f:
    findscope_data = json.load(f)

# Use the order from benchmark_names
benchmarks = list(benchmark_names.keys())

# Output directory
output_dir = 'experiments/plots/q4/all'
os.makedirs(output_dir, exist_ok=True)

# Legend labels and colors
labels = ['Not guided (baseline)', 'Guide QGen', 'Guide FindC', 'FindScope-3 (not guided)', 'Guide FindScope-2', 'Guide FindScope-3', 'Guide All']
bar_colors = [
    colors[0],    # Baseline (dark burgundy)
    colors[1],    # Guided QGen (orange)
    colors[5],    # Guided FindC (blue)
    colors[6],    # FindScope-3 not guided (gray)
    colors[2],    # Guided FindScope-2 (light blue)
    colors[3],    # Guided FindScope-3 (green)
    colors[4],    # Guide All (purple)
]

# For each benchmark, plot the number of queries
for benchmark in benchmarks:
    plt.figure(figsize=(12, 6))
    
    # Get values for each approach
    values = []
    
    # 1. Baseline (Not guided)
    baseline_q = baseline_data['growacq'][benchmark]['tot_q']
    values.append(baseline_q)
    
    # 2. Guided QGen (proba)
    try:
        guided_qgen_q = classifier_data['growacq'][benchmark]['obj_proba2']['rel_dim_block']['random_forest']['tot_q']
    except KeyError:
        guided_qgen_q = 0
    values.append(guided_qgen_q)

    # 3. Guided FindC
    try:
        guided_findc_q = findc_data['growacq'][benchmark]['max_viol']['rel_dim_block']['guided-findc']['tot_q']
    except KeyError:
        guided_findc_q = 0
    values.append(guided_findc_q)
    
    # 4. FindScope-3 not guided (cb-findscope-half)
    try:
        findscope_half_q = findscope_data['growacq'][benchmark]['max_viol']['rel_dim_block']['cb-findscope-half']['tot_q']
    except KeyError:
        findscope_half_q = 0
    values.append(findscope_half_q)
    
    # 5. Guided FindScope-2
    try:
        guided_findscope2_q = findscope_data['growacq'][benchmark]['max_viol']['rel_dim_block']['guided-findscope2']['tot_q']
    except KeyError:
        guided_findscope2_q = 0
    values.append(guided_findscope2_q)
    
    # 6. Guided FindScope-3 (cb-findscope)
    try:
        guided_findscope_q = findscope_data['growacq'][benchmark]['max_viol']['rel_dim_block']['cb-findscope']['tot_q']
    except KeyError:
        guided_findscope_q = 0
    values.append(guided_findscope_q)
    
    # 7. Guide All 
    try:
        guided_all = classifier_guide_all_data['growacq'][benchmark]['obj_proba2']['rel_dim_block']['random_forest']['tot_q']
    except KeyError:
        guided_all = 0
    values.append(guided_all)

    x = np.arange(len(labels))
    bars = plt.bar(x, values, color=bar_colors, edgecolor='black', linewidth=2)

    # Add numbers above bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{int(round(height))}',
                 ha='center', va='bottom', fontsize=20)

    plt.xticks(x, labels, rotation=45, ha='right')
    plt.ylabel('# of Queries')
    plt.title(benchmark_names.get(benchmark, benchmark))
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/q4_{benchmark}_queries.png', bbox_inches='tight', dpi=300)
    plt.close()

# Create a separate legend figure
plt.figure(figsize=(10, 1))
handles = [
    plt.Rectangle((0, 0), 1, 1, color=bar_colors[0], label=labels[0]),
    plt.Rectangle((0, 0), 1, 1, color=bar_colors[1], label=labels[1]),
    plt.Rectangle((0, 0), 1, 1, color=bar_colors[2], label=labels[2]),
    plt.Rectangle((0, 0), 1, 1, color=bar_colors[3], label=labels[3]),
    plt.Rectangle((0, 0), 1, 1, color=bar_colors[4], label=labels[4]),
    plt.Rectangle((0, 0), 1, 1, color=bar_colors[5], label=labels[5]),
    plt.Rectangle((0, 0), 1, 1, color=bar_colors[6], label=labels[6]),
]
plt.legend(handles, labels, loc='center', ncol=7)
plt.axis('off')
plt.savefig(f'{output_dir}/q4_legend.png', bbox_inches='tight', dpi=300)
plt.close()

# --- One figure with all benchmarks ---
plt.figure(figsize=(24, 5))
benchmarks_pretty = [benchmark_names.get(b, b) for b in benchmarks]

# Collect values for all benchmarks
baseline_qs = []
guided_qgen_qs = []
guided_findc_qs = []
findscope_half_qs = []
guided_findscope2_qs = []
guided_findscope_qs = []
guided_all_qs = []

for b in benchmarks:
    # Baseline
    baseline_qs.append(baseline_data['growacq'][b]['tot_q'])
    
    # Guided QGen
    try:
        q = classifier_data['growacq'][b]['obj_proba2']['rel_dim_block']['random_forest']['tot_q']
    except KeyError:
        q = 0
    guided_qgen_qs.append(q)
    
    # Guided FindC
    try:
        q = findc_data['growacq'][b]['max_viol']['rel_dim_block']['guided-findc']['tot_q']
    except KeyError:
        q = 0
    guided_findc_qs.append(q)
    
    # FindScope-3 not guided
    try:
        q = findscope_data['growacq'][b]['max_viol']['rel_dim_block']['cb-findscope-half']['tot_q']
    except KeyError:
        q = 0
    findscope_half_qs.append(q)
    
    # Guided FindScope-2
    try:
        q = findscope_data['growacq'][b]['max_viol']['rel_dim_block']['guided-findscope2']['tot_q']
    except KeyError:
        q = 0
    guided_findscope2_qs.append(q)
    
    # Guided FindScope-3
    try:
        q = findscope_data['growacq'][b]['max_viol']['rel_dim_block']['cb-findscope']['tot_q']
    except KeyError:
        q = 0
    guided_findscope_qs.append(q)

    # Guide All
    try:
        q = classifier_guide_all_data['growacq'][b]['obj_proba2']['rel_dim_block']['random_forest']['tot_q']
    except KeyError:
        q = 0
    guided_all_qs.append(q)

# Function to plot a subset of benchmarks
def plot_benchmark_subset(start_idx, end_idx, title_suffix):
    n_benchmarks = end_idx - start_idx
    bar_width = 0.25  # Increased from 0.15 to 0.2
    x = np.arange(n_benchmarks) * 2  # Increased spacing between benchmark groups to accommodate wider bars
    
    # Plot bars for each approach
    bars1 = plt.bar(x - 3*bar_width, baseline_qs[start_idx:end_idx], bar_width, color=bar_colors[0], edgecolor='black', linewidth=2)
    bars2 = plt.bar(x - 2*bar_width, guided_qgen_qs[start_idx:end_idx], bar_width, color=bar_colors[1], edgecolor='black', linewidth=2)
    bars3 = plt.bar(x - bar_width, guided_findc_qs[start_idx:end_idx], bar_width, color=bar_colors[2], edgecolor='black', linewidth=2)
    bars4 = plt.bar(x, findscope_half_qs[start_idx:end_idx], bar_width, color=bar_colors[3], edgecolor='black', linewidth=2)
    bars5 = plt.bar(x + bar_width, guided_findscope2_qs[start_idx:end_idx], bar_width, color=bar_colors[4], edgecolor='black', linewidth=2)
    bars6 = plt.bar(x + 2*bar_width, guided_findscope_qs[start_idx:end_idx], bar_width, color=bar_colors[5], edgecolor='black', linewidth=2)
    bars7 = plt.bar(x + 3*bar_width, guided_all_qs[start_idx:end_idx], bar_width, color=bar_colors[6], edgecolor='black', linewidth=2)

    # Add numbers above bars
    for bars in [bars1, bars2, bars3, bars4, bars5, bars6, bars7]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height, f'{int(round(height))}',
                     ha='center', va='bottom', fontsize=15)

    plt.xticks(x, benchmarks_pretty[start_idx:end_idx], rotation=0, ha='center')
    plt.ylabel('# of Queries')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/q4_benchmarks_{title_suffix}_queries.png', bbox_inches='tight', dpi=300)
    plt.close()

    # Percentage figure
    plt.figure(figsize=(16, 5))  # Increased width from 12 to 16
    
    # Calculate percentages relative to baseline
    guided_qgen_percentages = [100 * (q/b) if b > 0 else 0 for q, b in zip(guided_qgen_qs[start_idx:end_idx], baseline_qs[start_idx:end_idx])]
    guided_findc_percentages = [100 * (q/b) if b > 0 else 0 for q, b in zip(guided_findc_qs[start_idx:end_idx], baseline_qs[start_idx:end_idx])]
    findscope_half_percentages = [100 * (q/b) if b > 0 else 0 for q, b in zip(findscope_half_qs[start_idx:end_idx], baseline_qs[start_idx:end_idx])]
    guided_findscope2_percentages = [100 * (q/b) if b > 0 else 0 for q, b in zip(guided_findscope2_qs[start_idx:end_idx], baseline_qs[start_idx:end_idx])]
    guided_findscope_percentages = [100 * (q/b) if b > 0 else 0 for q, b in zip(guided_findscope_qs[start_idx:end_idx], baseline_qs[start_idx:end_idx])]
    guided_all_percentages = [100 * (q/b) if b > 0 else 0 for q, b in zip(guided_all_qs[start_idx:end_idx], baseline_qs[start_idx:end_idx])]

    # Plot bars for each approach
    bars2 = plt.bar(x - 2*bar_width, guided_qgen_percentages, bar_width, color=bar_colors[1], edgecolor='black', linewidth=2)
    bars3 = plt.bar(x - bar_width, guided_findc_percentages, bar_width, color=bar_colors[2], edgecolor='black', linewidth=2)
    bars4 = plt.bar(x, findscope_half_percentages, bar_width, color=bar_colors[3], edgecolor='black', linewidth=2)
    bars5 = plt.bar(x + bar_width, guided_findscope2_percentages, bar_width, color=bar_colors[4], edgecolor='black', linewidth=2)
    bars6 = plt.bar(x + 2*bar_width, guided_findscope_percentages, bar_width, color=bar_colors[5], edgecolor='black', linewidth=2)
    bars7 = plt.bar(x + 3*bar_width, guided_all_percentages, bar_width, color=bar_colors[6], edgecolor='black', linewidth=2)

    # Add percentage values above bars
    for bars in [bars2, bars3, bars4, bars5, bars6, bars7]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height, f'{int(round(height))}%',
                     ha='center', va='bottom', fontsize=12)

    plt.xticks(x, benchmarks_pretty[start_idx:end_idx], rotation=0, ha='center')
    plt.ylabel('% of Baseline Queries')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/q4_benchmarks_{title_suffix}_percentages.png', bbox_inches='tight', dpi=300)
    plt.close()

# Plot first 4 benchmarks
plt.figure(figsize=(24, 6))  # Increased width from 12 to 16
plot_benchmark_subset(0, 4, 'first4')

# Plot last 3 benchmarks
plt.figure(figsize=(18, 6))  # Increased width from 12 to 16
plot_benchmark_subset(4, 7, 'last3')

# Plot all benchmarks (original code)
plt.figure(figsize=(24, 6))
n_benchmarks = len(benchmarks)
bar_width = 0.12
x = np.arange(n_benchmarks) * 1.2

# Plot bars for each approach
bars1 = plt.bar(x - 3*bar_width, baseline_qs, bar_width, color=bar_colors[0], edgecolor='black', linewidth=2)
bars2 = plt.bar(x - 2*bar_width, guided_qgen_qs, bar_width, color=bar_colors[1], edgecolor='black', linewidth=2)
bars3 = plt.bar(x - bar_width, guided_findc_qs, bar_width, color=bar_colors[2], edgecolor='black', linewidth=2)
bars4 = plt.bar(x, findscope_half_qs, bar_width, color=bar_colors[3], edgecolor='black', linewidth=2)
bars5 = plt.bar(x + bar_width, guided_findscope2_qs, bar_width, color=bar_colors[4], edgecolor='black', linewidth=2)
bars6 = plt.bar(x + 2*bar_width, guided_findscope_qs, bar_width, color=bar_colors[5], edgecolor='black', linewidth=2)
bars7 = plt.bar(x + 3*bar_width, guided_all_qs, bar_width, color=bar_colors[6], edgecolor='black', linewidth=2)

# Add numbers above bars
for bars in [bars1, bars2, bars3, bars4, bars5, bars6, bars7]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{int(round(height))}',
                 ha='center', va='bottom', fontsize=12)

plt.xticks(x, benchmarks_pretty, rotation=0, ha='center')
plt.ylabel('# of Queries')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f'{output_dir}/q4_all_benchmarks_queries.png', bbox_inches='tight', dpi=300)
plt.close()

# --- Create percentage comparison figure ---
plt.figure(figsize=(24, 5))

# Calculate percentages relative to baseline
guided_qgen_percentages = [100 * (q/b) if b > 0 else 0 for q, b in zip(guided_qgen_qs, baseline_qs)]
guided_findc_percentages = [100 * (q/b) if b > 0 else 0 for q, b in zip(guided_findc_qs, baseline_qs)]
findscope_half_percentages = [100 * (q/b) if b > 0 else 0 for q, b in zip(findscope_half_qs, baseline_qs)]
guided_findscope2_percentages = [100 * (q/b) if b > 0 else 0 for q, b in zip(guided_findscope2_qs, baseline_qs)]
guided_findscope_percentages = [100 * (q/b) if b > 0 else 0 for q, b in zip(guided_findscope_qs, baseline_qs)]
guided_all_percentages = [100 * (q/b) if b > 0 else 0 for q, b in zip(guided_all_qs, baseline_qs)]

# Plot bars for each approach
bars2 = plt.bar(x - 2*bar_width, guided_qgen_percentages, bar_width, color=bar_colors[1], edgecolor='black', linewidth=2)
bars3 = plt.bar(x - bar_width, guided_findc_percentages, bar_width, color=bar_colors[2], edgecolor='black', linewidth=2)
bars4 = plt.bar(x, findscope_half_percentages, bar_width, color=bar_colors[3], edgecolor='black', linewidth=2)
bars5 = plt.bar(x + bar_width, guided_findscope2_percentages, bar_width, color=bar_colors[4], edgecolor='black', linewidth=2)
bars6 = plt.bar(x + 2*bar_width, guided_findscope_percentages, bar_width, color=bar_colors[5], edgecolor='black', linewidth=2)
bars7 = plt.bar(x + 3*bar_width, guided_all_percentages, bar_width, color=bar_colors[6], edgecolor='black', linewidth=2)

# Add percentage values above bars
for bars in [bars2, bars3, bars4, bars5, bars6, bars7]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{int(round(height))}%',
                 ha='center', va='bottom', fontsize=12)

plt.xticks(x, benchmarks_pretty, rotation=0, ha='center')
plt.ylabel('% of Baseline Queries')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f'{output_dir}/q4_all_benchmarks_percentages.png', bbox_inches='tight', dpi=300)
plt.close() 