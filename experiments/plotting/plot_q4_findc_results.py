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
    'jobshop': 'JS',
    'gtsudoku': 'GTSudoku',
    'random': 'Random',
    'jsudoku': 'JSudoku',
    'exam_timetable': 'ET',
    'sudoku': 'Sudoku',
    'nurse_rostering': 'NR'
}

# Load the data
with open('experiments/parsed_results/parsed_baseline_results.json', 'r') as f:
    baseline_data = json.load(f)

with open('experiments/parsed_results/parsed_findc_results.json', 'r') as f:
    findc_data = json.load(f)

# Benchmarks to plot (from baseline file)
benchmarks = list(baseline_data['growacq'].keys())

# Output directory
output_dir = 'experiments/plots/q4/findc'
os.makedirs(output_dir, exist_ok=True)

# Legend labels and colors
labels = ['Not guided', 'guided-findc']
bar_colors = [
    colors[0],    # Baseline (dark burgundy)
    colors[5],    # guided-findc (blue)
]

# For each benchmark, plot the number of queries
for benchmark in benchmarks:
    plt.figure(figsize=(8, 6))
    # Baseline
    baseline_q = baseline_data['growacq'][benchmark]['tot_q']
    # guided-findc
    try:
        guided_findc_q = findc_data['growacq'][benchmark]['max_viol']['rel_dim_block']['guided-findc']['tot_q']
    except KeyError:
        guided_findc_q = 0
    
    values = [baseline_q, guided_findc_q]
    x = np.arange(len(labels))
    bars = plt.bar(x, values, color=bar_colors, edgecolor='black', linewidth=2)

    # Add numbers above bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{int(round(height))}',
                 ha='center', va='bottom', fontsize=20)

    plt.xticks(x, labels, rotation=0)
    plt.ylabel('# of Queries')
    plt.title(benchmark_names.get(benchmark, benchmark))
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/findc_{benchmark}_queries.png', bbox_inches='tight', dpi=300)
    plt.close()

# Create a separate legend figure
plt.figure(figsize=(10, 1))
handles = [
    plt.Rectangle((0, 0), 1, 1, color=bar_colors[0], label=labels[0]),
    plt.Rectangle((0, 0), 1, 1, color=bar_colors[1], label=labels[1]),
]
plt.legend(handles, labels, loc='center', ncol=2)
plt.axis('off')
plt.savefig(f'{output_dir}/findc_legend.png', bbox_inches='tight', dpi=300)
plt.close()

# --- New: One figure with all benchmarks ---
plt.figure(figsize=(18, 5))
benchmarks_pretty = [benchmark_names.get(b, b) for b in benchmarks]

# Collect values for all benchmarks
baseline_qs = [baseline_data['growacq'][b]['tot_q'] for b in benchmarks]
guided_findc_qs = []
for b in benchmarks:
    try:
        q = findc_data['growacq'][b]['max_viol']['rel_dim_block']['guided-findc']['tot_q']
    except KeyError:
        q = 0
    guided_findc_qs.append(q)

n_benchmarks = len(benchmarks)
bar_width = 0.35
x = np.arange(n_benchmarks)

bars1 = plt.bar(x - bar_width/2, baseline_qs, bar_width, label='Not guided', color=bar_colors[0], edgecolor='black', linewidth=2)
bars2 = plt.bar(x + bar_width/2, guided_findc_qs, bar_width, label='guided-findc', color=bar_colors[1], edgecolor='black', linewidth=2)

# Add numbers above bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{int(round(height))}',
                 ha='center', va='bottom', fontsize=16)

plt.xticks(x, benchmarks_pretty, rotation=0, ha='center')
plt.ylabel('# of Queries')
#plt.title('Number of Queries for Each Benchmark')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig(f'{output_dir}/findc_all_benchmarks_queries.png', bbox_inches='tight', dpi=300)
plt.close() 