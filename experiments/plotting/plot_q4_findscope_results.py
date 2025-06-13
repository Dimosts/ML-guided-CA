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

# Variant mapping and ordering
variant_info = {
    'baseline': {
        'label': 'FS-2',
        'color': colors[0]
    },
    'guided-findscope2': {
        'label': 'FS-2 guided',
        'color': colors[2]
    },
    'cb-findscope-half': {
        'label': 'FS-3',
        'color': colors[4]
    },
    'cb-findscope': {
        'label': 'FS-3 guided',
        'color': colors[3]
    }
}

# Load the data
with open('experiments/parsed_results/parsed_baseline_results.json', 'r') as f:
    baseline_data = json.load(f)

with open('experiments/parsed_results/parsed_findscope_results.json', 'r') as f:
    findscope_data = json.load(f)

# Benchmarks to plot (from baseline file)
benchmarks = list(baseline_data['growacq'].keys())

# Output directory
output_dir = 'experiments/plots/q4/findscope'
os.makedirs(output_dir, exist_ok=True)

# Legend labels and colors
labels = [variant_info['baseline']['label'], 
          variant_info['guided-findscope2']['label'],
          variant_info['cb-findscope-half']['label'],
          variant_info['cb-findscope']['label']]
bar_colors = [variant_info['baseline']['color'],
              variant_info['guided-findscope2']['color'],
              variant_info['cb-findscope-half']['color'],
              variant_info['cb-findscope']['color']]

# For each benchmark, plot the number of queries
for benchmark in benchmarks:
    plt.figure(figsize=(12, 6))
    # Baseline
    baseline_q = baseline_data['growacq'][benchmark]['tot_q']
    
    # Findscope variants in desired order
    findscope_variants = ['guided-findscope2', 'cb-findscope-half', 'cb-findscope']
    findscope_qs = []
    
    for variant in findscope_variants:
        try:
            print(f"Trying to access {benchmark} - {variant}")
            q = findscope_data['growacq'][benchmark]['max_viol']['rel_dim_block'][variant]['tot_q']
            print(f"Found value: {q}")
        except KeyError as e:
            print(f"KeyError for {benchmark} - {variant}: {e}")
            q = 0
        findscope_qs.append(q)
    
    values = [baseline_q] + findscope_qs
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
    plt.savefig(f'{output_dir}/findscope_{benchmark}_queries.png', bbox_inches='tight', dpi=300)
    plt.close()

# Create a separate legend figure
plt.figure(figsize=(10, 1))
handles = [
    plt.Rectangle((0, 0), 1, 1, color=variant_info['baseline']['color'], label=variant_info['baseline']['label']),
    plt.Rectangle((0, 0), 1, 1, color=variant_info['guided-findscope2']['color'], label=variant_info['guided-findscope2']['label']),
    plt.Rectangle((0, 0), 1, 1, color=variant_info['cb-findscope-half']['color'], label=variant_info['cb-findscope-half']['label']),
    plt.Rectangle((0, 0), 1, 1, color=variant_info['cb-findscope']['color'], label=variant_info['cb-findscope']['label']),
]
plt.legend(handles, [h.get_label() for h in handles], loc='center', ncol=4)
plt.axis('off')
plt.savefig(f'{output_dir}/findscope_legend.png', bbox_inches='tight', dpi=300)
plt.close()

# --- One figure with all benchmarks ---
plt.figure(figsize=(18, 5))
benchmarks_pretty = [benchmark_names.get(b, b) for b in benchmarks]

# Collect values for all benchmarks
baseline_qs = [baseline_data['growacq'][b]['tot_q'] for b in benchmarks]
findscope_variants = ['guided-findscope2', 'cb-findscope-half', 'cb-findscope']
findscope_qs_by_variant = {variant: [] for variant in findscope_variants}

print("\nCollecting data for combined plot:")
for b in benchmarks:
    print(f"\nBenchmark: {b}")
    for variant in findscope_variants:
        try:
            q = findscope_data['growacq'][b]['max_viol']['rel_dim_block'][variant]['tot_q']
            print(f"  {variant}: {q}")
        except KeyError as e:
            print(f"  {variant}: KeyError - {e}")
            q = 0
        findscope_qs_by_variant[variant].append(q)

print("\nCollected data:")
for variant, qs in findscope_qs_by_variant.items():
    print(f"{variant}: {qs}")

n_benchmarks = len(benchmarks)
bar_width = 0.2  # Increased bar width since we have fewer variants
x = np.arange(n_benchmarks)

# Plot bars for each variant in the correct order
bars = []
# First plot baseline
bars.append(plt.bar(x - 1.5*bar_width, baseline_qs, bar_width, 
                   label=variant_info['baseline']['label'], 
                   color=variant_info['baseline']['color'], 
                   edgecolor='black', linewidth=2))

# Then plot other variants in the desired order
for variant in findscope_variants:
    offset = (findscope_variants.index(variant) - 0.5) * bar_width
    bars.append(plt.bar(x + offset, findscope_qs_by_variant[variant], bar_width,
                       label=variant_info[variant]['label'],
                       color=variant_info[variant]['color'],
                       edgecolor='black', linewidth=2))

# Add numbers above bars
for bar_group in bars:
    for bar in bar_group:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{int(round(height))}',
                 ha='center', va='bottom', fontsize=12)

plt.xticks(x, benchmarks_pretty, rotation=0, ha='center')
plt.ylabel('# of Queries')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig(f'{output_dir}/findscope_all_benchmarks_queries.png', bbox_inches='tight', dpi=300)
plt.close() 