ml_dict = {
    'random_forest': 'RF',
    'decision_tree': 'DT',
    'svm': 'SVM',
    'logistic_regression': 'LR',
    'mlp': 'MLP',
    'naive_bayes': 'NB',
}

feature_dict = {
    'rel_dim_block': 'Full',
    'rel_dim': 'RelDim',
    'simple_rel': 'Rel',
}

# Define colors, line styles and markers
# Using the new palette provided by the user
colors = [
    '#6f1926',  # dark burgundy
    '#de324c',  # red
    '#f4895f',  # peach
    '#f8e16f',  # light yellow
    '#95cf92',  # light green
    '#369acc',  # blue
    '#9656a2',  # purple
    '#cbabd1',  # lavender
]

linestyles = ["-", "--", "-.", ":", "-", "--", "-."]
markers = ["4", "o", "p", "1", "+", "x", "."]

# Define consistent colors for each classifier
classifier_colors = {
    'logistic_regression': colors[0],  # dark burgundy
    'random_forest': colors[1],       # red
    'naive_bayes': colors[2],         # peach
    'decision_tree': colors[3],       # light yellow
    'mlp': colors[4],                 # light green
    'svm': colors[5],                 # blue
}

