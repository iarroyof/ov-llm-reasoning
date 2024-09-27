# This was prototype of https://colab.research.google.com/drive/1-FwFkgE6xnjGqgqASTfoAj3KEeOT8V6e?authuser=1#scrollTo=cUX13ZmHcojq&line=10&uniqifier=1
# Training data was declared on https://colab.research.google.com/drive/1-FwFkgE6xnjGqgqASTfoAj3KEeOT8V6e?authuser=1#scrollTo=PlDNfl98DDAB&line=211&uniqifier=1
from phenvs import PhraseEntropyViews

pev = PhraseEntropyViews()

# Assuming you have your batched_data ready
result = pev.fit_entropies(batched_data, return_results=True)

# Plot scatter plot
x_coords, y_coords, labels = pev.plot_scatter(result)

# Get colors for the scatter plot
colors = ['blue' if label == 'high_info' else 'red' for label in labels]

# Train and plot MLP
mlp, X, y = pev.train_mlp(x_coords, y_coords, labels)
pev.plot_mlp_decision_boundary(mlp, X, y, x_coords, y_coords, colors, 'MLP Decision Boundary')

# Train and plot Online SVM
svm, scaler, X, y = pev.train_svm(x_coords, y_coords, labels)
pev.plot_svm_decision_boundary(svm, scaler, X, y, x_coords, y_coords, colors)
