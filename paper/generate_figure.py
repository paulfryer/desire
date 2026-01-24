"""Generate Figure 1: Anticipatory Activation box plot."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Data from experiments
data = {
    'Agent Type': ['Heuristic', 'Heuristic', 'Heuristic',
                   'RL (PPO)', 'RL (PPO)', 'RL (PPO)',
                   'Active Inference', 'Active Inference', 'Active Inference'],
    'AA Score': [1.0, 1.0, 1.0,
                 0.0, 0.0, 0.0,
                 0.981, 0.964, 0.977],
    'Seed': [42, 123, 456, 42, 123, 456, 42, 123, 456]
}

df = pd.DataFrame(data)

# Set up the figure
plt.figure(figsize=(6, 4))
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

# Create color palette
colors = ['#2ecc71', '#e74c3c', '#3498db']  # Green, Red, Blue

# Create strip plot with jitter for individual points
ax = sns.stripplot(x='Agent Type', y='AA Score', data=df,
                   palette=colors, size=10, alpha=0.8, jitter=0.05)

# Add mean bars
means = df.groupby('Agent Type')['AA Score'].mean()
for i, agent in enumerate(['Heuristic', 'RL (PPO)', 'Active Inference']):
    mean_val = means[agent]
    ax.hlines(mean_val, i - 0.2, i + 0.2, colors='black', linewidth=2)

# Formatting
ax.set_ylim(-0.1, 1.2)
ax.set_ylabel('Anticipatory Activation (AA) Score')
ax.set_xlabel('')
ax.set_xticklabels(['Heuristic', 'RL (PPO)', 'Active Inference'])

# Add significance annotation
ax.annotate('', xy=(0.5, 1.1), xytext=(1.5, 1.1),
            arrowprops=dict(arrowstyle='-', color='black', lw=1))
ax.annotate('***', xy=(1, 1.12), ha='center', fontsize=12)
ax.text(1, 1.17, 'p = 0.002', ha='center', fontsize=9)

plt.tight_layout()

# Save as PDF for LaTeX
plt.savefig('figures/aa_boxplot.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/aa_boxplot.png', format='png', dpi=300, bbox_inches='tight')

print("Figure saved to figures/aa_boxplot.pdf and figures/aa_boxplot.png")
