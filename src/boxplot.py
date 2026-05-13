from sklearn.datasets import fetch_california_housing
import pandas as pd

# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)

# Features + target as a single DataFrame
df = housing.frame

# Quick check
print(df.head())
print(df.shape)

# Create boxplot
df.boxplot(figsize=(12,6))

# Adjust layout
plt.tight_layout()

# Save boxplot in figs folder
plt.savefig("figs/boxplot.png")