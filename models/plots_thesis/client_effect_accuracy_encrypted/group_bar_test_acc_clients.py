import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSV data
csv_path = 'files/clients_effect-cfr_testacc.csv'
df = pd.read_csv(csv_path)

# Extract experiment type and client count
df['Experiment'] = df['Name'].str.extract(r'1e_\dcl_(\w+)')[0]
df['Clients'] = df['Name'].str.extract(r'_(\dcl)')[0]

# Group and calculate mean test accuracy for each experiment and client count
grouped_df = df.groupby(['Clients', 'Experiment'])['test_accuracy'].mean().reset_index()

# Create a new DataFrame for the bar chart
bar_chart_df = pd.DataFrame()

# Populate the new DataFrame with mean test accuracies for each experiment and client count
for experiment in df['Experiment'].unique():
    bar_chart_df[experiment] = grouped_df[grouped_df['Experiment'] == experiment].set_index('Clients')['test_accuracy']

print(bar_chart_df)
# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

# Set position of bar on X axis
bar_width = 0.15
r = np.arange(len(bar_chart_df.index))

# Make the plot for each experiment
for i, column in enumerate(bar_chart_df.columns):
    print(i, column)
    ax.bar(r + i * bar_width, bar_chart_df[column], width=bar_width, label=column)

# Add the centralized baseline accuracy to the DataFrame
centralized_accuracy = 0.9103  # replace with the actual centralized accuracy value
# bar_chart_df.loc['Centralized'] = [centralized_accuracy] * len(df['Experiment'].unique())

ax.bar(r + 5 * bar_width, 0.9103, width=bar_width, label='Centralized')

# Add xticks on the middle of the group bars
ax.set_xlabel('Clients')
ax.set_xticks(r + bar_width * (len(bar_chart_df.columns) - 1) / 2)
ax.set_xticklabels(bar_chart_df.index)


# Set the y-axis label
ax.set_ylabel('Test Accuracy')

# Add title and legend
ax.set_title('Grouped Bar Chart of Test Accuracies by Client Count')
ax.legend(title='Experiment Type', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.show()
