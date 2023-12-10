import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 16,  # Choose a size that fits well in your document
    'font.family': 'Arial',  # Or 'Times New Roman', or another
    'xtick.labelsize': 16,  # Smaller size for axis ticks if needed
    'ytick.labelsize': 16
})

# Path to the CSV file
csv_path = 'BALNMP_50r_1e.csv'  # Update this to your CSV file path

# Load the CSV file into a DataFrame
data = pd.read_csv(csv_path, sep=';', index_col=0)

# Drop the 'Overall time' row if it's included
data = data.drop('Overall time', errors='ignore')

# Remove the FAWE column and the Quantize and Dequantize rows
data = data.drop(columns=['FAWE', 'Unnamed: 6'], errors='ignore')
data = data.drop(index=['Quantize', 'Dequantize'], errors='ignore')

# Fill NaN values with 0 for stacking
data = data.fillna(0)

# Transpose the DataFrame to have experiments on the x-axis and metrics on the y-axis
data_transposed = data.transpose()

# Define the order for stacking the bars
stack_order = ['Encrypt', 'Train', 'Aggregate', 'Decrypt']
data_transposed = data_transposed[stack_order]

# Plot the stacked bar chart
fig, ax = plt.subplots(figsize=(10, 7))
stacked_bars = data_transposed.plot(kind='bar', stacked=True, ax=ax)

# Define a threshold for when to use the arrow
arrow_threshold = 0.2  # This is an arbitrary number and may need adjusting

# Annotate each bar segment with its value
# for bar in stacked_bars.containers:
#     for rect in bar:
#         # Get the bar height and width
#         height = rect.get_height()
#         width = rect.get_width()
#         # Get the center of the bar segment
#         center_x = rect.get_x() + width / 2
#         # If the bar is too small for the text to fit inside, use an arrow
#         if height < arrow_threshold:
#             # Position the annotation above the bar with an arrow pointing down
#             ax.annotate(f'{height:.2f}',
#                         xy=(center_x, rect.get_y() + height),
#                         xytext=(center_x, rect.get_y() + height + 10),  # Offset by 10 points
#                         ha='center', va='center', fontsize=8,
#                         arrowprops=dict(arrowstyle="->", color='black'))
#         else:
#             # If the bar is big enough, place the label inside the bar
#             ax.annotate(f'{height:.2f}',
#                         (center_x, rect.get_y() + height / 2),
#                         ha='center', va='center', fontsize=8)

# Set the labels and title
ax.set_xlabel('Experiments')
ax.set_ylabel('Time (seconds)')
# ax.set_title('Time Metrics of Different Experiments for BALNMP Model')

# Rotate the x-axis labels for better readability
plt.xticks(rotation=0)

# Save the plot as an SVG file
svg_filename = 'time_effect_stacked_bar_chart_values_with_arrows_balnmp.pdf'  # Update this to your desired file path
plt.savefig(svg_filename, format='pdf')

# Show the plot
plt.show()
