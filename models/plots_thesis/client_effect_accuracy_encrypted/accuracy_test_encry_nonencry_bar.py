import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 17,  # Choose a size that fits well in your document
    'font.family': 'Arial',  # Or 'Times New Roman', or another
    'xtick.labelsize': 17,  # Smaller size for axis ticks if needed
    'ytick.labelsize': 17
})
# Test accuracies for each method (replace these with your actual accuracies)
accuracies = {
    'Benchmark': 0.95,  # Replace with your benchmark accuracy
    'BC_BFV': 0.92,
    'BC_CKKS': 0.93,
    'FA_CKKS': 0.91,
    'GBFA_CKKS': 0.90
}

# Create lists of names and values for plotting
names = list(accuracies.keys())
values = list(accuracies.values())

# Create the bar chart
plt.figure(figsize=(10, 6))
bars = plt.bar(names, values, color=['green' if name == 'Benchmark' else 'blue' for name in names])

# Highlight the benchmark bar
benchmark_bar = bars[names.index('Benchmark')]
benchmark_bar.set_color('green')

# Add a title and labels
plt.title('Comparison of Test Accuracies')
plt.xlabel('Method')
plt.ylabel('Test Accuracy')

# Optionally, add the actual accuracy values on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

# Save the plot as an SVG file
svg_filename = 'test_accuracy_comparison.svg'  # Update this to your desired file path
plt.savefig(svg_filename, format='svg')

# Show the plot
plt.show()
