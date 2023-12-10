import pandas as pd
import matplotlib.pyplot as plt
import json

plt.rcParams.update({
    'font.size': 17,  # Choose a size that fits well in your document
    'font.family': 'Arial',  # Or 'Times New Roman', or another
    'xtick.labelsize': 17,  # Smaller size for axis ticks if needed
    'ytick.labelsize': 17
})
# Replace the file path with your actual file path
csv_path = 'files/loss_acc_clients_cfr.csv'

# Read the CSV data
df = pd.read_csv(csv_path)

# Set the epoch as the index
df.set_index('Epoch', inplace=True)

# Define the colors for each line
colors = {
    'accuracy_2cl': 'blue',
    'loss_2cl': 'skyblue',
    'accuracy_4cl': 'green',
    'loss_4cl': 'yellowgreen',
    'accuracy_8cl': 'darkred',
    'loss_8cl': 'lightcoral'
}

# Create a figure with two subplots (one for accuracy and one for loss)
fig, (ax_accuracy, ax_loss) = plt.subplots(2, 1, figsize=(14, 14), sharex=True)

# Plot accuracy for each client count on the accuracy subplot (multiply by 100)
for cl in ['2cl', '4cl', '8cl']:
    ax_accuracy.plot(df.index, df[f'CFR_300r_1e_{cl}_BC_BFV - round_accuracy'] * 100, label=f'{cl} Accuracy', color=colors[f'accuracy_{cl}'])

# Read the JSON data for the baseline experiment
baseline_json_path = 'files/training_metrics_cnn_centralized.json'
with open(baseline_json_path, 'r') as json_file:
    baseline_data = json.load(json_file)

# Extract accuracy data from the baseline JSON
baseline_epochs = [epoch['epoch'] for epoch in baseline_data['epochs']]
baseline_accuracy = [epoch['val_accuracy'] for epoch in baseline_data['epochs']]

# Plot the baseline experiment accuracy on the accuracy subplot
ax_accuracy.plot(baseline_epochs, baseline_accuracy, label='Baseline Accuracy', color='purple')

# Set labels and legend for the accuracy subplot
ax_accuracy.set_ylabel('Accuracy')
ax_accuracy.set_ylim([0, 100])  # Adjust the ylim as needed
ax_accuracy.legend()

# Plot loss for each client count on the loss subplot
for cl in ['2cl', '4cl', '8cl']:
    ax_loss.plot(df.index, df[f'CFR_300r_1e_{cl}_BC_BFV - round_loss'], label=f'{cl} Loss', color=colors[f'loss_{cl}'])

# Extract loss data from the baseline JSON
baseline_loss = [epoch['train_loss'] for epoch in baseline_data['epochs']]

# Plot the baseline experiment loss on the loss subplot
ax_loss.plot(baseline_epochs, baseline_loss, label='Baseline Loss', color='orchid')

# Set labels and legend for the loss subplot
ax_loss.set_xlabel('Epoch')
ax_loss.set_ylabel('Loss')
ax_loss.set_ylim([0, 2])  # Adjust the ylim as needed
ax_loss.legend()

# Show grid on both subplots
ax_accuracy.grid(True)
ax_loss.grid(True)

# Save the accuracy plot as an SVG file
fig_accuracy = plt.figure(1)
fig_accuracy.savefig('accuracy_plot_clients_248_BC_BFV.svg', format='svg')

# Save the loss plot as an SVG file
fig_loss = plt.figure(2)
fig_loss.savefig('loss_plot_clients_248_BC_BFV.svg', format='svg')

# Show the plot
plt.show()
