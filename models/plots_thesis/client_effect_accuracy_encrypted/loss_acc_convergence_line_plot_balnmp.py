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
# csv_path = 'loss_acc_clients_cfr.csv'
csv_path = 'files/loss_acc_clients_balnmp.csv'

# Read the CSV data
df = pd.read_csv(csv_path)

# Set the epoch as the index
df.set_index('Epoch', inplace=True)

# Define the colors for each line
colors = {
    'accuracy_2cl': 'darkorange',
    'accuracy_4cl': 'maroon',
    'accuracy_8cl': 'slateblue',
}

# Create a figure for accuracy
fig_accuracy, ax_accuracy = plt.subplots(figsize=(14, 7))

# Plot accuracy for each client count on the accuracy subplot (multiply by 100)
for cl in ['2cl', '4cl', '8cl']:
    ax_accuracy.plot(df.index, df[f'BALNMP_1e_50r_{cl}_FAWE - round_accuracy'] * 100, label=f'{cl} Accuracy', color=colors[f'accuracy_{cl}'])

# Read the JSON data for the baseline experiment
baseline_json_path = 'files/training_metrics_balnmp_centralized.json'
with open(baseline_json_path, 'r') as json_file:
    baseline_data = json.load(json_file)

# Extract accuracy data from the baseline JSON
baseline_epochs = [epoch['epoch'] for epoch in baseline_data['epochs']]
baseline_accuracy = [epoch['val_accuracy'] for epoch in baseline_data['epochs']]

# Plot the baseline experiment accuracy on the accuracy subplot
# ax_accuracy.plot(baseline_epochs, baseline_accuracy, label='Baseline Accuracy', color='red')
ax_accuracy.axhline(y=82.13, color='sienna', linestyle='--')

# Set labels and legend for the accuracy subplot
ax_accuracy.set_xlabel('Communication Round')
ax_accuracy.set_ylabel('Accuracy (%)')
ax_accuracy.legend()
ax_accuracy.set_xlim(left=1)
ax_accuracy.grid(True)

# Save the accuracy plot as an pdf file
fig_accuracy.savefig('plots/client_effect_accuracy_plot_FAWE_clients_BALNMP.pdf', format='pdf')

# Show the accuracy plot
plt.show()



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
csv_path = 'files/loss_acc_clients_balnmp.csv'

# Read the CSV data
df = pd.read_csv(csv_path)

# Set the epoch as the index
df.set_index('Epoch', inplace=True)

# Define the colors for each line
colors = {
    'loss_2cl': 'darkorange',
    'loss_4cl': 'maroon',
    'loss_8cl': 'slateblue',
}

# Create a figure for loss
fig_loss, ax_loss = plt.subplots(figsize=(14, 7))

# Plot loss for each client count on the loss subplot
for cl in ['2cl', '4cl', '8cl']:
    ax_loss.plot(df.index, df[f'BALNMP_1e_50r_{cl}_FAWE - round_loss'], label=f'{cl} Loss', color=colors[f'loss_{cl}'])

# Read the JSON data for the baseline experiment
baseline_json_path = 'files/training_metrics_balnmp_centralized.json'
with open(baseline_json_path, 'r') as json_file:
    baseline_data = json.load(json_file)

# Extract loss data from the baseline JSON
baseline_epochs = [epoch['epoch'] for epoch in baseline_data['epochs']]
baseline_loss = [epoch['train_loss'] for epoch in baseline_data['epochs']]

# Plot the baseline experiment loss on the loss subplot
# ax_loss.plot(baseline_epochs, baseline_loss, label='Baseline Loss', color='orchid')
ax_loss.axhline(y=92.64574242490943, color='sienna', linestyle='--')
# Set labels and legend for the loss subplot
ax_loss.set_xlabel('Communication Round')
ax_loss.set_ylabel('Loss')

ax_loss.legend()
ax_loss.grid(True)
ax_loss.set_xlim(left=1)


# Save the loss plot as an pdf file
fig_loss.savefig('plots/client_effect_loss_plot_FAWE_clients_BALNMP.pdf', format='pdf')

# Show the loss plot
plt.show()
