import pandas as pd
import matplotlib.pyplot as plt
import json

plt.rcParams.update({
    'font.size': 17,  # Choose a size that fits well in your document
    'font.family': 'Arial',  # Or 'Times New Roman', or another
    'xtick.labelsize': 17,  # Smaller size for axis ticks if needed
    'ytick.labelsize': 17
})
def plot_combined_accuracy_loss(df, baseline_data, metrics):
    # Plot Accuracy
    fig_accuracy, ax_accuracy = plt.subplots(figsize=(14, 7))
    colors_accuracy =  ['blue', 'orange', 'purple', 'black', 'cyan']   # Colors for accuracy lines

    # Plot accuracy for each metric
    for metric, color in zip(metrics, colors_accuracy):
        ax_accuracy.plot(df.index, df[f'BALNMP_50r_1e_2cl_{metric} - round_accuracy'] * 100, label=f'{metric} Accuracy', color=color)

    # Plot Baseline Accuracy
    baseline_accuracy = [epoch['val_accuracy'] for epoch in baseline_data['epochs']][:50]
    ax_accuracy.plot(range(1, len(baseline_accuracy) + 1), baseline_accuracy,  label='Baseline Accuracy', color='sienna', linestyle='--')

    ax_accuracy.set_xlabel('Communication Round')
    ax_accuracy.set_ylabel('Accuracy (%)')
    ax_accuracy.legend()
    # ax_accuracy.set_title('Combined Accuracy over Epochs for BALNMP')
    ax_accuracy.grid(True)
    ax_accuracy.set_xlim(left=1)
    fig_accuracy.savefig('plots/algo_effect_combined_accuracy_plot_banlnmp.pdf', format='pdf')
    plt.close(fig_accuracy)

    # Plot Loss
    fig_loss, ax_loss = plt.subplots(figsize=(14, 7))
    colors_loss =  ['blue', 'orange', 'purple', 'black', 'cyan']   # Colors for loss lines

    # Plot loss for each metric
    for metric, color in zip(metrics, colors_loss):
        ax_loss.plot(df.index, df[f'BALNMP_50r_1e_2cl_{metric} - round_loss'], label=f'{metric} Loss', color=color)

    # Plot Baseline Loss
    baseline_loss = [epoch['train_loss'] for epoch in baseline_data['epochs']][:50]
    ax_loss.plot(range(1, len(baseline_loss) + 1), baseline_loss,  label='Baseline Loss', color='sienna', linestyle='--')

    ax_loss.set_xlabel('Communication Round')
    ax_loss.set_ylabel('Loss')
    ax_loss.legend()
    # ax_loss.set_title('Combined Loss over Epochs for BALNMP')
    ax_loss.grid(True)
    ax_loss.set_xlim(left=1)
    fig_loss.savefig('plots/algo_effect_combined_loss_plot_balnmp.pdf', format='pdf')
    plt.close(fig_loss)

# Load the data
csv_path = 'updated_balnmp_50r1e_loss_acc.csv'
json_path = 'files/training_metrics_balnmp_centralized.json'
df = pd.read_csv(csv_path)
df.set_index('Epoch', inplace=True)

# Load the baseline data
with open(json_path, 'r') as file:
    baseline_data = json.load(file)

# Define the metrics
metrics = ['BC_BFV', 'BC_CKKS', 'FA_CKKS', 'GBFA_CKKS', 'FAWE']

# Generate the combined plots for accuracy and loss
plot_combined_accuracy_loss(df, baseline_data, metrics)
