import pandas as pd
import matplotlib.pyplot as plt
import json

plt.rcParams.update({
    'font.size': 17,  # Choose a size that fits well in your document
    'font.family': 'Arial',  # Or 'Times New Roman', or another
    'xtick.labelsize': 17,  # Smaller size for axis ticks if needed
    'ytick.labelsize': 17
})
# Function to plot accuracy and loss
def plot_accuracy_loss(df, baseline_data, metrics, rounds_epochs, plot_type):
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(14, 7))
        colors = {
            'accuracy_200r_1e': 'cyan',
            'accuracy_40r_5e': 'darkorange',
            'accuracy_150r_5e': 'slateblue',
            'loss_200r_1e': 'cyan',
            'loss_40r_5e': 'darkorange',
            'loss_150r_5e': 'slateblue',
        }

        # Plot for each rounds_epochs group
        for r_e in rounds_epochs:
            label = f'{r_e}'
            if plot_type == 'accuracy':
                ax.plot(df.index, df[f'CFR_{r_e}_{metric} - round_accuracy'] * 100, label=label+' Accuracy', color=colors[f'accuracy_{label}'])
                last_epoch = df[f'CFR_{r_e}_{metric} - round_accuracy'].last_valid_index()
            elif plot_type == 'loss':
                ax.plot(df.index, df[f'CFR_{r_e}_{metric} - round_loss'], label=label+' Loss', color=colors[f'accuracy_{label}'])
                last_epoch = df[f'CFR_{r_e}_{metric} - round_loss'].last_valid_index()
            ax.axvline(x=last_epoch, color=colors[f'accuracy_{label}'], linestyle='dotted')

        # Plot the baseline data
        baseline_key = 'val_accuracy' if plot_type == 'accuracy' else 'train_loss'
        baseline_values = [epoch[baseline_key] for epoch in baseline_data['epochs']]
        if plot_type == 'loss':
            # ax.plot(range(1, len(baseline_values) + 1), baseline_values, label='Baseline', color='red')
            ax.axhline(y=0.019877133469352968, color='sienna', linestyle='--')
        elif plot_type == 'accuracy':
            # ax.plot(range(1, len(baseline_values) + 1), baseline_values, label='Baseline', color='red')
            ax.axhline(y=91.03, color='sienna', linestyle='--')

        # Setting labels and titles
        ylabel = 'Accuracy (%)' if plot_type == 'accuracy' else 'Loss'
        ax.set_xlabel('Communication Round')
        ax.set_ylabel(ylabel)
        ax.set_xlim([1, 200])
        ax.legend(loc='lower right')
        ax.grid(True)

        # Save the plot as SVG
        fig.savefig(f'plots/round_epoch_effect_{metric}_{plot_type}_over_epochs.pdf', format='pdf')
        plt.close(fig)


# Load the data
csv_path = 'updated_cfr_epoc_round_loss_acc.csv'
json_path = 'files/training_metrics_cnn_centralized.json'
df = pd.read_csv(csv_path)
df.set_index('Epoch', inplace=True)

# Load the baseline data
with open(json_path, 'r') as file:
    baseline_data = json.load(file)

# Define the metrics and the groupings by rounds and epochs
metrics = ['BC_BFV', 'BC_CKKS', 'FA_CKKS', 'GBFA_CKKS', 'FAWE']
rounds_epochs = ['200r_1e', '40r_5e', '150r_5e']

# Generate the plots for accuracy and loss
plot_accuracy_loss(df, baseline_data, metrics, rounds_epochs, 'accuracy')
plot_accuracy_loss(df, baseline_data, metrics, rounds_epochs, 'loss')
