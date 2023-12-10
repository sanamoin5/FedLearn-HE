import pandas as pd
import matplotlib.pyplot as plt
import json

plt.rcParams.update({
    'font.size': 17,  # Choose a size that fits well in your document
    'font.family': 'Arial',  # Or 'Times New Roman', or another
    'xtick.labelsize': 17,  # Smaller size for axis ticks if needed
    'ytick.labelsize': 17
})

def plot_accuracy_loss(df, baseline_data, metric, rounds_epochs, plot_type):
    fig, ax = plt.subplots(figsize=(14, 7))

    # Define colors for plotting
    colors = {
        'accuracy_50r_1e': 'cyan',
        'accuracy_50r_5e': 'darkorange',
        'accuracy_10r_5e': 'slateblue',
        'loss_50r_1e': 'cyan',
        'loss_50r_5e': 'darkorange',
        'loss_10r_5e': 'slateblue',
    }
    # Plot for each rounds_epochs group
    for r_e in rounds_epochs:
        label = f'BALNMP_{r_e}_{metric} - round_{plot_type}'
        if label in df.columns:
            color_label = f'{plot_type}_{r_e}'
            if plot_type=='accuracy':
                ax.plot(df.index, df[label] * 100 if plot_type == 'accuracy' else df[label],
                        label=f'{r_e} Accuracy', color=colors[color_label])
                last_epoch = df[label].last_valid_index()
            else:
                ax.plot(df.index, df[label] * 100 if plot_type == 'accuracy' else df[label],
                        label=f'{r_e} Loss', color=colors[color_label])
                last_epoch = df[label].last_valid_index()
            ax.axvline(x=last_epoch, color=colors[color_label], linestyle='dotted')

    # Plot the baseline data
    baseline_key = 'val_accuracy' if plot_type == 'accuracy' else 'train_loss'
    baseline_values = [epoch[baseline_key] for epoch in baseline_data['epochs']][:50]
    if baseline_key=='val_accuracy':
        # ax.plot(range(1, len(baseline_values) + 1), baseline_values,
        #         label='Baseline Accuracy', color='red')
        ax.axhline(y=82.13, color='sienna', linestyle='--')
    else:
        # ax.plot(range(1, len(baseline_values) + 1), baseline_values,
        #         label='Baseline Loss', color='red')
        ax.axhline(y=92.64574242490943, color='sienna', linestyle='--')

    # Setting labels and titles
    ylabel = 'Accuracy (%)' if plot_type == 'accuracy' else 'Loss'
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.set_xlim([1, 50])  # Limit x-axis to 200 epochs
    ax.legend()
    ax.grid(True)

    # Save the plot as SVG
    fig.savefig(f'plots/round_epoch_{metric}_{plot_type}_over_epochs_balnmp.pdf', format='pdf')
    plt.close(fig)


# Load the data
csv_path = 'updated_balnmp_epoch_round_loss_acc.csv'
json_path = 'files/training_metrics_balnmp_centralized.json'
df = pd.read_csv(csv_path)
df.set_index('Epoch', inplace=True)

# Load the baseline data
with open(json_path, 'r') as file:
    baseline_data = json.load(file)

# Define the metrics and the groupings by rounds and epochs
metrics = ['FAWE']
rounds_epochs = ['50r_1e', '50r_5e', '10r_5e']

# Generate the plots for accuracy and loss
plot_accuracy_loss(df, baseline_data, metrics[0], rounds_epochs, 'accuracy')
plot_accuracy_loss(df, baseline_data, metrics[0], rounds_epochs, 'loss')
