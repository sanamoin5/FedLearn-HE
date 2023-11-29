import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_data(round_loss_file, round_accuracy_file, local_eval_accuracy_file, test_accuracy, label):
    # Reading CSV files
    round_loss = pd.read_csv(round_loss_file)
    round_accuracy = pd.read_csv(round_accuracy_file)
    local_eval_accuracy = pd.read_csv(local_eval_accuracy_file)

    # Converting steps to epochs (3 steps per epoch)
    round_loss['Epoch'] = round_loss['Step'] // 3
    round_accuracy['Epoch'] = round_accuracy['Step'] // 3
    local_eval_accuracy['Epoch'] = local_eval_accuracy['Step'] // 3

    # Extracting data
    loss = round_loss.groupby('Epoch').mean()
    accuracy = round_accuracy.groupby('Epoch').mean()
    local_eval = local_eval_accuracy.groupby('Epoch').mean()

    # Plotting epoch vs. round loss
    plt.figure(figsize=(10, 6))
    plt.plot(loss.index, loss.iloc[:, 0], label='Round Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Epoch vs Round Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'round_loss_plot_{label}.svg', format='svg')
    plt.show()

    # Plotting epoch vs. round accuracy and local eval accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(accuracy.index, accuracy.iloc[:, 0], label='Round Accuracy', color='green')
    plt.plot(local_eval.index, local_eval.iloc[:, 0], label='Local Eval Accuracy', color='orange')

    # Plotting hardcoded test accuracy at the last epoch
    if len(accuracy.index) > 0:
        plt.scatter(accuracy.index[-1], test_accuracy, label='Test Accuracy (last epoch)', color='red', marker='o')

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Epoch vs Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'accuracy_plot_{label}.svg', format='svg')
    plt.show()


# Example usage
plot_data('nofed/cfr_roundloss_2cl_fawe.csv', 'nofed/cfr_roundacc_2cl_fawe.csv', 'nofed/cfr_localevalacc_2cl_fawe.csv',
          91.58, 'CFR Fed')
plot_data('nofed/balnmp_roundloss_2cl_fawe.csv', 'nofed/balnmp_roundacc_2cl_fawe.csv', 'nofed/balnmp_localevalacc_2cl_fawe.csv',
          80.23, 'BALNMP Fed')
