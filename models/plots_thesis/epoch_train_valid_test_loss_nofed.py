import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

def plot_data(file_path, accuracy_label, loss_label):
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Extracting data
    epochs = np.array([item['epoch'] for item in data['epochs']])
    train_accuracy = np.array([item['train_accuracy'] for item in data['epochs']])
    valid_accuracy = np.array([item['valid_accuracy'] for item in data['epochs']])
    loss = np.array([item['loss'] for item in data['epochs']])

    # Handling test accuracy data
    test_epochs = np.array([item['epoch'] for item in data['test_metrics']])
    test_accuracy = np.array([item['test_accuracy'] for item in data['test_metrics']])

    # Function for smoothing curves
    def smooth_curve(x, y):
        x_smooth = np.linspace(x.min(), x.max(), len(x) * 10)  # Dynamically determine smoothing points
        y_smooth = make_interp_spline(x, y)(x_smooth)
        return x_smooth, y_smooth

    # Smoothing data
    train_acc_smooth_x, train_acc_smooth_y = smooth_curve(epochs, train_accuracy)
    valid_acc_smooth_x, valid_acc_smooth_y = smooth_curve(epochs, valid_accuracy)
    loss_smooth_x, loss_smooth_y = smooth_curve(epochs, loss)

    # Plotting accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(train_acc_smooth_x, train_acc_smooth_y, label='Train Accuracy', color='blue')
    plt.plot(valid_acc_smooth_x, valid_acc_smooth_y, label='Validation Accuracy', color='green')

    # Plotting last epoch test accuracy
    if len(test_epochs) > 0:
        plt.scatter(test_epochs[-1], test_accuracy[-1], label='Test Accuracy (last epoch)', color='red', marker='o')

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Epoch vs Accuracy: {accuracy_label}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'nofed/{accuracy_label}_accuracy_plot.svg', format='svg')
    plt.show()

    # Plotting loss
    plt.figure(figsize=(10, 6))
    plt.plot(loss_smooth_x, loss_smooth_y, label='Loss', color='purple')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Epoch vs Loss: {loss_label}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'nofed/{loss_label}_loss_plot.svg', format='svg')
    plt.show()

# no fed data
plot_data('nofed/client_0_metrics_cnn_300ep.json', 'NoFed ResNet: Client 0', 'NoFed ResNet: Client 0')
plot_data('nofed/client_1_metrics_cnn_300ep.json', 'NoFed ResNet: Client 1', 'NoFed ResNet: Client 1')
plot_data('nofed/client_0_balnmp_metrics.json', 'NoFed BALNMP: Client 0', 'NoFed BALNMP: Client 0')
plot_data('nofed/client_1_balnmp_metrics.json', 'NoFed BALNMP: Client 1', 'NoFed BALNMP: Client 1')