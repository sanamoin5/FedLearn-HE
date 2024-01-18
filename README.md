# FedLearn-HE: Federated Learning with Homomorphic Encryption in Healthcare

## Overview
FedLearn-HE is a project that demonstrates the integration of Federated Learning (FL) and Homomorphic Encryption (HE) techniques in the context of neural network models, with a focus on applications in healthcare. This project incorporates advanced FL and HE methods to secure data classification tasks while ensuring privacy and efficient communication.

## Features
- **Neural Network Models**: Includes implementations of models like ResNet18 and custom convolutional neural networks for benchmark datasets such as CIFAR-10 and MNIST.
- **Datasets**: Employs CIFAR-10 and MNIST datasets for demonstration.
- **Security and Efficiency**: Showcases the application of HE for secure aggregation in FL.

## Installation and Setup
1. Clone the repository.
2. Set up the environment using the provided `environment.yml` file.
3. Run `main.py` to start the program.

## Usage
The application starts with `main.py`, which initializes the Federated Learning process. The core functionalities include:

- **Configurable Parameters**: The application reads command-line arguments for various settings, including the choice of federated learning algorithm, neural network model, homomorphic encryption scheme, and other parameters.
- **Client Initialization**: Based on the input parameters, the client is set up to determine the neural network model and the configuration for FL and HE.
- **Model Selection**: Supports various neural network models such as MNIST CNN, CIFAR ResNet, etc., located in `models/nn_models`.
- **Homomorphic Encryption**: Implements HE schemes (`models/homomorphic_encryption.py`) for secure data handling.

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.

