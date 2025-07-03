# Programming-Lecture
Project Overview: TensorFlow vs PyTorch on MNIST Classification
This project presents a comparative implementation of a basic neural network using TensorFlow and PyTorch, two of the most widely used deep learning frameworks. The model is trained and evaluated on the MNIST dataset, a classic benchmark for handwritten digit recognition.

Objective

The goal of this project is to:
Implement the same neural network architecture in both TensorFlow and PyTorch.
Train the models on MNIST and compare:
Code structure and developer experience
Training and inference performance
Ease of model export (TFLite and ONNX)
Export both trained models for potential deployment on edge devices or cross-platform environments.

Model Architecture

Input Layer: 28x28 grayscale image, flattened to 784 features
Hidden Layer: Fully connected (Dense/Linear) with 64 neurons + ReLU activation
Output Layer: 10 neurons (for 10 digit classes)
TensorFlow: softmax activation
PyTorch: raw logits passed to CrossEntropyLoss


1. Install Requirements
pip install tensorflow torch torchvision

2. Run TensorFlow Script
python mnist_tensorflow.py

3. Run PyTorch Script
python mnist_pytorch.py

Both scripts will:
Load and normalize MNIST data
Train the model for 5 epochs
Evaluate on the test set
Save an exported model file (.tflite or .onnx)


Table 1: Training and Inference Results
Framework Training Time (s) Test Accuracy Inference Time (s)
TensorFlow ˜40.91 s            ˜0.9725       ˜1.86 s
PyTorch    ˜81.75 s             ˜0.9705       ˜1.91 s
