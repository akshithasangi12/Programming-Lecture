# Programming-Lecture
# MNIST Classification with TensorFlow and PyTorch

This project demonstrates training a simple neural network for handwritten digit classification using the MNIST dataset. The code includes implementations in:
-  TensorFlow with Keras API
-  PyTorch
-  Model export to TFLite and ONNX formats

---
 
### Install required packages:

Exported Models
TFLite: model.tflite (from TensorFlow Keras model)
ONNX: model.onnx (from PyTorch model)

1. TensorFlow Keras (with .fit() training)

-Trains a simple MLP on MNIST
-Evaluates accuracy
-Exports the model to model.tflite

2.PyTorch Version

Trains a similar MLP using the MNIST dataset
Evaluates test accuracy
Exports the model to model.onnx

3. TensorFlow Custom Training Loop
Uses tf.data.Dataset and a custom training loop with tf.function
Displays loss and accuracy every 100 steps

Evaluates test accuracy

