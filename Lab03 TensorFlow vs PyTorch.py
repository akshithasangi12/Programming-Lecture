import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import time

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Define model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train
start = time.time()
model.fit(x_train, y_train, epochs=5, batch_size=32)
end = time.time()
print(f"TF Training time: {end-start:.2f} seconds")

# Evaluate
model.evaluate(x_test, y_test)

# Export to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)


!pip install onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

# Data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # flatten 28x28 → 784
])

train_loader = DataLoader(
    datasets.MNIST(root='./data', train=True, download=True, transform=transform),
    batch_size=32, shuffle=True)
test_loader = DataLoader(
    datasets.MNIST(root='./data', train=False, download=True, transform=transform),
    batch_size=1000)

# Define model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

model = Net()
optimizer = optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()

# Train
start = time.time()
for epoch in range(5):
    for x, y in train_loader:
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
end = time.time()
print(f"PyTorch Training time: {end - start:.2f} seconds")

# Evaluate
model.eval()
correct = 0
with torch.no_grad():
    for x, y in test_loader:
        output = model(x)
        pred = output.argmax(1)
        correct += (pred == y).sum().item()
print(f"Test accuracy: {correct / len(test_loader.dataset):.4f}")

# Export to ONNX
import onnx  # Ensure ONNX is installed: pip install onnx
dummy_input = torch.randn(1, 784)
torch.onnx.export(model, dummy_input, "model.onnx",
                  input_names=["input"], output_names=["output"])


import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import time

# Load and preprocess
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

batch_size = 32
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

# Define model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Optimizer, Loss, Metrics
loss_fn = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
test_acc_metric = tf.keras.metrics.CategoricalAccuracy()

# Eager training loop
@tf.function  # Optional: speeds up training using graph mode
def train_step(x_batch, y_batch):
    with tf.GradientTape() as tape:
        logits = model(x_batch, training=True)
        loss = loss_fn(y_batch, logits)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_acc_metric.update_state(y_batch, logits)
    return loss

# Training
epochs = 4
start = time.time()
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    for step, (x_batch, y_batch) in enumerate(train_dataset):
        loss = train_step(x_batch, y_batch)
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.numpy():.4f}, Accuracy: {train_acc_metric.result().numpy():.4f}")
    print(f"Training Accuracy for epoch {epoch+1}: {train_acc_metric.result().numpy():.4f}")
    train_acc_metric.reset_state()
end = time.time()
print(f"\nTF Training time: {end - start:.2f} seconds")

# Evaluate
for x_batch, y_batch in test_dataset:
    test_logits = model(x_batch, training=False)
    test_acc_metric.update_state(y_batch, test_logits)

print(f"Test Accuracy: {test_acc_metric.result().numpy():.4f}")
