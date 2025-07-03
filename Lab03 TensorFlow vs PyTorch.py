TensorFlow

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import time

print("Starting TensorFlow training...")

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0   # Normalize to [0, 1]
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
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

loss_fn = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
train_acc_metric = tf.keras.metrics.CategoricalAccuracy()

@tf.function
def train_step(x_batch, y_batch):
    with tf.GradientTape() as tape:
        logits = model(x_batch, training=True)
        loss = loss_fn(y_batch, logits)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_acc_metric.update_state(y_batch, logits)
    return loss

epochs = 5
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

# Compile the model for evaluation
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

# Evaluation with model.evaluate() and inference time measurement
print("\nTensorFlow Evaluation with timing:")
start_tf_infer = time.time()
test_loss, test_accuracy = model.evaluate(test_dataset, verbose=2)
end_tf_infer = time.time()
print(f"TensorFlow Test Accuracy: {test_accuracy:.4f}")
print(f"TensorFlow Inference Time: {end_tf_infer - start_tf_infer:.2f} seconds")

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("model_tf.tflite", "wb") as f:
    f.write(tflite_model)


PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

print("\nStarting PyTorch training...")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data loading and preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),          # Converts to tensor [0,1]
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

# Define PyTorch model (simple MLP like TF model)
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        logits = self.fc2(x)
        return logits

model_pt = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_pt.parameters(), lr=0.001)

# Training loop
epochs = 5
start = time.time()
for epoch in range(epochs):
    model_pt.train()
    running_loss = 0.0
    correct = 0
    total = 0
    print(f"\nEpoch {epoch+1}/{epochs}")
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model_pt(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if i % 100 == 0:
            print(f"Step {i}, Loss: {loss.item():.4f}, Accuracy: {correct/total:.4f}")

    epoch_acc = correct / total
    print(f"Training Accuracy for epoch {epoch+1}: {epoch_acc:.4f}")

end = time.time()
print(f"\nPyTorch Training time: {end - start:.2f} seconds")

# Evaluation loop with timing
print("\nPyTorch Evaluation with timing:")
model_pt.eval()
correct = 0
total = 0
start_pt_infer = time.time()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model_pt(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
end_pt_infer = time.time()

pt_accuracy = correct / total
pt_inference_time = end_pt_infer - start_pt_infer
print(f"PyTorch Test Accuracy: {pt_accuracy:.4f}")
print(f"PyTorch Inference Time: {pt_inference_time:.2f} seconds")

# Save PyTorch model
torch.save(model_pt.state_dict(), "model_pytorch.pth")
