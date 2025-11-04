"""
BUGGY TENSORFLOW MNIST CODE
This code contains several common errors that need to be debugged:
1. Dimension mismatches in model architecture
2. Incorrect loss function for the task
3. Wrong optimizer configuration
4. Data preprocessing issues
5. Incorrect metric tracking
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Set random seed
tf.random.set_seed(42)
np.random.seed(42)

# Load MNIST data
print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# BUG 1: Incorrect data preprocessing - not normalizing properly
# Data should be normalized to [0, 1] but here it's left as [0, 255]
x_train = x_train.astype('float32')  # Missing division by 255
x_test = x_test.astype('float32')    # Missing division by 255

# BUG 2: Wrong data shape - CNN expects (batch, height, width, channels)
# Not expanding dimensions to add channel dimension
print(f"Training data shape: {x_train.shape}")  # Will be (60000, 28, 28) instead of (60000, 28, 28, 1)

# BUG 3: Labels are not one-hot encoded but using categorical crossentropy
# y_train and y_test should be one-hot encoded or use sparse_categorical_crossentropy
print(f"Label shape: {y_train.shape}")  # Will be (60000,) instead of (60000, 10)

# Build CNN model
print("\nBuilding CNN model...")
model = keras.Sequential([
    # BUG 4: Input shape mismatch - expecting wrong dimensions
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28)),  # Missing channel dimension
    layers.MaxPooling2D((2, 2)),
    
    # BUG 5: Dimension mismatch in conv layers
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    layers.Flatten(),
    
    # BUG 6: Wrong number of units in dense layer
    layers.Dense(128, activation='relu'),
    
    # BUG 7: Using softmax but with wrong loss function
    layers.Dense(10, activation='softmax')
])

# BUG 8: Wrong loss function - using binary_crossentropy for multi-class classification
# Should use categorical_crossentropy or sparse_categorical_crossentropy
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',  # WRONG! This is for binary classification
    metrics=['accuracy']
)

# BUG 9: Wrong optimizer learning rate - too high
# Default Adam learning rate of 0.001 might be too aggressive
optimizer = keras.optimizers.Adam(learning_rate=0.1)  # Too high!
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nModel Summary:")
model.summary()

# BUG 10: Missing validation split
# Not monitoring validation performance during training
print("\nTraining model...")
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=64,
    verbose=1
    # Missing validation_split or validation_data
)

# BUG 11: Incorrect evaluation
# Evaluating on training data instead of test data
print("\nEvaluating model...")
test_loss, test_accuracy = model.evaluate(x_train, y_train, verbose=0)  # Should use x_test, y_test
print(f"Test accuracy: {test_accuracy:.4f}")

# BUG 12: Prediction shape mismatch
# Not reshaping single prediction correctly
sample = x_test[0]  # Shape: (28, 28) but needs (1, 28, 28, 1)
prediction = model.predict(sample)  # This will cause error
print(f"Prediction: {np.argmax(prediction)}")

print("\nTraining complete!")
