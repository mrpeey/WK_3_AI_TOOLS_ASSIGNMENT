"""
FIXED TENSORFLOW MNIST CODE
This is the corrected version with all bugs fixed and detailed explanations.

FIXES APPLIED:
1. ✓ Proper data normalization
2. ✓ Correct data shape with channel dimension
3. ✓ Appropriate loss function
4. ✓ Proper optimizer configuration
5. ✓ Validation data monitoring
6. ✓ Correct evaluation and prediction
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

print("="*80)
print("FIXED TENSORFLOW MNIST CNN")
print("="*80)

# Load MNIST data
print("\n[1] Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print(f"   Original training data shape: {x_train.shape}")
print(f"   Original label shape: {y_train.shape}")

# FIX 1: Proper data normalization
# Normalize pixel values from [0, 255] to [0, 1]
print("\n[2] Normalizing data...")
x_train = x_train.astype('float32') / 255.0  # FIXED: Divide by 255
x_test = x_test.astype('float32') / 255.0    # FIXED: Divide by 255
print(f"   ✓ Data normalized to range [0, 1]")
print(f"   Sample pixel range: [{x_train.min():.2f}, {x_train.max():.2f}]")

# FIX 2: Add channel dimension for CNN
# CNN expects shape: (batch_size, height, width, channels)
print("\n[3] Reshaping data for CNN...")
x_train = np.expand_dims(x_train, axis=-1)  # FIXED: Add channel dimension
x_test = np.expand_dims(x_test, axis=-1)    # FIXED: Add channel dimension
print(f"   ✓ Training data shape: {x_train.shape}")  # Now (60000, 28, 28, 1)
print(f"   ✓ Test data shape: {x_test.shape}")      # Now (10000, 28, 28, 1)

# FIX 3: Labels are already in correct format for sparse_categorical_crossentropy
# y_train and y_test are integers [0-9], which works with sparse_categorical_crossentropy
print("\n[4] Checking label format...")
print(f"   Label dtype: {y_train.dtype}")
print(f"   Label range: [{y_train.min()}, {y_train.max()}]")
print(f"   ✓ Labels are integers (suitable for sparse_categorical_crossentropy)")

# Build CNN model with correct architecture
print("\n[5] Building CNN model...")
model = keras.Sequential([
    # FIX 4: Correct input shape with channel dimension
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), name='conv1'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2), name='pool1'),
    
    # Additional conv layers with proper dimensions
    layers.Conv2D(64, (3, 3), activation='relu', name='conv2'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2), name='pool2'),
    
    layers.Conv2D(128, (3, 3), activation='relu', name='conv3'),
    layers.BatchNormalization(),
    
    # Flatten before dense layers
    layers.Flatten(name='flatten'),
    
    # Dense layers with dropout for regularization
    layers.Dense(128, activation='relu', name='dense1'),
    layers.Dropout(0.5, name='dropout1'),
    
    # Output layer: 10 units for 10 digit classes
    layers.Dense(10, activation='softmax', name='output')
], name='MNIST_CNN')

# FIX 5 & 6: Correct loss function and proper optimizer
print("\n[6] Compiling model...")
# FIXED: Use sparse_categorical_crossentropy for integer labels
# FIXED: Use appropriate learning rate
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),  # Proper learning rate
    loss='sparse_categorical_crossentropy',  # FIXED: Correct loss for integer labels
    metrics=['accuracy']
)

print("   ✓ Model compiled successfully")
print(f"   Loss function: sparse_categorical_crossentropy")
print(f"   Optimizer: Adam (lr=0.001)")

print("\n[7] Model Summary:")
print("-"*80)
model.summary()
print("-"*80)

# Count parameters
total_params = model.count_params()
print(f"\nTotal parameters: {total_params:,}")

# FIX 7: Add validation split for monitoring
print("\n[8] Training model with validation monitoring...")
print("-"*80)

# Define callbacks for better training
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        verbose=1,
        min_lr=1e-6
    )
]

# FIXED: Include validation data
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.1,  # FIXED: Use 10% of training data for validation
    callbacks=callbacks,
    verbose=1
)

# FIX 8: Evaluate on correct test data
print("\n[9] Evaluating model on TEST data...")
print("-"*80)
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)  # FIXED: Use test data
print(f"   Test Loss: {test_loss:.4f}")
print(f"   Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

if test_accuracy >= 0.95:
    print("   ✓ Goal achieved: Test accuracy > 95%")
else:
    print(f"   ⚠ Test accuracy {test_accuracy*100:.2f}% < 95%")

# FIX 9: Correct prediction with proper reshaping
print("\n[10] Testing predictions...")
print("-"*80)

# Make predictions on a few samples
num_samples = 5
sample_indices = np.random.choice(len(x_test), num_samples, replace=False)

print(f"\nPredicting {num_samples} random samples:")
for i, idx in enumerate(sample_indices, 1):
    # FIXED: Proper reshaping for single prediction
    sample = x_test[idx]
    sample_input = np.expand_dims(sample, axis=0)  # Shape: (1, 28, 28, 1)
    
    # Make prediction
    prediction_probs = model.predict(sample_input, verbose=0)
    predicted_class = np.argmax(prediction_probs[0])
    confidence = prediction_probs[0][predicted_class]
    true_label = y_test[idx]
    
    # Check if correct
    status = "✓" if predicted_class == true_label else "✗"
    
    print(f"   Sample {i}: True={true_label}, Predicted={predicted_class}, "
          f"Confidence={confidence:.2%} {status}")

# Plot training history
print("\n[11] Generating training plots...")
print("-"*80)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot accuracy
axes[0].plot(history.history['accuracy'], label='Training Accuracy', marker='o')
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s')
axes[0].set_title('Model Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True)

# Plot loss
axes[1].plot(history.history['loss'], label='Training Loss', marker='o')
axes[1].plot(history.history['val_loss'], label='Validation Loss', marker='s')
axes[1].set_title('Model Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('fixed_model_training.png', dpi=150, bbox_inches='tight')
print("   ✓ Training plots saved as 'fixed_model_training.png'")

# Save the model
print("\n[12] Saving model...")
model.save('mnist_cnn_fixed.h5')
model.save('mnist_cnn_fixed_savedmodel', save_format='tf')
print("   ✓ Model saved as 'mnist_cnn_fixed.h5'")
print("   ✓ Model saved as 'mnist_cnn_fixed_savedmodel/' (TensorFlow SavedModel format)")

# Summary of fixes
print("\n" + "="*80)
print("SUMMARY OF FIXES APPLIED")
print("="*80)
print("""
1. ✓ Data Normalization: Divided pixel values by 255 to get range [0, 1]
2. ✓ Shape Correction: Added channel dimension using np.expand_dims()
3. ✓ Loss Function: Changed from 'binary_crossentropy' to 'sparse_categorical_crossentropy'
4. ✓ Input Shape: Corrected input_shape to (28, 28, 1) with channel dimension
5. ✓ Optimizer: Used appropriate learning rate (0.001 instead of 0.1)
6. ✓ Validation: Added validation_split=0.1 to monitor overfitting
7. ✓ Evaluation: Used test data (x_test, y_test) instead of training data
8. ✓ Prediction: Properly reshaped single samples with np.expand_dims()
9. ✓ Callbacks: Added EarlyStopping and ReduceLROnPlateau for better training
10. ✓ Regularization: Added BatchNormalization and Dropout layers
""")

print("\n" + "="*80)
print("DEBUGGING TIPS FOR TENSORFLOW")
print("="*80)
print("""
Common TensorFlow Errors and Solutions:

1. DIMENSION MISMATCH:
   Error: "Dimensions must be equal"
   Fix: Check input/output shapes at each layer using model.summary()
   
2. LOSS FUNCTION MISMATCH:
   Error: "logits and labels must have the same shape"
   Fix: Use sparse_categorical_crossentropy for integer labels
        Use categorical_crossentropy for one-hot encoded labels
        Use binary_crossentropy only for binary classification

3. SHAPE ERRORS IN PREDICTION:
   Error: "Input 0 is incompatible with layer"
   Fix: Add batch dimension with np.expand_dims(x, axis=0)

4. LEARNING RATE ISSUES:
   Symptom: Loss explodes or doesn't decrease
   Fix: Reduce learning rate (try 0.001, 0.0001)
        Use learning rate scheduling

5. OVERFITTING:
   Symptom: Training accuracy high, validation accuracy low
   Fix: Add Dropout layers, use L2 regularization
        Increase training data or use data augmentation

6. UNDERFITTING:
   Symptom: Both training and validation accuracy low
   Fix: Increase model complexity (more layers/units)
        Train for more epochs
        Check if data is properly preprocessed

7. MEMORY ERRORS:
   Error: "OOM when allocating tensor"
   Fix: Reduce batch size
        Use gradient accumulation
        Use mixed precision training

8. GRADIENT ISSUES:
   Symptom: Loss becomes NaN
   Fix: Check learning rate (reduce if too high)
        Clip gradients
        Check for division by zero in custom loss
""")

print("\n" + "="*80)
print("✅ ALL FIXES APPLIED SUCCESSFULLY")
print("="*80)
