"""Test the app functions"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

# Define model
class CNN_MNIST(nn.Module):
    def __init__(self):
        super(CNN_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x

# Test if model loads
try:
    model = CNN_MNIST()
    model.load_state_dict(torch.load('best_mnist_cnn.pth', map_location='cpu'))
    model.eval()
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Model loading error: {e}")
    exit(1)

# Simulate canvas image (RGBA format like streamlit_drawable_canvas)
canvas_img = np.zeros((400, 400, 4), dtype=np.uint8)
# Draw a simple "7" in the alpha channel (white on black)
canvas_img[100:150, 150:300, 3] = 255  # Horizontal top
canvas_img[100:300, 270:300, 3] = 255  # Diagonal down

print("\n✓ Created simulated canvas image")
print(f"  Shape: {canvas_img.shape}")

# Test preprocessing
img = canvas_img[:, :, -1]  # Alpha channel
print(f"✓ Extracted alpha channel: {img.shape}, max={img.max()}, non-zero={np.count_nonzero(img)}")

coords = cv2.findNonZero(img)
if coords is None:
    print("✗ No drawing found!")
    exit(1)

x, y, w, h = cv2.boundingRect(coords)
print(f"✓ Bounding box: x={x}, y={y}, w={w}, h={h}")

# Add padding
padding = 20
x = max(0, x - padding)
y = max(0, y - padding)
w = min(img.shape[1] - x, w + 2 * padding)
h = min(img.shape[0] - y, h + 2 * padding)

img_cropped = img[y:y+h, x:x+w]
img_resized = cv2.resize(img_cropped, (20, 20), interpolation=cv2.INTER_AREA)

img_centered = np.zeros((28, 28), dtype=np.uint8)
img_centered[4:24, 4:24] = img_resized

img_normalized = img_centered.astype('float32') / 255.0
img_tensor = torch.from_numpy(img_normalized).unsqueeze(0).unsqueeze(0)

print(f"✓ Preprocessed image: {img_tensor.shape}")

# Make prediction
with torch.no_grad():
    output = model(img_tensor)
    probabilities = F.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][predicted_class].item()

print(f"\n🎯 PREDICTION: {predicted_class}")
print(f"   Confidence: {confidence*100:.2f}%")
print(f"   All probabilities: {probabilities[0].numpy()}")
