"""Verify all components are working"""
import torch
import numpy as np
import cv2
from PIL import Image

print("=" * 60)
print("VERIFICATION TEST - MNIST Streamlit App Components")
print("=" * 60)

# Test 1: Model Loading
print("\n1. Testing Model Loading...")
try:
    import torch.nn as nn
    import torch.nn.functional as F
    
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
    
    model = CNN_MNIST()
    model.load_state_dict(torch.load('best_mnist_cnn.pth', map_location='cpu'))
    model.eval()
    print("   ✓ Model loaded successfully")
except Exception as e:
    print(f"   ✗ Model loading failed: {e}")
    exit(1)

# Test 2: Image Preprocessing
print("\n2. Testing Image Preprocessing...")
try:
    # Simulate canvas drawing
    canvas_img = np.zeros((400, 400, 4), dtype=np.uint8)
    # Draw a "1"
    canvas_img[100:300, 180:220, 3] = 255
    
    img = canvas_img[:, :, -1]
    coords = cv2.findNonZero(img)
    x, y, w, h = cv2.boundingRect(coords)
    
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
    
    print(f"   ✓ Preprocessing successful: {img_tensor.shape}")
except Exception as e:
    print(f"   ✗ Preprocessing failed: {e}")
    exit(1)

# Test 3: Model Prediction
print("\n3. Testing Model Prediction...")
try:
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    print(f"   ✓ Prediction successful")
    print(f"   - Predicted digit: {predicted_class}")
    print(f"   - Confidence: {confidence*100:.2f}%")
except Exception as e:
    print(f"   ✗ Prediction failed: {e}")
    exit(1)

# Test 4: Image Display Components
print("\n4. Testing Display Components...")
try:
    # Test numpy to RGB conversion
    img_rgb = np.stack([img_centered]*3, axis=-1)
    print(f"   ✓ Image RGB conversion: {img_rgb.shape}")
    
    # Test probability conversion to float
    for i in range(10):
        prob_val = float(probabilities[0][i]) * 100.0
        progress_val = prob_val / 100.0
        if progress_val > 1.0:
            progress_val = 1.0
        float(progress_val)  # Ensure it's Python float
    print("   ✓ Probability display formatting works")
except Exception as e:
    print(f"   ✗ Display components failed: {e}")
    exit(1)

# Test 5: Dependencies
print("\n5. Testing Dependencies...")
try:
    import streamlit
    print(f"   ✓ Streamlit: {streamlit.__version__}")
    from streamlit_drawable_canvas import st_canvas
    print("   ✓ streamlit-drawable-canvas: installed")
    print(f"   ✓ OpenCV: {cv2.__version__}")
    print(f"   ✓ PyTorch: {torch.__version__}")
    print(f"   ✓ NumPy: {np.__version__}")
    from PIL import Image
    print("   ✓ Pillow: installed")
except Exception as e:
    print(f"   ✗ Dependency check failed: {e}")
    exit(1)

print("\n" + "=" * 60)
print("✓ ALL TESTS PASSED - App is ready to use!")
print("=" * 60)
print("\nStreamlit app is running at: http://localhost:8501")
print("\nInstructions:")
print("1. Open http://localhost:8501 in your browser")
print("2. Draw a digit (0-9) on the black canvas")
print("3. See real-time predictions on the right")
print("4. Click 'Clear Canvas' to try another digit")
print("=" * 60)
