"""
Convert PyTorch MNIST model to ONNX format for lightweight deployment
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# CNN Model Definition (same as in mnist_streamlit_app.py)
class CNN_MNIST(nn.Module):
    """CNN model for MNIST classification"""
    
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

def convert_to_onnx():
    """Convert PyTorch model to ONNX format"""
    print("Loading PyTorch model...")
    device = torch.device('cpu')
    model = CNN_MNIST().to(device)
    
    try:
        model.load_state_dict(torch.load('best_mnist_cnn.pth', map_location=device))
        print("✓ Model loaded successfully")
    except FileNotFoundError:
        print("❌ Model file 'best_mnist_cnn.pth' not found!")
        return
    
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 1, 28, 28, device=device)
    
    # Export to ONNX
    output_path = "mnist_model.onnx"
    print(f"Converting to ONNX format: {output_path}")
    
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            verbose=False
        )
    
    print(f"✓ Model successfully converted to {output_path}")
    print(f"✓ File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    
    # Test the ONNX model
    print("\nTesting ONNX model...")
    import onnxruntime as ort
    
    ort_session = ort.InferenceSession(output_path)
    
    # Run inference
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    
    print("✓ ONNX model test successful!")
    print(f"✓ Output shape: {ort_outs[0].shape}")

if __name__ == "__main__":
    import os
    convert_to_onnx()
